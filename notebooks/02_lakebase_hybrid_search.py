# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 02 · Hybrid search for grocery dietary queries with Lakebase Search
# MAGIC
# MAGIC We turn the synthetic StreetsMart catalog from notebook 01 into a search engine using
# MAGIC [**Lakebase Search**](https://docs.databricks.com/aws/en/oltp/projects/lakebase-search):
# MAGIC
# MAGIC | Step | Lakebase Search building block |
# MAGIC |---|---|
# MAGIC | 1. Product text vector embeddings in the lakehouse | `ai_query` + a Foundation Model embedding endpoint |
# MAGIC | 2. Serve the Delta table from Postgres | **Synced table** with `type_overrides` (embedding → `vector(1024)`) and a generated `tsvector` extra column |
# MAGIC | 3. Vector (semantic) search | `lakebase_vector` → **`lakebase_ann`** index, `<=>` cosine distance |
# MAGIC | 4. Keyword search | `lakebase_text` → **`lakebase_bm25`** index, `<@>` + `to_bm25query()` |
# MAGIC | 5. Hybrid search | **Reciprocal Rank Fusion** over the top 40 of each, as in the docs — plus dietary hard filters |
# MAGIC
# MAGIC **The core idea: plain search is sometimes not enough. Filter first, then rank.** Dietary restrictions are
# MAGIC constraints, not similarity, so they're applied as hard `WHERE` filters that remove unsafe candidates *before* BM25,
# MAGIC vector search and RRF rank what's left. Section 7 shows the difference on the same queries with and without filters.
# MAGIC
# MAGIC **Prerequisites**
# MAGIC * The dataset CSV published with the blog, `grocery_products.csv` (3,128 products). Upload it to a Volume folder
# MAGIC   (default `/Volumes/<catalog>/<schema>/blog_assets/data`) or point `data_location` at a public URL folder.
# MAGIC   Step 0 loads it into a Unity Catalog table. (Notebook `01_generate_grocery_catalog` is optional: it regenerates the
# MAGIC   same data from scratch.)
# MAGIC * A Lakebase project on Postgres 16+ with **Lakebase Search enabled**: *Lakebase project → Settings → Lakebase Search →
# MAGIC   Enable*. Enabling is irreversible and restarts the project's computes, so use a dedicated project for this demo.
# MAGIC   (The cell in step 1 creates the project if it doesn't exist, then stops and asks you to enable search.)

# COMMAND ----------

# MAGIC %pip install -q "psycopg[binary]>=3.2" "databricks-sdk>=0.162.0"
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "rkm_cat", "Catalog")
dbutils.widgets.text("schema", "grocery_search", "Schema")
dbutils.widgets.text("lakebase_project", "grocery-search", "Lakebase project id")
dbutils.widgets.text("lakebase_branch", "production", "Lakebase branch")
dbutils.widgets.text("lakebase_database", "databricks_postgres", "Postgres database")
dbutils.widgets.text("embedding_endpoint", "databricks-gte-large-en", "Embedding endpoint")
dbutils.widgets.text("data_location", "", "Dataset CSV folder: Volume path or https URL (default: /Volumes/<catalog>/<schema>/blog_assets/data)")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")            # also becomes the Postgres schema of the synced table
PROJECT_ID = dbutils.widgets.get("lakebase_project")
BRANCH_ID = dbutils.widgets.get("lakebase_branch")
PG_DATABASE = dbutils.widgets.get("lakebase_database")
EMBEDDING_ENDPOINT = dbutils.widgets.get("embedding_endpoint")
DATA_LOCATION = (dbutils.widgets.get("data_location") or f"/Volumes/{CATALOG}/{SCHEMA}/blog_assets/data").rstrip("/")
EMBEDDING_DIM = 1024  # databricks-gte-large-en

SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.grocery_products_search"   # Delta: products + embedding + search_text
SYNCED_TABLE = f"{CATALOG}.{SCHEMA}.products_synced"           # UC synced table
PG_TABLE = f"{SCHEMA}.products_synced"                         # ...and its Postgres twin
BM25_INDEX = f"{SCHEMA}.products_bm25"                         # to_bm25query() needs the schema-qualified index name

for name in ("CATALOG", "SCHEMA", "PROJECT_ID", "BRANCH_ID", "PG_DATABASE", "EMBEDDING_ENDPOINT", "EMBEDDING_DIM",
             "DATA_LOCATION", "SOURCE_TABLE", "SYNCED_TABLE", "PG_TABLE", "BM25_INDEX"):
    print(f"{name:<20} {globals()[name]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0 · Load the dataset into Unity Catalog
# MAGIC
# MAGIC The list columns (`allergens`, `may_contain`, `dietary_tags`) are stored as JSON arrays in the CSV.

# COMMAND ----------

import json

import pandas as pd
from pyspark.sql import types as T

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.blog_assets")

try:
    products_csv = pd.read_csv(f"{DATA_LOCATION}/grocery_products.csv", dtype={"upc": str, "sku": str})
except FileNotFoundError as e:
    raise FileNotFoundError(f"Upload grocery_products.csv to {DATA_LOCATION} "
                            "(or set the data_location widget), then re-run.") from e

for c in ("allergens", "may_contain", "dietary_tags"):
    products_csv[c] = products_csv[c].apply(json.loads)

product_schema = T.StructType([
    T.StructField("product_id", T.LongType()), T.StructField("sku", T.StringType()), T.StructField("upc", T.StringType()),
    T.StructField("brand", T.StringType()), T.StructField("product_name", T.StringType()), T.StructField("size", T.StringType()),
    T.StructField("category", T.StringType()), T.StructField("subcategory", T.StringType()),
    T.StructField("archetype", T.StringType()), T.StructField("description", T.StringType()),
    T.StructField("ingredients", T.StringType()), T.StructField("allergens", T.ArrayType(T.StringType())),
    T.StructField("may_contain", T.ArrayType(T.StringType())), T.StructField("dietary_tags", T.ArrayType(T.StringType())),
    T.StructField("price", T.DoubleType()),
])
(spark.createDataFrame(products_csv[[f.name for f in product_schema]].to_dict("records"), product_schema)
 .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.grocery_products"))
print(f"Loaded {len(products_csv):,} products from {DATA_LOCATION}")
display(spark.table(f"{CATALOG}.{SCHEMA}.grocery_products").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Connect to Lakebase and check that Lakebase Search is enabled

# COMMAND ----------

import time

import psycopg
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.postgres import Project, ProjectSpec

w = WorkspaceClient()

try:
    w.postgres.get_project(name=f"projects/{PROJECT_ID}")
    print(f"Using existing Lakebase project '{PROJECT_ID}'")
except NotFound:
    print(f"Creating Lakebase project '{PROJECT_ID}' (takes a minute or two)...")
    w.postgres.create_project(
        project=Project(spec=ProjectSpec(display_name="StreetsMart grocery hybrid search")),
        project_id=PROJECT_ID,
    ).wait()

BRANCH = f"projects/{PROJECT_ID}/branches/{BRANCH_ID}"
endpoint = next(e for e in w.postgres.list_endpoints(parent=BRANCH)
                if "READ_WRITE" in str(e.status.endpoint_type or e.spec.endpoint_type))
PG_HOST = endpoint.status.hosts.host
PG_USER = w.current_user.me().user_name


def connect():
    token = w.postgres.generate_database_credential(endpoint=endpoint.name).token
    return psycopg.connect(host=PG_HOST, dbname=PG_DATABASE, user=PG_USER, password=token,
                           sslmode="require", autocommit=True)


conn = connect()
print(conn.execute("SELECT version()").fetchone()[0])

preload = conn.execute("SHOW shared_preload_libraries").fetchone()[0]
if "lakebase_vector" not in preload or "lakebase_text" not in preload:
    raise RuntimeError(
        f"Lakebase Search is not enabled on project '{PROJECT_ID}'. In the Lakebase UI open the project, go to "
        "Settings → Lakebase Search → Enable Lakebase Search (irreversible; restarts computes), then re-run this notebook.")
print("Lakebase Search is enabled ✔")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Install the Lakebase Search extensions
# MAGIC
# MAGIC Install them **before** creating the synced table: the `vector(1024)` type override needs the vector type to exist in the
# MAGIC destination database, and creating a synced table does not install extensions.

# COMMAND ----------

conn.execute("""
  CREATE EXTENSION IF NOT EXISTS lakebase_vector CASCADE;   -- ANN vector search (installs pgvector)
  CREATE EXTENSION IF NOT EXISTS lakebase_text;             -- BM25 full-text search
""")
EXT_VERSIONS = dict(conn.execute(
    "SELECT extname, extversion FROM pg_extension WHERE extname IN ('lakebase_vector','lakebase_text','vector')").fetchall())
print(EXT_VERSIONS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Prepare the source table in the lakehouse
# MAGIC
# MAGIC `ai_query` calls the embedding endpoint in parallel from Spark. We embed a compact "product card" (name, brand,
# MAGIC category, description, ingredients, dietary tags) and also build `search_text`, the text that BM25 will index.
# MAGIC SKUs and UPCs are deliberately left out of `search_text`: they are matched exactly instead (step 6).

# COMMAND ----------

spark.sql(f"""
  CREATE OR REPLACE TABLE {SOURCE_TABLE} AS
  WITH cards AS (
    SELECT *,
      concat_ws(' ', product_name, '| Brand:', brand, '|', category, '>', subcategory, '|', description,
                '| Ingredients:', ingredients, '| Dietary:', array_join(dietary_tags, ', ')) AS embed_text,
      concat_ws(' ', product_name, brand, category, subcategory, array_join(dietary_tags, ' '),
                description, ingredients) AS search_text
    FROM {CATALOG}.{SCHEMA}.grocery_products
  )
  SELECT * EXCEPT (embed_text), ai_query('{EMBEDDING_ENDPOINT}', embed_text) AS embedding
  FROM cards
""")
display(spark.sql(f"SELECT count(*) AS products, max(size(embedding)) AS dims FROM {SOURCE_TABLE}"))

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {SOURCE_TABLE} LIMIT 10"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Sync it into Lakebase with search-ready column types
# MAGIC
# MAGIC One synced table does the heavy lifting:
# MAGIC * **`type_overrides`** maps `embedding ARRAY<FLOAT>` → `vector(1024)` (instead of the default `JSONB`)
# MAGIC * **`extra_columns`** adds `search_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', search_text)) STORED`
# MAGIC
# MAGIC Arrays such as `allergens` and `dietary_tags` arrive as `JSONB`, which is perfect for hard filters (`?&`, `?|`).
# MAGIC In Postgres the UC schema becomes the schema, so the table is `<schema>.products_synced`.
# MAGIC
# MAGIC We use `SNAPSHOT` mode (full refresh on each sync). For incremental freshness use `TRIGGERED` (needs Change Data Feed on the
# MAGIC source) or `CONTINUOUS`. Re-running this notebook deletes and recreates the synced table.

# COMMAND ----------

from databricks.sdk.service.postgres import (
    NewPipelineSpec,
    SyncedTable,
    SyncedTableSyncedTableSpec,
    SyncedTableSyncedTableSpecExtraColumn,
    SyncedTableSyncedTableSpecExtraColumnMaintenance,
    SyncedTableSyncedTableSpecPgSpecificType,
    SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy,
    SyncedTableSyncedTableSpecTypeOverride,
)

try:
    w.postgres.get_synced_table(name=f"synced_tables/{SYNCED_TABLE}")
    print("Deleting previous synced table...")
    w.postgres.delete_synced_table(name=f"synced_tables/{SYNCED_TABLE}").wait()
except NotFound:
    pass
conn.execute(f"DROP TABLE IF EXISTS {PG_TABLE} CASCADE")

w.postgres.create_synced_table(
    synced_table_id=SYNCED_TABLE,
    synced_table=SyncedTable(spec=SyncedTableSyncedTableSpec(
        source_table_full_name=SOURCE_TABLE,
        branch=BRANCH,
        postgres_database=PG_DATABASE,
        primary_key_columns=["product_id"],
        scheduling_policy=SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy.SNAPSHOT,
        create_database_objects_if_missing=True,
        new_pipeline_spec=NewPipelineSpec(storage_catalog=CATALOG, storage_schema=SCHEMA),
        type_overrides=[SyncedTableSyncedTableSpecTypeOverride(
            column_name="embedding", pg_type=SyncedTableSyncedTableSpecPgSpecificType.PG_SPECIFIC_TYPE_VECTOR,
            size=EMBEDDING_DIM)],
        # Generated-column expressions may only use built-in Postgres objects, so use the built-in 'english' config here
        extra_columns=[SyncedTableSyncedTableSpecExtraColumn(
            column_name="search_tsv", column_type="tsvector",
            compute="to_tsvector('english', search_text)",
            maintenance=SyncedTableSyncedTableSpecExtraColumnMaintenance.STORED_GENERATED)],
    )),
).wait()

# The initial snapshot runs asynchronously after creation
while True:
    status = w.postgres.get_synced_table(name=f"synced_tables/{SYNCED_TABLE}").status
    state = str(status.detailed_state)
    print(time.strftime("%H:%M:%S"), state)
    if "ONLINE" in state:
        break
    if "FAILED" in state:
        raise RuntimeError(f"Sync failed: {status.message}")
    time.sleep(30)

print(conn.execute(f"SELECT count(*), count(search_tsv), count(embedding) FROM {PG_TABLE}").fetchone())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Build the Lakebase Search indexes
# MAGIC
# MAGIC Synced tables are read-only in Postgres, but you can add indexes:
# MAGIC * **`lakebase_ann`** on `embedding` — IVF partitioning + RaBitQ quantization; storage-backed, so it survives scale-to-zero without warmup
# MAGIC * **`lakebase_bm25`** on `search_tsv` — computes corpus statistics at build time, so build it **after** the data is loaded
# MAGIC * GIN indexes on the JSONB `dietary_tags` / `allergens` / `may_contain` arrays for hard filters
# MAGIC * a B-tree on `sku` for exact identifier lookups

# COMMAND ----------

t0 = time.perf_counter()
conn.execute(f"""
  CREATE INDEX IF NOT EXISTS products_emb_ann     ON {PG_TABLE} USING lakebase_ann  (embedding vector_cosine_ops);
  CREATE INDEX IF NOT EXISTS products_bm25        ON {PG_TABLE} USING lakebase_bm25 (search_tsv);
  CREATE INDEX IF NOT EXISTS products_tags_gin    ON {PG_TABLE} USING gin (dietary_tags);
  CREATE INDEX IF NOT EXISTS products_allerg_gin  ON {PG_TABLE} USING gin (allergens);
  CREATE INDEX IF NOT EXISTS products_maycont_gin ON {PG_TABLE} USING gin (may_contain);
  CREATE INDEX IF NOT EXISTS products_sku         ON {PG_TABLE} (sku);
  ANALYZE {PG_TABLE};
""")
print(f"Indexes built in {time.perf_counter() - t0:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ### The three basic queries from the docs
# MAGIC
# MAGIC Vector, BM25 and hybrid (RRF over the top 40 of each) — exactly the patterns from the Lakebase Search documentation,
# MAGIC pointed at our catalog.

# COMMAND ----------

import pandas as pd


def embed(texts, batch_size=16):
    out = []
    for i in range(0, len(texts), batch_size):
        resp = w.serving_endpoints.query(name=EMBEDDING_ENDPOINT, input=texts[i:i + batch_size])
        out.extend(d.embedding for d in resp.data)
    return out


def vec_literal(v):
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def q(sql, params=()):
    cur = conn.execute(sql, params)
    return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])


demo_query = "dairy-free substitute for heavy cream"
demo_vec = vec_literal(embed([demo_query])[0])

print("Vector search")
display(q(f"""
  SELECT product_id, brand, product_name
  FROM {PG_TABLE}
  ORDER BY embedding <=> %s::vector
  LIMIT 5""", (demo_vec,)))

print("BM25 keyword search (lower score = more relevant)")
display(q(f"""
  SELECT product_id, brand, product_name,
         search_tsv <@> to_bm25query(to_tsvector('english', %s), '{BM25_INDEX}') AS score
  FROM {PG_TABLE}
  ORDER BY score
  LIMIT 5""", (demo_query,)))

print("Hybrid search (Reciprocal Rank Fusion)")
display(q(f"""
  WITH vector_ranked AS (
    SELECT product_id, RANK() OVER (ORDER BY dist) AS rank
    FROM (SELECT product_id, embedding <=> %(vec)s::vector AS dist
          FROM {PG_TABLE} ORDER BY dist LIMIT 40) v
  ),
  keyword_ranked AS (
    SELECT product_id, RANK() OVER (ORDER BY score) AS rank
    FROM (SELECT product_id, search_tsv <@> to_bm25query(to_tsvector('english', %(q)s), '{BM25_INDEX}') AS score
          FROM {PG_TABLE} ORDER BY score LIMIT 40) k
  )
  SELECT d.product_id, d.brand, d.product_name, v.rank AS vector_rank, k.rank AS keyword_rank,
          --this is where the combined ranking is computed
         COALESCE(1.0 / (60 + v.rank), 0) + COALESCE(1.0 / (60 + k.rank), 0) AS rrf_score
  FROM {PG_TABLE} d
  LEFT JOIN vector_ranked  v ON d.product_id = v.product_id
  LEFT JOIN keyword_ranked k ON d.product_id = k.product_id
  WHERE v.product_id IS NOT NULL OR k.product_id IS NOT NULL
  ORDER BY rrf_score DESC, d.product_id
  LIMIT 10""", dict(vec=demo_vec, q=demo_query)))

# COMMAND ----------

# MAGIC %md
# MAGIC All three return **real heavy cream** — the one thing this shopper can't buy. "Dairy-free" is a *constraint*, not a
# MAGIC similarity signal. The next step turns the docs' hybrid pattern into reusable functions that add:
# MAGIC
# MAGIC * **Hard dietary filters** applied inside each retriever (before the `LIMIT 40`), with `lakebase_ann.prefilter` /
# MAGIC   `lakebase_bm25.prefilter` on so each still returns a full candidate list. Excluded allergens are also checked against
# MAGIC   `may_contain` — "made on shared equipment with peanuts" is not OK for a peanut allergy.
# MAGIC * **Weights** on each side of the RRF sum.
# MAGIC * **An exact SKU/UPC short-circuit.** Under plain RRF an item found by only one retriever scores at most `1/61`, less
# MAGIC   than one ranked ~20th by both (`2/80`) — wrong for identifier lookups.
# MAGIC
# MAGIC The SQL in section 6 repeats the queries above on purpose. Above is the documented pattern as-is, which is the
# MAGIC baseline that fails. Section 6 is the same SQL wrapped in functions, plus these three additions. Comparing the two
# MAGIC shows exactly what production search needs beyond the docs.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Search functions in Postgres
# MAGIC
# MAGIC Defined in a separate `grocery_app` schema (the synced table's schema is managed by the sync). Apps call one function —
# MAGIC directly or through the Lakebase Data API (`POST /rpc/hybrid_search`).
# MAGIC
# MAGIC **Why `SET lakebase_ann.prefilter = on` / `lakebase_bm25.prefilter = on`?** These settings decide when the `WHERE`
# MAGIC filter runs relative to the index's top-K search:
# MAGIC * **Off (post-filter):** the index finds its top 40 first, then the filter removes unsafe rows. For "dairy-free
# MAGIC   substitute for heavy cream" the top 40 are mostly real dairy cream, so only a few candidates survive.
# MAGIC * **On (pre-filter):** the filter is applied during the index search, so each retriever still returns a full 40
# MAGIC   candidates that pass.
# MAGIC
# MAGIC Setting them on the function (`SET …`) applies them only while the function runs, so callers can't forget them.
# MAGIC Very selective filters can make the index search do more work, so measure on large catalogs. Here the filtered
# MAGIC hybrid search was actually faster, because the GIN-indexed filters shrink the candidate set.

# COMMAND ----------

SEARCH_FUNCTIONS = f"""
CREATE SCHEMA IF NOT EXISTS grocery_app;

-- BM25 keyword candidates (lakebase_text). Lower <@> score = more relevant; 0 = no query term matched.
CREATE OR REPLACE FUNCTION grocery_app.keyword_search(
    q_text text, k int DEFAULT 40,
    required_tags text[] DEFAULT '{{}}', excluded_allergens text[] DEFAULT '{{}}')
RETURNS TABLE (product_id bigint, rank bigint)
LANGUAGE sql STABLE
SET lakebase_bm25.prefilter = on
AS $$
  SELECT product_id, RANK() OVER (ORDER BY score) AS rank
  FROM (
    SELECT p.product_id,
           p.search_tsv <@> to_bm25query(to_tsvector('english', q_text), '{BM25_INDEX}') AS score
    FROM {PG_TABLE} p
    WHERE p.dietary_tags ?& required_tags
      AND NOT (p.allergens   ?| excluded_allergens)
      AND NOT (p.may_contain ?| excluded_allergens)
    ORDER BY score
    LIMIT k
  ) kw
  WHERE score < 0
$$;

-- ANN vector candidates (lakebase_vector, cosine distance)
CREATE OR REPLACE FUNCTION grocery_app.semantic_search(
    q_embedding vector({EMBEDDING_DIM}), k int DEFAULT 40,
    required_tags text[] DEFAULT '{{}}', excluded_allergens text[] DEFAULT '{{}}')
RETURNS TABLE (product_id bigint, rank bigint)
LANGUAGE sql STABLE
SET lakebase_ann.prefilter = on
AS $$
  SELECT product_id, RANK() OVER (ORDER BY dist) AS rank
  FROM (
    SELECT p.product_id, p.embedding <=> q_embedding AS dist
    FROM {PG_TABLE} p
    WHERE p.dietary_tags ?& required_tags
      AND NOT (p.allergens   ?| excluded_allergens)
      AND NOT (p.may_contain ?| excluded_allergens)
    ORDER BY dist
    LIMIT k
  ) v
$$;

-- Hybrid: the Lakebase Search RRF pattern + weights, hard filters and an exact-SKU short-circuit
CREATE OR REPLACE FUNCTION grocery_app.hybrid_search(
    q_text text, q_embedding vector({EMBEDDING_DIM}), match_count int DEFAULT 10,
    required_tags text[] DEFAULT '{{}}', excluded_allergens text[] DEFAULT '{{}}',
    keyword_weight float8 DEFAULT 1.0, semantic_weight float8 DEFAULT 1.0,
    rrf_k int DEFAULT 60, candidates int DEFAULT 40)
RETURNS TABLE (product_id bigint, sku text, brand text, product_name text,
               keyword_rank bigint, semantic_rank bigint, rrf_score float8)
LANGUAGE sql STABLE AS $$
  WITH keyword_ranked AS (
    SELECT * FROM grocery_app.keyword_search(q_text, candidates, required_tags, excluded_allergens)
  ),
  vector_ranked AS (
    SELECT * FROM grocery_app.semantic_search(q_embedding, candidates, required_tags, excluded_allergens)
  ),
  exact_match AS (   -- identifiers are deterministic: an exact SKU / UPC hit always comes first
    SELECT p.product_id FROM {PG_TABLE} p
    WHERE p.sku = upper(btrim(q_text)) OR p.upc = btrim(q_text)
  )
  SELECT d.product_id, d.sku, d.brand, d.product_name, k.rank, v.rank,
         COALESCE(keyword_weight  / (rrf_k + k.rank), 0)
       + COALESCE(semantic_weight / (rrf_k + v.rank), 0)
       -- Exact SKU/UPC bonus. The best possible RRF score is about 1/61 + 1/61 = 0.033, so +1.0 always puts an exact
       -- identifier match first. Without it, a product found by only one retriever (max 1/61) would lose to one ranked
       -- ~20th by both (2/80). The bonus ignores the dietary filters: the shopper asked for this product by identifier.
       + CASE WHEN e.product_id IS NOT NULL THEN 1.0 ELSE 0 END AS rrf_score
  FROM {PG_TABLE} d
  LEFT JOIN vector_ranked  v ON d.product_id = v.product_id
  LEFT JOIN keyword_ranked k ON d.product_id = k.product_id
  LEFT JOIN exact_match    e ON d.product_id = e.product_id
  WHERE v.product_id IS NOT NULL OR k.product_id IS NOT NULL OR e.product_id IS NOT NULL
  ORDER BY rrf_score DESC, d.product_id
  LIMIT match_count
$$;
"""
conn.execute(SEARCH_FUNCTIONS)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query helpers
# MAGIC
# MAGIC `extract_filters` is a deliberately simple *query-understanding* step: it maps dietary phrases in the query
# MAGIC ("gluten free", "celiac", "nut-free", "soy allergy", "lactose intolerant") to structured filters. In an app these
# MAGIC would usually also come from facet checkboxes.

# COMMAND ----------

import re

TAG_PATTERNS = {
    "gluten-free": r"gluten[- ]?free|celiac|coeliac",
    "dairy-free": r"dairy[- ]?free|non[- ]?dairy|lactose|without dairy",
    "nut-free": r"nut[- ]?free",
    "vegan": r"\bvegan\b",
    "vegetarian": r"\bvegetarian\b",
    "organic": r"\borganic\b",
    "kosher": r"\bkosher\b",
    "keto": r"\bketo\b",
    "non-gmo": r"non[- ]?gmo",
    "soy-free": r"soy[- ]?free",
    "sugar-free": r"sugar[- ]?free|without sugar|zero sugar",  # NOT "no sugar added": that's a different label claim
}
ALLERGY_PATTERNS = {
    "peanuts": r"peanut allerg", "tree nuts": r"\b(tree )?nut allerg", "soy": r"soy allerg",
    "milk": r"(milk|dairy) allerg", "wheat": r"wheat allerg", "egg": r"egg allerg", "sesame": r"sesame allerg",
}


def extract_filters(query):
    ql = query.lower()
    tags = [t for t, pat in TAG_PATTERNS.items() if re.search(pat, ql)]
    allergens = [a for a, pat in ALLERGY_PATTERNS.items() if re.search(pat, ql)]
    if "nut-free" in tags:
        allergens += ["peanuts", "tree nuts"]
    return sorted(set(tags)), sorted(set(allergens))


def search(query, mode="hybrid", k=10, embedding=None, use_filters=False,
           keyword_weight=1.0, semantic_weight=1.0, rrf_k=60):
    """Returns (product_ids, latency_ms). mode: keyword | semantic | hybrid"""
    tags, allergens = extract_filters(query) if use_filters else ([], [])
    if embedding is None and mode != "keyword":
        embedding = embed([query])[0]
    t0 = time.perf_counter()
    if mode == "keyword":
        rows = conn.execute("SELECT product_id FROM grocery_app.keyword_search(%s, %s, %s, %s) ORDER BY rank, product_id",
                            (query, k, tags, allergens)).fetchall()
    elif mode == "semantic":
        rows = conn.execute("SELECT product_id FROM grocery_app.semantic_search(%s::vector, %s, %s, %s) ORDER BY rank, product_id",
                            (vec_literal(embedding), k, tags, allergens)).fetchall()
    else:
        rows = conn.execute(
            "SELECT product_id FROM grocery_app.hybrid_search(%s, %s::vector, %s, %s, %s, %s, %s, %s)",
            (query, vec_literal(embedding), k, tags, allergens, keyword_weight, semantic_weight, rrf_k)).fetchall()
    return [r[0] for r in rows], (time.perf_counter() - t0) * 1000


def show(query, use_filters=False, k=5, **weights):
    """Side-by-side top-k for keyword, semantic and hybrid."""
    emb = embed([query])[0]
    frames = []
    for mode in ("keyword", "semantic", "hybrid"):
        ids, _ = search(query, mode, k, emb, use_filters, **(weights if mode == "hybrid" else {}))
        details = conn.execute(f"""
            SELECT product_id, brand || ' ' || product_name,
                   (SELECT string_agg(a, ', ') FROM jsonb_array_elements_text(allergens) a)
            FROM {PG_TABLE} WHERE product_id = ANY(%s)""", (ids,)).fetchall()
        by_id = {d[0]: d[1:] for d in details}
        for rank, pid in enumerate(ids, 1):
            name, allergens = by_id[pid]
            frames.append(dict(mode=mode, rank=rank, product=name, allergens=allergens or "—",
                               label=f"{name}  [contains: {allergens}]" if allergens else name))
    print("Query:", query, "| filters:", extract_filters(query) if use_filters else "off")
    return pd.DataFrame(frames).pivot(index="rank", columns="mode", values="label").reindex(columns=["keyword", "semantic", "hybrid"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · See the difference
# MAGIC
# MAGIC **Intent + constraint.** Without filters, every retriever returns heavy cream. With the extracted `dairy-free` filter, the
# MAGIC same hybrid query returns oat, coconut and cashew creams.

# COMMAND ----------

display(show("dairy-free substitute for heavy cream"))

# COMMAND ----------

display(show("dairy-free substitute for heavy cream", use_filters=True))

# COMMAND ----------

# MAGIC %md
# MAGIC **Exact query.** A SKU is noise to an embedding model; the exact-match branch nails it.

# COMMAND ----------

sample_sku = conn.execute(f"SELECT sku FROM {PG_TABLE} WHERE category = 'Pantry' ORDER BY product_id LIMIT 1 OFFSET 40").fetchone()[0]
display(show(sample_sku))

# COMMAND ----------

# MAGIC %md
# MAGIC **Allergy query.** *"stir fry seasoning I can use with a soy allergy that is vegetarian"*. First, without filters:
# MAGIC soy sauce and tamari are, semantically, stir-fry seasoning, so they rank highly in vector search, and hybrid search
# MAGIC puts them first because both retrievers agree on them.

# COMMAND ----------

display(show("stir fry seasoning I can use with a soy allergy that is vegetarian"))

# COMMAND ----------

# MAGIC %md
# MAGIC **Now the same query with filters.** `extract_filters` turns "soy allergy" into `excluded_allergens = {soy}` and
# MAGIC "vegetarian" into `required_tags = {vegetarian}`. Both are applied inside each retriever before ranking, so soy sauce
# MAGIC and tamari are excluded, and only soy-free, vegetarian options such as coconut aminos remain.

# COMMAND ----------

display(show("stir fry seasoning I can use with a soy allergy that is vegetarian", use_filters=True))

# COMMAND ----------

