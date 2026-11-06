# Smarter grocery search: filter allergies and diets first, then rank (Lakebase Search)

Grocery search for StreetsMart, a fictional retailer, built with
[Lakebase Search](https://docs.databricks.com/aws/en/oltp/projects/lakebase-search).
Plain search treats "dairy-free" or "soy allergy" as just more words and still returns products the shopper can't eat.
This project filters first, then ranks:

1. **Filter:** remove every product that breaks the shopper's dietary tags or allergens (hard `WHERE` filters inside each retriever).
2. **Rank:** order what's left with BM25 (`lakebase_bm25`), vector search (`lakebase_ann`) and Reciprocal Rank Fusion.

| Path | What |
|---|---|
| `data/grocery_products.csv` | 3,128 synthetic StreetsMart products (list columns as JSON arrays) |
| `notebooks/02_lakebase_hybrid_search.py` | Loads the CSV → `ai_query` embeddings → synced table (`vector(1024)` + generated `tsvector`) → `lakebase_ann` / `lakebase_bm25` indexes → hybrid search functions with dietary filters and exact SKU/UPC match → before/after examples |
| `notebooks/01_generate_grocery_catalog.py` | *Optional*: the seeded generator that reproduces the CSV |

## Run it

1. Create a Lakebase project (Postgres 16+) and enable **Lakebase Search** in its settings
   (*Settings → Lakebase Search → Enable*). Enabling is irreversible, so use a dedicated project.
2. Upload `data/grocery_products.csv` to a Unity Catalog Volume (default `/Volumes/<catalog>/<schema>/blog_assets/data`),
   or set the notebook's `data_location` widget to where it's published.
3. Import `notebooks/02_lakebase_hybrid_search.py` into your workspace, set the `catalog`, `schema` and `lakebase_project`
   widgets, and run it on serverless.
