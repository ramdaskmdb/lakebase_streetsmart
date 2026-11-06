# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 01 · (Optional) Generate the synthetic grocery catalog
# MAGIC
# MAGIC > **You don't need to run this notebook.** The dataset is published with the blog as a CSV (`grocery_products.csv`),
# MAGIC > and notebook 02 loads it. Run this one only to see how the data was made, or to regenerate or modify it. With the
# MAGIC > default seed it reproduces the published CSV exactly. It also writes it to `<output_dir>`.
# MAGIC
# MAGIC This notebook builds the dataset for the **Lakebase hybrid search** blog:
# MAGIC
# MAGIC | Table | What it holds |
# MAGIC |---|---|
# MAGIC | `grocery_products` | ~3,500 products for a fictional retailer, **StreetsMart**: SKU, UPC, brand, size, rich description, ingredients, allergens, "may contain" warnings and dietary tags |
# MAGIC
# MAGIC The generator is deliberately *template-driven and seeded* rather than LLM-generated, so that:
# MAGIC * results are reproducible run-to-run, and
# MAGIC * we know the ground truth for every product (its "archetype", tags and allergens), so every search result can be checked against it.
# MAGIC
# MAGIC The descriptions are written the way real product copy is written: they describe **what the product is and how to use it**
# MAGIC ("whips into fluffy peaks once chilled, simmer into curries") — they rarely contain the words a shopper types
# MAGIC ("dairy-free substitute for heavy cream"). That vocabulary gap is exactly what makes hybrid search necessary.

# COMMAND ----------

dbutils.widgets.text("catalog", "rkm_cat", "Catalog")
dbutils.widgets.text("schema", "grocery_search", "Schema")
dbutils.widgets.text("num_products", "3500", "Approx. number of products")
dbutils.widgets.text("seed", "42", "Random seed")
dbutils.widgets.text("output_dir", "", "CSV output folder (default: /Volumes/<catalog>/<schema>/blog_assets/data)")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
NUM_PRODUCTS = int(dbutils.widgets.get("num_products"))
SEED = int(dbutils.widgets.get("seed"))
OUTPUT_DIR = (dbutils.widgets.get("output_dir") or f"/Volumes/{CATALOG}/{SCHEMA}/blog_assets/data").rstrip("/")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Product archetypes
# MAGIC
# MAGIC Each archetype is a *kind* of product (e.g. "coconut cream"). Every generated product belongs to one archetype and
# MAGIC inherits its allergens and dietary tags. `uses` and `texture` phrases feed the description, and `claims` lists the
# MAGIC tags that are allowed to be printed on the label.
# MAGIC
# MAGIC Note the **distractors**: real heavy cream sits next to coconut cream, wheat pasta next to chickpea pasta, peanut
# MAGIC butter next to sunflower seed butter. A search engine that surfaces the wrong one is not just unhelpful — for a
# MAGIC shopper with an allergy, it is unsafe.

# COMMAND ----------

import random
import re

ARCHETYPES = [
    # ---------------- Dairy & alternatives ----------------
    dict(key="heavy_cream", category="Dairy & Eggs", subcategory="Cream",
         names=["Heavy Whipping Cream", "Heavy Cream", "Ultra-Pasteurized Heavy Cream"], variants=["", "Organic"],
         sizes=["16 fl oz", "32 fl oz", "8 fl oz"], ingredients="cream (milk), carrageenan",
         allergens=["milk"], tags=["vegetarian", "gluten-free", "keto"],
         texture=["rich, 36% milkfat", "thick and luscious", "farm-fresh"],
         uses=["whip into soft peaks for desserts", "enrich pan sauces and soups", "stir into coffee", "make homemade ice cream"]),
    dict(key="coconut_cream", category="Pantry", subcategory="Coconut Products",
         names=["Coconut Cream", "Organic Coconut Cream", "Thick Coconut Cream"], variants=["", "Unsweetened"],
         sizes=["13.5 fl oz can", "5.4 fl oz can", "4 x 13.5 fl oz"], ingredients="coconut extract, water, guar gum",
         allergens=[], tags=["vegan", "dairy-free", "gluten-free", "keto", "paleo", "nut-free"],
         texture=["thick and spoonable", "velvety", "rich and silky"],
         uses=["whip into fluffy peaks once chilled", "simmer into curries and creamy soups",
               "swirl into pasta sauces for body", "pour over berries"]),
    dict(key="cashew_cooking_cream", category="Dairy & Eggs", subcategory="Plant-Based Cream",
         names=["Cashew Cooking Cream", "Cashew Culinary Cream"], variants=["", "Original"],
         sizes=["16 fl oz", "8 fl oz"], ingredients="filtered water, cashews, sunflower oil, sea salt, gellan gum",
         allergens=["tree nuts"], tags=["vegan", "dairy-free", "gluten-free", "soy-free"],
         texture=["silky", "velvety smooth", "neutral-tasting"],
         uses=["make silky alfredo-style sauces", "finish tomato soups", "fold into mashed potatoes", "bake into gratins"]),
    dict(key="oat_cooking_cream", category="Dairy & Eggs", subcategory="Plant-Based Cream",
         names=["Oat Cooking Cream", "Oat Culinary Base"], variants=["", "Extra Rich"],
         sizes=["16.9 fl oz", "8.4 fl oz"], ingredients="oat base (water, oats), rapeseed oil, gellan gum, sea salt",
         allergens=[], tags=["vegan", "dairy-free", "nut-free", "soy-free"],
         texture=["smooth", "mild and creamy"],
         uses=["thicken sauces without splitting", "make creamy pasta bakes", "add body to soups and stews"]),
    dict(key="oat_milk", category="Dairy & Eggs", subcategory="Plant-Based Milk",
         names=["Oat Milk", "Oat Beverage", "Oatmilk"], variants=["Original", "Unsweetened", "Barista Edition", "Vanilla", "Extra Creamy"],
         sizes=["64 fl oz", "32 fl oz", "6 x 11 fl oz"], ingredients="oat base (water, oats), sunflower oil, calcium carbonate, sea salt",
         allergens=[], tags=["vegan", "dairy-free", "nut-free", "soy-free"],
         texture=["creamy", "naturally sweet", "smooth"],
         uses=["froth into a velvety microfoam for lattes", "pour over cereal", "blend into smoothies", "bake into muffins"]),
    dict(key="almond_milk", category="Dairy & Eggs", subcategory="Plant-Based Milk",
         names=["Almond Milk", "Almondmilk"], variants=["Original", "Unsweetened", "Vanilla", "Chocolate"],
         sizes=["64 fl oz", "32 fl oz"], ingredients="almond base (water, almonds), calcium carbonate, sea salt, locust bean gum",
         allergens=["tree nuts"], tags=["vegan", "dairy-free", "gluten-free", "soy-free", "keto"],
         texture=["light", "nutty", "smooth"],
         uses=["pour over cereal", "blend into smoothies", "splash into coffee"]),
    dict(key="whole_milk", category="Dairy & Eggs", subcategory="Milk",
         names=["Whole Milk", "Vitamin D Whole Milk", "Grass-Fed Whole Milk"], variants=["", "Organic"],
         sizes=["1 gal", "half gal", "1 qt"], ingredients="grade A milk, vitamin D3",
         allergens=["milk"], tags=["vegetarian", "gluten-free"],
         texture=["fresh", "creamy"], uses=["pour over cereal", "steam for lattes", "bake into cakes"]),
    dict(key="greek_yogurt", category="Dairy & Eggs", subcategory="Yogurt",
         names=["Greek Yogurt", "Strained Greek Yogurt"], variants=["Plain", "Vanilla", "Honey", "Strawberry"],
         sizes=["32 oz", "5.3 oz", "4 x 5.3 oz"], ingredients="cultured pasteurized milk, live active cultures",
         allergens=["milk"], tags=["vegetarian", "gluten-free", "high-protein"],
         texture=["thick and tangy", "ultra-creamy"], uses=["top with granola", "swap for sour cream", "blend into smoothies"]),
    dict(key="coconut_yogurt", category="Dairy & Eggs", subcategory="Plant-Based Yogurt",
         names=["Coconut Yogurt", "Coconutmilk Yogurt Alternative"], variants=["Plain", "Vanilla", "Mango"],
         sizes=["24 oz", "5.3 oz"], ingredients="coconut cream, water, tapioca starch, live vegan cultures",
         allergens=[], tags=["vegan", "dairy-free", "gluten-free", "soy-free", "nut-free"],
         texture=["tangy", "creamy"], uses=["top with fruit and granola", "stir into overnight oats", "dollop on curries"]),
    dict(key="cheddar", category="Dairy & Eggs", subcategory="Cheese",
         names=["Sharp Cheddar", "Aged White Cheddar", "Mild Cheddar Shreds"], variants=["", "Organic"],
         sizes=["8 oz", "16 oz", "2 lb"], ingredients="pasteurized milk, cheese cultures, salt, enzymes, annatto",
         allergens=["milk"], tags=["vegetarian", "gluten-free", "keto"],
         texture=["bold, tangy", "aged 12 months", "melty"], uses=["melt over burgers", "build a cheese board", "bake into mac and cheese"]),
    dict(key="plant_cheese_shreds", category="Dairy & Eggs", subcategory="Plant-Based Cheese",
         names=["Plant-Based Mozzarella Shreds", "Cheddar-Style Shreds", "Oat-Based Shreds"], variants=["", "Mozzarella Style"],
         sizes=["7 oz", "8 oz"], ingredients="water, coconut oil, potato starch, tapioca, sea salt, natural flavor",
         allergens=[], tags=["vegan", "dairy-free", "gluten-free", "soy-free", "nut-free"],
         texture=["stretchy", "melts and bubbles", "gooey"],
         uses=["melt over homemade pizza", "layer into lasagna", "top nachos"]),
    dict(key="butter", category="Dairy & Eggs", subcategory="Butter",
         names=["Unsalted Butter", "Salted Sweet Cream Butter", "European-Style Butter"], variants=["", "Organic"],
         sizes=["16 oz", "8 oz"], ingredients="pasteurized cream (milk), salt",
         allergens=["milk"], tags=["vegetarian", "gluten-free", "keto"],
         texture=["rich", "82% butterfat"], uses=["laminate into croissants", "spread on toast", "brown for sauces"]),
    dict(key="vegan_butter", category="Dairy & Eggs", subcategory="Plant-Based Butter",
         names=["Plant Butter Sticks", "Vegan Butter", "Plant-Based Buttery Spread"], variants=["", "Salted"],
         sizes=["16 oz", "8 oz"], ingredients="palm fruit oil, canola oil, olive oil, water, sea salt, sunflower lecithin",
         allergens=[], tags=["vegan", "dairy-free", "gluten-free", "nut-free", "soy-free"],
         texture=["rich", "spreadable"],
         uses=["bakes into flaky pie crusts and biscuits", "cream with sugar for cookies", "melt over popcorn"]),
    # ---------------- Pasta, grains, baking ----------------
    dict(key="wheat_pasta", category="Pantry", subcategory="Pasta",
         names=["Penne Rigate", "Spaghetti", "Rotini", "Elbow Macaroni"], variants=["", "Whole Wheat"],
         sizes=["16 oz", "32 oz", "12 oz"], ingredients="durum wheat semolina, niacin, iron, thiamine, riboflavin, folic acid",
         allergens=["wheat"], tags=["vegan"],
         texture=["bronze-cut", "classic al dente"], uses=["toss with marinara", "bake into casseroles", "make pasta salad"]),
    dict(key="gf_pasta", category="Pantry", subcategory="Pasta",
         names=["Brown Rice Penne", "Brown Rice Spaghetti", "Corn & Quinoa Fusilli", "Rice Rotini"], variants=["", "Organic"],
         sizes=["12 oz", "16 oz"], ingredients="brown rice flour, water",
         allergens=[], tags=["vegan", "gluten-free", "dairy-free", "nut-free", "soy-free", "egg-free"],
         texture=["holds its shape al dente", "never mushy"], uses=["toss with pesto", "bake into casseroles", "pair with marinara"]),
    dict(key="chickpea_pasta", category="Pantry", subcategory="Pasta",
         names=["Chickpea Rotini", "Chickpea Penne", "Red Lentil Spaghetti", "Chickpea Shells"], variants=["", "Organic"],
         sizes=["8 oz", "16 oz"], ingredients="chickpea flour, pea protein, tapioca, xanthan gum",
         allergens=[], tags=["vegan", "gluten-free", "high-protein", "dairy-free", "nut-free", "egg-free"],
         texture=["hearty", "nutty"], uses=["make a protein-packed mac and cheese", "toss with roasted vegetables"],
         stats=["20g of protein and 8g of fiber per serving", "twice the protein of traditional noodles"]),
    dict(key="ap_flour", category="Baking", subcategory="Flour",
         names=["All-Purpose Flour", "Unbleached All-Purpose Flour", "Bread Flour"], variants=["", "Organic"],
         sizes=["5 lb", "2 lb", "10 lb"], ingredients="unbleached enriched wheat flour, malted barley flour",
         allergens=["wheat"], tags=["vegan"],
         texture=["finely milled", "unbleached"], uses=["bake breads and pizza dough", "thicken gravies", "make cookies"]),
    dict(key="almond_flour", category="Baking", subcategory="Flour",
         names=["Blanched Almond Flour", "Super-Fine Almond Flour"], variants=["", "Organic"],
         sizes=["16 oz", "32 oz", "3 lb"], ingredients="blanched almonds",
         allergens=["tree nuts"], tags=["vegan", "gluten-free", "keto", "paleo", "dairy-free", "grain-free"],
         texture=["super-fine", "moist and tender crumb"], uses=["bake macarons", "make tender cakes and muffins", "bread chicken cutlets"],
         stats=["3g net carbs per serving", "only one ingredient"]),
    dict(key="gf_flour_blend", category="Baking", subcategory="Flour",
         names=["1-to-1 Gluten-Free Baking Flour", "Measure-for-Measure Flour Blend"], variants=[""],
         sizes=["22 oz", "44 oz"], ingredients="rice flour, potato starch, sorghum flour, tapioca flour, xanthan gum",
         allergens=[], tags=["vegan", "gluten-free", "nut-free", "dairy-free"],
         texture=["cup-for-cup", "light"], uses=["swap cup-for-cup in your favorite cake and cookie recipes", "make pancakes and brownies"]),
    dict(key="eggs", category="Dairy & Eggs", subcategory="Eggs",
         names=["Large Brown Eggs", "Pasture-Raised Large Eggs", "Cage-Free Large White Eggs"], variants=["", "Organic"],
         sizes=["12 ct", "18 ct", "6 ct"], ingredients="eggs",
         allergens=["egg"], tags=["vegetarian", "gluten-free", "keto", "paleo", "high-protein"],
         texture=["farm-fresh", "rich golden yolks"], uses=["scramble for breakfast", "bake into cakes", "make custards"]),
    dict(key="egg_replacer", category="Baking", subcategory="Baking Aids",
         names=["Plant-Based Egg Replacer", "Egg-Free Baking Mix", "Flax Egg Blend"], variants=[""],
         sizes=["16 oz", "8 oz"], ingredients="potato starch, tapioca flour, baking soda, psyllium husk fiber",
         allergens=[], tags=["vegan", "egg-free", "gluten-free", "nut-free", "dairy-free"],
         texture=["shelf-stable", "neutral-tasting"],
         uses=["bind cookie dough and brownie batter", "lift cakes and muffins", "one tablespoon plus water stands in for one large egg"]),
    dict(key="cane_sugar", category="Baking", subcategory="Sweeteners",
         names=["Pure Cane Sugar", "Granulated Sugar", "Organic Cane Sugar"], variants=[""],
         sizes=["4 lb", "2 lb", "10 lb"], ingredients="cane sugar",
         allergens=[], tags=["vegan", "gluten-free"],
         texture=["fine granulated", "sparkling"], uses=["sweeten coffee and tea", "bake cookies and cakes", "make simple syrup"]),
    dict(key="monk_fruit", category="Baking", subcategory="Sweeteners",
         names=["Monk Fruit Sweetener", "Monk Fruit & Allulose Blend", "Golden Monk Fruit Sweetener"], variants=["", "Granular"],
         sizes=["16 oz", "32 oz"], ingredients="allulose, erythritol, monk fruit extract",
         allergens=[], tags=["vegan", "keto", "sugar-free", "gluten-free"],
         texture=["measures cup-for-cup like sugar", "no bitter aftertaste"],
         uses=["sweeten coffee", "bake brownies and cookies", "stir into lemonade"],
         stats=["zero calories and zero net carbs", "zero glycemic impact"]),
    dict(key="honey", category="Pantry", subcategory="Sweeteners",
         names=["Raw Wildflower Honey", "Clover Honey", "Manuka Honey"], variants=[""],
         sizes=["16 oz", "12 oz squeeze", "32 oz"], ingredients="raw honey",
         allergens=[], tags=["gluten-free", "paleo"],
         texture=["golden", "floral"], uses=["drizzle over yogurt", "sweeten tea", "glaze roasted carrots"]),
    dict(key="maple_syrup", category="Pantry", subcategory="Sweeteners",
         names=["Pure Maple Syrup", "Grade A Amber Maple Syrup"], variants=["", "Organic"],
         sizes=["12 fl oz", "32 fl oz"], ingredients="pure maple syrup",
         allergens=[], tags=["vegan", "gluten-free", "paleo"],
         texture=["rich amber", "robust"], uses=["pour over pancakes", "sweeten oatmeal", "glaze salmon"]),
    # ---------------- Sauces & spreads ----------------
    dict(key="soy_sauce", category="Pantry", subcategory="Sauces",
         names=["Soy Sauce", "Naturally Brewed Soy Sauce", "Less Sodium Soy Sauce"], variants=[""],
         sizes=["10 fl oz", "20 fl oz"], ingredients="water, wheat, soybeans, salt",
         allergens=["soy", "wheat"], tags=["vegan"],
         texture=["deep umami", "naturally brewed"], uses=["season stir-fries", "make marinades", "dip sushi"]),
    dict(key="tamari", category="Pantry", subcategory="Sauces",
         names=["Tamari", "Gluten-Free Tamari", "Reduced Sodium Tamari"], variants=["", "Organic"],
         sizes=["10 fl oz", "20 fl oz"], ingredients="water, soybeans, salt, sugar",
         allergens=["soy"], tags=["vegan", "gluten-free"],
         texture=["rich, smooth", "brewed without wheat"], uses=["season stir-fries", "dip dumplings and sushi", "splash into ramen broth"]),
    dict(key="coconut_aminos", category="Pantry", subcategory="Sauces",
         names=["Coconut Aminos", "Coconut Aminos Seasoning Sauce"], variants=["", "Organic"],
         sizes=["8 fl oz", "16.9 fl oz"], ingredients="organic coconut tree sap, sea salt",
         allergens=[], tags=["vegan", "gluten-free", "soy-free", "paleo", "low-sodium"],
         texture=["savory-sweet", "umami-rich"],
         uses=["season stir-fries and fried rice", "make marinades and dressings", "dip sushi"],
         stats=["73% less sodium than conventional seasoning sauces", "made from just two ingredients"]),
    dict(key="peanut_butter", category="Pantry", subcategory="Nut & Seed Butters",
         names=["Creamy Peanut Butter", "Crunchy Peanut Butter", "Natural Peanut Butter"], variants=["", "No Salt Added"],
         sizes=["16 oz", "28 oz", "40 oz"], ingredients="dry roasted peanuts, salt",
         allergens=["peanuts"], tags=["vegan", "gluten-free", "high-protein"],
         texture=["creamy", "crunchy"], uses=["spread on toast", "pack a PB&J for lunch", "blend into smoothies"]),
    dict(key="sunbutter", category="Pantry", subcategory="Nut & Seed Butters",
         names=["Sunflower Seed Butter", "Creamy Sunflower Spread"], variants=["", "No Sugar Added"],
         sizes=["16 oz", "28 oz"], ingredients="roasted sunflower seeds, cane sugar, salt",
         allergens=[], tags=["vegan", "gluten-free", "nut-free", "peanut-free", "soy-free", "dairy-free"],
         texture=["creamy", "roasty"], uses=["make sandwiches for school lunchboxes", "dip apple slices", "bake into energy bites"],
         stats=["made in a dedicated facility free from the top allergens", "safe for classrooms"]),
    dict(key="almond_butter", category="Pantry", subcategory="Nut & Seed Butters",
         names=["Almond Butter", "Roasted Almond Butter"], variants=["Creamy", "Crunchy"],
         sizes=["12 oz", "16 oz"], ingredients="dry roasted almonds",
         allergens=["tree nuts"], tags=["vegan", "gluten-free", "keto", "paleo"],
         texture=["creamy", "toasty"], uses=["swirl into oatmeal", "spread on rice cakes"]),
    # ---------------- Snacks ----------------
    dict(key="wheat_crackers", category="Snacks", subcategory="Crackers",
         names=["Wheat Thins-Style Crackers", "Buttery Round Crackers", "Water Crackers"], variants=["", "Reduced Fat"],
         sizes=["9 oz", "13.7 oz"], ingredients="enriched flour (wheat flour), canola oil, sugar, salt",
         allergens=["wheat"], tags=["vegetarian"],
         texture=["crisp", "flaky", "buttery"], uses=["serve with cheese and charcuterie", "crumble over soup"]),
    dict(key="gf_crackers", category="Snacks", subcategory="Crackers",
         names=["Brown Rice Crackers", "Rice Thin Crackers", "Almond Flour Crackers"], variants=["Sea Salt", "Sesame", "Rosemary"],
         sizes=["4.25 oz", "8 oz"], ingredients="whole grain brown rice, sesame seeds, sea salt",
         allergens=["sesame"], tags=["vegan", "gluten-free", "dairy-free"],
         texture=["light and crispy", "crunchy"], uses=["pair with cheese and charcuterie", "scoop hummus"]),
    dict(key="seed_crackers", category="Snacks", subcategory="Crackers",
         names=["Seed Crackers", "Flax & Chia Seed Crisps"], variants=["Original", "Everything", "Rosemary"],
         sizes=["4.5 oz", "6 oz"], ingredients="flax seeds, chia seeds, pumpkin seeds, sunflower seeds, sea salt",
         allergens=[], tags=["vegan", "gluten-free", "keto", "paleo", "grain-free", "nut-free"],
         texture=["crunchy, seedy", "hearty"], uses=["dip into guacamole", "top with smoked salmon", "pair with soft cheeses"]),
    dict(key="potato_chips", category="Snacks", subcategory="Chips",
         names=["Kettle Cooked Potato Chips", "Classic Potato Chips"], variants=["Sea Salt", "Salt & Vinegar", "BBQ", "Jalapeño"],
         sizes=["8 oz", "1 oz", "13 oz"], ingredients="potatoes, sunflower oil, sea salt",
         allergens=[], tags=["vegan", "gluten-free"],
         texture=["extra crunchy", "thick-cut"], uses=["pack alongside sandwiches", "serve with dip"]),
    dict(key="pork_rinds", category="Snacks", subcategory="Chips",
         names=["Pork Rinds", "Oven-Baked Pork Rinds", "Chicharrones"], variants=["Original", "Hot & Spicy", "Sea Salt"],
         sizes=["3 oz", "1 oz"], ingredients="pork skins, salt",
         allergens=[], tags=["gluten-free", "keto", "paleo", "high-protein"],
         texture=["airy, crunchy", "light"], uses=["snack straight from the bag", "crush into a breading for chicken"],
         stats=["0g carbs and 9g protein per serving", "zero carbs"]),
    dict(key="granola", category="Breakfast", subcategory="Cereal & Granola",
         names=["Maple Pecan Granola", "Honey Almond Granola", "Classic Oat Granola"], variants=["", "Low Sugar"],
         sizes=["12 oz", "24 oz"], ingredients="rolled oats, pecans, almonds, maple syrup, coconut oil",
         allergens=["tree nuts"], tags=["vegetarian"],
         texture=["big clusters", "crunchy"], uses=["sprinkle over yogurt", "eat by the handful", "pour with milk"]),
    dict(key="protein_bar", category="Snacks", subcategory="Bars",
         names=["Whey Protein Bar", "Chocolate Peanut Protein Bar", "Cookie Dough Protein Bar"], variants=["", "12-Pack"],
         sizes=["2.1 oz", "12 x 2.1 oz"], ingredients="whey protein isolate, peanuts, soluble corn fiber, cocoa, almonds",
         allergens=["milk", "peanuts", "soy"], tags=["gluten-free", "high-protein"],
         texture=["chewy", "fudgy"], uses=["refuel after the gym", "stash in a gym bag"],
         stats=["20g protein, 1g sugar"]),
    dict(key="dried_mango", category="Snacks", subcategory="Dried Fruit",
         names=["Dried Mango", "Unsweetened Mango Slices"], variants=[""],
         sizes=["6 oz", "16 oz"], ingredients="mango",
         allergens=[], tags=["vegan", "gluten-free", "paleo", "nut-free"],
         texture=["chewy", "tangy-sweet"], uses=["toss into trail mix", "snack on the go"]),
    # ---------------- Meat, seafood, plant protein ----------------
    dict(key="chicken_breast", category="Meat & Seafood", subcategory="Poultry",
         names=["Boneless Skinless Chicken Breasts", "Air-Chilled Chicken Breast"], variants=["", "Organic"],
         sizes=["1.5 lb", "3 lb"], ingredients="chicken",
         allergens=[], tags=["gluten-free", "keto", "paleo", "high-protein"],
         texture=["tender", "air-chilled"], uses=["grill or roast", "slice into stir-fries"]),
    dict(key="ground_beef", category="Meat & Seafood", subcategory="Beef",
         names=["85% Lean Ground Beef", "Grass-Fed Ground Beef"], variants=[""],
         sizes=["1 lb", "2 lb"], ingredients="beef",
         allergens=[], tags=["gluten-free", "keto", "paleo", "high-protein"],
         texture=["juicy"], uses=["form into burger patties", "brown for tacos and chili"]),
    dict(key="tofu", category="Meat & Seafood", subcategory="Plant-Based Protein",
         names=["Extra Firm Tofu", "Super Firm Tofu", "Silken Tofu"], variants=["", "Organic"],
         sizes=["14 oz", "16 oz"], ingredients="water, soybeans, calcium sulfate",
         allergens=["soy"], tags=["vegan", "gluten-free", "high-protein"],
         texture=["firm", "custardy"], uses=["press and pan-fry", "crumble into a scramble", "cube for stir-fries"]),
    dict(key="tempeh", category="Meat & Seafood", subcategory="Plant-Based Protein",
         names=["Original Tempeh", "Three-Grain Tempeh"], variants=["", "Organic"],
         sizes=["8 oz"], ingredients="cultured soybeans, water, brown rice",
         allergens=["soy"], tags=["vegan", "high-protein"],
         texture=["nutty", "firm, fermented"], uses=["marinate and grill", "crumble into chili", "slice into sandwiches"]),
    dict(key="plant_burger", category="Meat & Seafood", subcategory="Plant-Based Protein",
         names=["Plant-Based Burger Patties", "Pea Protein Burgers", "Beyond-Style Grillers"], variants=["2-Pack", "4-Pack"],
         sizes=["8 oz", "16 oz"], ingredients="water, pea protein, canola oil, coconut oil, rice protein, beet juice extract",
         allergens=[], tags=["vegan", "gluten-free", "soy-free", "high-protein", "dairy-free"],
         texture=["juicy", "seared, meaty texture"], uses=["sizzle on the grill", "smash on a griddle"]),
    dict(key="seitan", category="Meat & Seafood", subcategory="Plant-Based Protein",
         names=["Seitan Strips", "Traditional Seitan"], variants=["", "Smoky"],
         sizes=["8 oz"], ingredients="vital wheat gluten, water, soy sauce (wheat, soybeans), spices",
         allergens=["wheat", "soy"], tags=["vegan", "high-protein"],
         texture=["chewy", "meaty"], uses=["toss into stir-fries", "stuff into sandwiches"]),
    dict(key="chickpeas", category="Pantry", subcategory="Canned Beans",
         names=["Garbanzo Beans", "Chickpeas", "Low Sodium Chickpeas"], variants=["", "Organic"],
         sizes=["15 oz can", "4 x 15 oz"], ingredients="chickpeas, water, sea salt",
         allergens=[], tags=["vegan", "gluten-free", "nut-free"],
         texture=["tender", "creamy"], uses=["blend into hummus", "roast until crispy", "toss into salads"]),
    dict(key="smoked_salmon", category="Meat & Seafood", subcategory="Seafood",
         names=["Cold-Smoked Atlantic Salmon", "Wild Sockeye Lox"], variants=[""],
         sizes=["4 oz", "8 oz"], ingredients="salmon, salt, sugar, natural wood smoke",
         allergens=["fish"], tags=["gluten-free", "keto", "paleo", "high-protein"],
         texture=["silky", "delicately smoked"], uses=["layer on bagels", "top blinis"]),
    # ---------------- Bakery, tortillas, swaps ----------------
    dict(key="sourdough", category="Bakery", subcategory="Bread",
         names=["Sourdough Loaf", "Country White Sourdough", "Seeded Sourdough"], variants=["", "Sliced"],
         sizes=["24 oz", "16 oz"], ingredients="unbleached wheat flour, water, sourdough starter, salt",
         allergens=["wheat"], tags=["vegan"],
         texture=["crackly crust", "open, chewy crumb"], uses=["toast for avocado toast", "make grilled cheese"]),
    dict(key="gf_bread", category="Bakery", subcategory="Bread",
         names=["Gluten-Free Sandwich Bread", "Gluten-Free Multigrain Loaf"], variants=["", "Seeded"],
         sizes=["18 oz", "24 oz"], ingredients="water, tapioca starch, brown rice flour, eggs, sorghum flour, psyllium",
         allergens=["egg"], tags=["vegetarian", "gluten-free", "dairy-free", "nut-free"],
         texture=["soft", "fluffy"], uses=["build sandwiches", "toast for breakfast"]),
    dict(key="cauli_crust", category="Frozen", subcategory="Pizza",
         names=["Cauliflower Pizza Crust", "Cauliflower Crust Margherita Pizza"], variants=["", "2-Pack"],
         sizes=["10 oz", "2 x 6 oz"], ingredients="cauliflower, mozzarella cheese (milk), egg whites, almond flour",
         allergens=["milk", "egg", "tree nuts"], tags=["gluten-free", "keto", "grain-free", "vegetarian"],
         texture=["thin and crispy", "golden"], uses=["bake for homemade pizza night", "top with your favorite toppings"],
         stats=["6g net carbs per slice", "cauliflower is the first ingredient"]),
    dict(key="corn_tortillas", category="Bakery", subcategory="Tortillas",
         names=["Yellow Corn Tortillas", "White Corn Tortillas", "Street Taco Tortillas"], variants=[""],
         sizes=["30 ct", "12 ct"], ingredients="stone-ground corn masa, water, lime",
         allergens=[], tags=["vegan", "gluten-free", "nut-free", "dairy-free"],
         texture=["soft", "stone-ground"], uses=["warm on a comal for tacos", "fry into tostadas", "make enchiladas"]),
    dict(key="flour_tortillas", category="Bakery", subcategory="Tortillas",
         names=["Flour Tortillas", "Burrito-Size Flour Tortillas"], variants=["", "Whole Wheat"],
         sizes=["10 ct", "20 ct"], ingredients="enriched wheat flour, water, vegetable shortening, baking powder, salt",
         allergens=["wheat"], tags=["vegetarian"],
         texture=["soft and pliable"], uses=["roll into burritos", "make quesadillas"]),
    dict(key="cauli_rice", category="Frozen", subcategory="Vegetables",
         names=["Riced Cauliflower", "Cauliflower Rice", "Cauliflower Rice Stir-Fry Blend"], variants=["", "Organic"],
         sizes=["12 oz", "24 oz"], ingredients="cauliflower",
         allergens=[], tags=["vegan", "gluten-free", "keto", "paleo", "grain-free", "nut-free"],
         texture=["fluffy", "lightly steamed"], uses=["sauté into fried 'rice'", "serve under curries and stir-fries", "stir into burrito bowls"],
         stats=["3g net carbs per serving", "just one ingredient"]),
    dict(key="white_rice", category="Pantry", subcategory="Rice & Grains",
         names=["Jasmine Rice", "Long Grain White Rice", "Basmati Rice"], variants=[""],
         sizes=["2 lb", "5 lb", "10 lb"], ingredients="rice",
         allergens=[], tags=["vegan", "gluten-free"],
         texture=["fragrant", "fluffy"], uses=["serve with curries", "make fried rice"]),
    dict(key="veg_broth", category="Pantry", subcategory="Broth & Stock",
         names=["Vegetable Broth", "Low Sodium Vegetable Stock"], variants=["", "Organic"],
         sizes=["32 fl oz", "6 x 32 fl oz"], ingredients="vegetable stock (water, carrots, celery, onions), sea salt",
         allergens=[], tags=["vegan", "gluten-free", "nut-free", "dairy-free"],
         texture=["savory", "slow-simmered"], uses=["build soups and risottos", "cook grains for more flavor"]),
    dict(key="chicken_broth", category="Pantry", subcategory="Broth & Stock",
         names=["Chicken Bone Broth", "Chicken Broth"], variants=["", "Low Sodium"],
         sizes=["32 fl oz", "16.9 fl oz"], ingredients="chicken stock, salt, onion, celery",
         allergens=[], tags=["gluten-free", "paleo", "keto"],
         texture=["rich", "slow-simmered"], uses=["sip warm", "build soups and risottos"]),
    dict(key="ice_cream", category="Frozen", subcategory="Desserts",
         names=["Vanilla Bean Ice Cream", "Chocolate Fudge Ice Cream", "Salted Caramel Ice Cream"], variants=[""],
         sizes=["1.5 qt", "1 pt"], ingredients="cream, skim milk, sugar, egg yolks, vanilla bean",
         allergens=["milk", "egg"], tags=["vegetarian", "gluten-free"],
         texture=["super-premium", "dense and creamy"], uses=["scoop onto warm pie", "make milkshakes"]),
    dict(key="nondairy_frozen", category="Frozen", subcategory="Desserts",
         names=["Oatmilk Frozen Dessert", "Coconut Milk Frozen Dessert"], variants=["Vanilla", "Chocolate", "Mint Chip"],
         sizes=["1 pt"], ingredients="oat base (water, oats), cane sugar, coconut oil, sunflower lecithin",
         allergens=[], tags=["vegan", "dairy-free", "nut-free"],
         texture=["scoopable", "creamy"], uses=["scoop into cones", "pair with brownies"]),
    dict(key="gelatin", category="Baking", subcategory="Baking Aids",
         names=["Unflavored Gelatin", "Beef Gelatin Powder"], variants=[""],
         sizes=["1 oz (4 envelopes)", "16 oz"], ingredients="beef gelatin",
         allergens=[], tags=["gluten-free", "paleo", "keto"],
         texture=["unflavored"], uses=["set panna cotta", "make gummies"]),
    dict(key="agar", category="Baking", subcategory="Baking Aids",
         names=["Agar Agar Powder", "Agar Flakes"], variants=[""],
         sizes=["2 oz", "4 oz"], ingredients="agar (seaweed)",
         allergens=[], tags=["vegan", "gluten-free", "keto", "nut-free"],
         texture=["sets firm at room temperature", "flavorless"],
         uses=["sets fruit jellies and panna cotta", "thickens custards and glazes", "makes plant-based gummies"],
         stats=["derived from red seaweed"]),
    # ---------------- Beverages ----------------
    dict(key="kombucha", category="Beverages", subcategory="Functional Drinks",
         names=["Raw Kombucha", "Sparkling Kombucha"], variants=["Ginger Lemon", "Hibiscus", "Original", "Mango Passionfruit"],
         sizes=["16 fl oz", "12 x 12 fl oz"], ingredients="kombucha (filtered water, black tea, green tea, cane sugar, live cultures), ginger juice",
         allergens=[], tags=["vegan", "gluten-free", "nut-free"],
         texture=["effervescent", "tart"], uses=["sip chilled as an afternoon pick-me-up", "mix into mocktails"],
         stats=["billions of living cultures per bottle", "naturally fermented in small batches"]),
    dict(key="electrolyte_mix", category="Beverages", subcategory="Functional Drinks",
         names=["Electrolyte Drink Mix", "Hydration Multiplier Stick Packs"], variants=["Lemon Lime", "Watermelon", "Berry"],
         sizes=["16 ct", "30 ct"], ingredients="sodium citrate, potassium chloride, magnesium, stevia leaf extract, citric acid",
         allergens=[], tags=["vegan", "sugar-free", "keto", "gluten-free"],
         texture=["light and refreshing"], uses=["mix into water after a workout or a hot day", "pack for long hikes"],
         stats=["1000mg sodium, 200mg potassium per stick", "replaces what you lose in sweat"]),
    dict(key="sports_drink", category="Beverages", subcategory="Soft Drinks",
         names=["Sports Drink", "Thirst Quencher"], variants=["Fruit Punch", "Lemon-Lime", "Glacier Blue"],
         sizes=["28 fl oz", "8 x 20 fl oz"], ingredients="water, sugar, dextrose, citric acid, salt, sodium citrate",
         allergens=[], tags=["vegan", "gluten-free"],
         texture=["bright", "sweet"], uses=["chill for game day"]),
    dict(key="cold_brew", category="Beverages", subcategory="Coffee",
         names=["Cold Brew Coffee Concentrate", "Black Cold Brew"], variants=["", "Unsweetened"],
         sizes=["32 fl oz", "11 fl oz"], ingredients="cold brew coffee (water, coffee)",
         allergens=[], tags=["vegan", "gluten-free", "keto", "sugar-free"],
         texture=["smooth, low-acid", "bold"], uses=["pour over ice", "top with oat milk"]),
]

BRANDS = ["Meadow Lane", "Golden Acre", "Harvest Moon", "Blue Ridge Pantry", "Kindred Kitchen", "Northfield Farms",
          "Sprout & Stone", "Little Fern", "Coastal Larder", "Urban Granary", "Sunny Hollow", "Prairie Gold",
          "True Root", "Wild Thyme Co.", "Oak & Ember", "StreetsMart"]  # StreetsMart = the retailer's private label

CATEGORY_CODES = {"Dairy & Eggs": "DAI", "Pantry": "PAN", "Baking": "BAK", "Snacks": "SNK", "Breakfast": "BRK",
                  "Meat & Seafood": "MTS", "Bakery": "BKY", "Frozen": "FRZ", "Beverages": "BEV"}

ALL_ALLERGENS = ["milk", "egg", "wheat", "soy", "peanuts", "tree nuts", "fish", "shellfish", "sesame"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Description generator
# MAGIC
# MAGIC Descriptions mix several sentence templates so no two products read identically. Dietary claims are phrased in
# MAGIC different ways ("Made without milk", "No dairy here", "100% plant-based") — just like real labels — which is
# MAGIC one reason exact keyword matching on tags is brittle. We also add realistic **"may contain" cross-contact warnings**:
# MAGIC a granola that says *"processed in a facility that also handles peanuts"* contains the token `peanut`, which
# MAGIC will fool a naive keyword search for peanut butter.

# COMMAND ----------

CLAIM_PHRASES = {
    "vegan": ["100% plant-based.", "Vegan and made without animal ingredients.", "Certified vegan.", "Plant-based through and through."],
    "vegetarian": ["Vegetarian.", ""],
    "dairy-free": ["Made without milk.", "No dairy here.", "Dairy-free.", "Free from milk and lactose."],
    "gluten-free": ["Certified gluten-free.", "Made in a dedicated gluten-free facility.", "Gluten-free.", "Tested to under 10 ppm gluten."],
    "nut-free": ["Made in a nut-free facility.", "Free from peanuts and tree nuts.", ""],
    "peanut-free": ["Peanut-free.", ""],
    "soy-free": ["Soy-free.", "Contains no soy.", ""],
    "egg-free": ["Egg-free.", "Made without eggs.", ""],
    "keto": ["Keto-friendly.", "Fits a low-carb lifestyle.", ""],
    "paleo": ["Paleo-friendly.", ""],
    "grain-free": ["Grain-free.", ""],
    "sugar-free": ["No sugar added.", "Sugar-free.", "Zero sugar."],
    "low-sodium": ["Lower in sodium.", ""],
    "high-protein": ["A good source of protein.", "Packed with protein.", ""],
    "organic": ["USDA Organic.", "Certified organic ingredients."],
    "non-gmo": ["Non-GMO Project Verified.", "Made with non-GMO ingredients."],
    "kosher": ["Certified kosher.", "Kosher."],
}

OPENERS = [
    "{Texture} {noun} that you can {use0}.",
    "Our {texture} {noun} is made to {use0}.",
    "{Brand} {noun}: {texture}, and ready to {use0}.",
    "Meet your new pantry staple — {texture} {noun} to {use0}.",
    "Crafted in small batches, this {texture} {noun} is perfect to {use0}.",
]
SECONDS = [
    "Try it to {use1}.", "It's also great to {use1}.", "Use it to {use1} or {use2}.", "Home cooks reach for it to {use1}.", "",
]


def _cap(s):
    return s[:1].upper() + s[1:] if s else s


def make_description(rng, arch, brand, display_name, tags, may_contain):
    uses = arch["uses"][:]
    rng.shuffle(uses)
    uses = (uses * 3)[:3]
    texture = rng.choice(arch["texture"])
    noun = display_name.lower()
    fmt = dict(Texture=_cap(texture), texture=texture, noun=noun, Brand=brand,
               use0=uses[0], use1=uses[1], use2=uses[2])
    parts = [rng.choice(OPENERS).format(**fmt), rng.choice(SECONDS).format(**fmt)]
    if arch.get("stats") and rng.random() < 0.8:
        parts.append(_cap(rng.choice(arch["stats"])) + ".")
    # Print a random subset of the claims (labels rarely list everything)
    claims = [rng.choice(CLAIM_PHRASES[t]) for t in tags if t in CLAIM_PHRASES and rng.random() < 0.7]
    parts.extend(c for c in claims if c)
    if may_contain:
        parts.append(rng.choice([
            "Processed in a facility that also handles {x}.", "May contain traces of {x}.",
            "Made on shared equipment with {x}.",
        ]).format(x=" and ".join(may_contain)))
    return re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip()


def generate_products(n_target, seed):
    rng = random.Random(seed)
    per_arch = max(1, n_target // len(ARCHETYPES))
    rows, seen = [], set()
    pid = 100000
    for arch in ARCHETYPES:
        brands = rng.sample(BRANDS, k=min(len(BRANDS), rng.randint(5, 9)))
        attempts = 0
        made = 0
        while made < per_arch and attempts < per_arch * 20:
            attempts += 1
            brand = rng.choice(brands)
            base = rng.choice(arch["names"])
            variant = rng.choice(arch["variants"])
            size = rng.choice(arch["sizes"])
            key = (brand, base, variant, size)
            if key in seen:
                continue
            seen.add(key)
            made += 1
            pid += 1

            tags = list(arch["tags"])
            if "vegan" in tags:
                tags.append("vegetarian")
            if variant == "Organic" or "Organic" in base or rng.random() < 0.15:
                tags.append("organic")
            if rng.random() < 0.35:
                tags.append("non-gmo")
            if rng.random() < 0.25:
                tags.append("kosher")
            tags = sorted(set(tags))

            allergens = list(arch["allergens"])
            # Cross-contact warnings; never warn about something the product already contains
            may_contain = []
            if rng.random() < 0.18:
                candidates = [a for a in ["peanuts", "tree nuts", "milk", "soy", "wheat", "sesame"] if a not in allergens]
                # A product labelled nut-free / gluten-free can't carry a conflicting warning
                if "nut-free" in tags or "peanut-free" in tags:
                    candidates = [a for a in candidates if a not in ("peanuts", "tree nuts")]
                if "gluten-free" in tags:
                    candidates = [a for a in candidates if a != "wheat"]
                if "dairy-free" in tags or "vegan" in tags:
                    candidates = [a for a in candidates if a != "milk"]
                if candidates:
                    may_contain = rng.sample(candidates, k=1)

            display = " ".join(x for x in [variant if variant not in ("", "Organic") else "", base] if x)
            if variant == "Organic" and "Organic" not in base:
                display = "Organic " + display
            product_name = f"{display}, {size}"
            sku = f"FC-{CATEGORY_CODES[arch['category']]}-{pid % 100000:05d}"
            upc = f"0{rng.randint(10**10, 10**11 - 1)}"
            price = round(rng.uniform(1.49, 14.99), 2)
            rows.append(dict(
                product_id=pid, sku=sku, upc=upc, brand=brand, product_name=product_name, size=size,
                category=arch["category"], subcategory=arch["subcategory"], archetype=arch["key"],
                description=make_description(rng, arch, brand, display, tags, may_contain),
                ingredients=arch["ingredients"], allergens=allergens, may_contain=may_contain,
                dietary_tags=tags, price=price,
            ))
    return rows


products = generate_products(NUM_PRODUCTS, SEED)
print(f"Generated {len(products):,} products across {len(ARCHETYPES)} archetypes")
for p in products[:3]:
    print("-", p["brand"], "|", p["product_name"], "|", p["sku"], "\n ", p["description"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Unity Catalog
# MAGIC
# MAGIC Delta is the system of record. Notebook 02 embeds these rows with `ai_query` and loads them into Lakebase.
# MAGIC (In production you could also keep Lakebase in sync with a **synced table** — see the blog.)

# COMMAND ----------

from pyspark.sql import types as T

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

product_schema = T.StructType([
    T.StructField("product_id", T.LongType()), T.StructField("sku", T.StringType()), T.StructField("upc", T.StringType()),
    T.StructField("brand", T.StringType()), T.StructField("product_name", T.StringType()), T.StructField("size", T.StringType()),
    T.StructField("category", T.StringType()), T.StructField("subcategory", T.StringType()),
    T.StructField("archetype", T.StringType()), T.StructField("description", T.StringType()),
    T.StructField("ingredients", T.StringType()), T.StructField("allergens", T.ArrayType(T.StringType())),
    T.StructField("may_contain", T.ArrayType(T.StringType())), T.StructField("dietary_tags", T.ArrayType(T.StringType())),
    T.StructField("price", T.DoubleType()),
])
(spark.createDataFrame(products, product_schema)
 .write.mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable(f"{CATALOG}.{SCHEMA}.grocery_products"))

display(spark.table(f"{CATALOG}.{SCHEMA}.grocery_products").limit(20))

# COMMAND ----------

display(spark.sql(f"""
  SELECT category, count(*) AS products,
         count_if(array_contains(dietary_tags, 'gluten-free')) AS gluten_free,
         count_if(array_contains(dietary_tags, 'vegan'))       AS vegan,
         count_if(size(may_contain) > 0)                       AS with_cross_contact_warning
  FROM {CATALOG}.{SCHEMA}.grocery_products
  GROUP BY category ORDER BY products DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export the CSV
# MAGIC
# MAGIC Same format that notebook 02 loads. List columns are JSON arrays.

# COMMAND ----------

import csv
import json
import os

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.blog_assets")
os.makedirs(OUTPUT_DIR, exist_ok=True)
product_cols = [f.name for f in product_schema]
with open(f"{OUTPUT_DIR}/grocery_products.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(product_cols)
    for p in products:
        wr.writerow([json.dumps(p[c]) if isinstance(p[c], list) else p[c] for c in product_cols])
print("Wrote grocery_products.csv to", OUTPUT_DIR)