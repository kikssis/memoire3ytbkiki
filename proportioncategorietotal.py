import pandas as pd

df = pd.read_csv("dataset_clean.csv")

# Enlever Unknown si tu veux uniquement les catégories interprétables
df_clean = df[df["category_clean"] != "Unknown"].copy()

distribution = (
    df_clean["category_clean"]
    .value_counts()
    .reset_index()
)

distribution.columns = ["category", "count"]

distribution["percentage"] = (
    distribution["count"] / distribution["count"].sum() * 100
)

print(distribution)

distribution.to_csv(
    "category_distribution_global.csv",
    index=False,
    encoding="utf-8-sig"
)