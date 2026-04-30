import pandas as pd

# ===============================
# LOAD DATA
# ===============================

df = pd.read_csv("dataset_clean.csv")

print("Dataset :", len(df))


# ===============================
# ANALYSE PAR DEPTH
# ===============================

# enlever Unknown
df = df[df["category_clean"] != "Unknown"].copy()

# distribution par depth
depth_distribution = (
   df.groupby("depth")["category_clean"]
   .value_counts(normalize=True)
   .rename("prob")
   .reset_index()
)

print("\nDistribution par depth :")
print(depth_distribution.head(20))


# ===============================
# FOCUS : PEOPLE & BLOGS
# ===============================

people_trend = (
   depth_distribution[
       depth_distribution["category_clean"] == "People & Blogs"
   ]
)

print("\nEvolution People & Blogs :")
print(people_trend)


# ===============================
# SAVE
# ===============================

depth_distribution.to_csv("depth_distribution.csv", index=False)
people_trend.to_csv("people_trend.csv", index=False)

print("\nFichiers créés :")
print("- depth_distribution.csv")
print("- people_trend.csv")