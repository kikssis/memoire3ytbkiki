import pandas as pd

df = pd.read_csv("dataset_clean.csv")

print("\n==============================")
print("15.5 — ANALYSE D’INFLUENCE")
print("==============================")

# Proportion de chaque catégorie dans chaque condition
prop = pd.crosstab(
    df["condition"],
    df["category"],
    normalize="index"
)

mapping = {
    "sport": "Sports",
    "music": "Music",
    "gaming": "Gaming",
    "news": "News & Politics",
    "science": "Science & Technology",
    "people": "People & Blogs",
    "people_blog": "People & Blogs",
}

results = []

for condition in prop.index:
    condition_lower = str(condition).lower()

    target_cat = None

    for key, youtube_category in mapping.items():
        if condition_lower.startswith(key):
            target_cat = youtube_category
            break

    if target_cat is None:
        continue

    if target_cat not in prop.columns:
        continue

    p_condition = prop.loc[condition, target_cat]

    other_conditions = prop.drop(index=condition)
    p_others_mean = other_conditions[target_cat].mean()

    difference = p_condition - p_others_mean

    if p_others_mean > 0:
        ratio = p_condition / p_others_mean
    else:
        ratio = None

    results.append({
        "condition": condition,
        "target_category": target_cat,
        "p_condition": p_condition,
        "p_others_mean": p_others_mean,
        "difference": difference,
        "ratio": ratio
    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["target_category", "condition"]
).reset_index(drop=True)

print("\nRésultats :")
print(results_df.round(3))

results_df.to_csv(
    "analyse_influence_15_5.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nFichier créé : analyse_influence_15_5.csv")
