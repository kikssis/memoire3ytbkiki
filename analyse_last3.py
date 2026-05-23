# ============================================================
# 16.3 — MODÈLE SÉQUENTIEL LAST 3
# P(category_t | history_1, history_2, history_3)
# ============================================================

import pandas as pd


# ============================================================
# 1. CHARGER LE DATASET
# ============================================================

df = pd.read_csv("dataset_clean.csv")

print("Dataset chargé :", len(df), "lignes")


# ============================================================
# 2. NETTOYAGE MINIMAL
# ============================================================

if "category_clean" not in df.columns:
    df["category_clean"] = df["category"]

needed_cols = [
    "category_clean",
    "history_1",
    "history_2",
    "history_3",
    "condition"
]

for col in needed_cols:
    if col not in df.columns:
        df[col] = "Unknown"

    df[col] = df[col].fillna("Unknown")


# ============================================================
# 3. CONSTRUIRE L'ÉTAT LAST 3
# ============================================================

df["state_last_3"] = (
    df["history_3"].astype(str)
    + " -> "
    + df["history_2"].astype(str)
    + " -> "
    + df["history_1"].astype(str)
)

print("\nExemples d'états last 3 :")
print(df["state_last_3"].head(10))


# ============================================================
# 4. TABLE P(category | last_3)
# ============================================================

last3_counts = pd.crosstab(
    df["state_last_3"],
    df["category_clean"]
)

last3_probabilities = pd.crosstab(
    df["state_last_3"],
    df["category_clean"],
    normalize="index"
)

print("\n==============================")
print("TABLE LAST 3 — BRUT")
print("==============================")
print(last3_counts.head(20))

print("\n==============================")
print("TABLE LAST 3 — PROBABILITÉS")
print("==============================")
print(last3_probabilities.head(20).round(3))


# ============================================================
# 5. SAUVEGARDE
# ============================================================

last3_counts.to_csv(
    "last3_transition_counts.csv",
    encoding="utf-8-sig"
)

last3_probabilities.to_csv(
    "last3_transition_probabilities.csv",
    encoding="utf-8-sig"
)

print("\nFichiers créés :")
print("- last3_transition_counts.csv")
print("- last3_transition_probabilities.csv")


# ============================================================
# 6. TOP PRÉDICTIONS PAR ÉTAT
# ============================================================

top_rows = []

for state in last3_probabilities.index:

    probs = last3_probabilities.loc[state]
    counts = last3_counts.loc[state]

    top_category = probs.idxmax()
    top_probability = probs.max()
    total_observations = counts.sum()

    top_rows.append({
        "state_last_3": state,
        "top_next_category": top_category,
        "top_probability": top_probability,
        "total_observations": total_observations
    })

top_last3 = pd.DataFrame(top_rows)

# On garde les états suffisamment observés
top_last3_filtered = top_last3[
    top_last3["total_observations"] >= 5
].sort_values(
    by=["top_probability", "total_observations"],
    ascending=False
)

print("\n==============================")
print("TOP ÉTATS LAST 3")
print("==============================")
print(top_last3_filtered.head(30).to_string(index=False))

top_last3.to_csv(
    "last3_top_predictions_all.csv",
    index=False,
    encoding="utf-8-sig"
)

top_last3_filtered.to_csv(
    "last3_top_predictions_filtered.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nFichiers créés :")
print("- last3_top_predictions_all.csv")
print("- last3_top_predictions_filtered.csv")


# ============================================================
# 7. LAST 3 PAR CONDITION
# ============================================================

conditions = sorted(df["condition"].unique())

for condition in conditions:

    subset = df[df["condition"] == condition]

    if len(subset) == 0:
        continue

    condition_probs = pd.crosstab(
        subset["state_last_3"],
        subset["category_clean"],
        normalize="index"
    )

    safe_condition = condition.replace(" ", "_").replace("/", "_")

    filename = f"last3_probabilities_{safe_condition}.csv"

    condition_probs.to_csv(
        filename,
        encoding="utf-8-sig"
    )

    print("Fichier créé :", filename)


# ============================================================
# 8. LECTURE RAPIDE
# ============================================================

print("\n==============================")
print("LECTURE RAPIDE")
print("==============================")

print(
    "Cette analyse montre quelles catégories apparaissent le plus souvent "
    "après une séquence de trois catégories précédentes."
)

print(
    "Les états avec une forte probabilité indiquent des trajectoires "
    "séquentielles relativement prévisibles."
)