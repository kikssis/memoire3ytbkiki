import pandas as pd

# =========================
# CHARGER LE DATASET
# =========================

df = pd.read_csv("dataset_clean.csv")

# =========================
# FILTRER CONDITION NEUTRE
# =========================

neutral_df = df[
    df["condition"].str.contains("neutral", case=False, na=False)
].copy()

print("\n==============================")
print("RÉPARTITION CONDITION NEUTRE")
print("==============================")

print("\nNombre de lignes :", len(neutral_df))

# =========================
# PROPORTIONS DES CATÉGORIES
# =========================

proportions = (
    neutral_df["category"]
    .value_counts(normalize=True)
    .reset_index()
)

proportions.columns = ["category", "proportion"]

# Pourcentage
proportions["percentage"] = proportions["proportion"] * 100

# =========================
# AFFICHAGE
# =========================

print("\nProportions des catégories :\n")

print(
    proportions
    .round({
        "proportion": 4,
        "percentage": 2
    })
)

# =========================
# SAUVEGARDE CSV
# =========================

output_file = "neutral_category_proportions.csv"

proportions.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nFichier créé : {output_file}")