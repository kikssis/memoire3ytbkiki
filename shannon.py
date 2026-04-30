import pandas as pd
import numpy as np

df = pd.read_csv("dataset_clean.csv")

print("\n==============================")
print("18 — ENTROPIE DE SHANNON")
print("==============================")


# =========================
# 1. FONCTION SHANNON
# =========================

def shannon(series):
    p = series.value_counts(normalize=True)
    return -np.sum(p * np.log2(p))


# =========================
# 2. CALCUL PAR CONDITION
# =========================

results = []

for condition in df["condition"].unique():

    subset = df[df["condition"] == condition]

    # enlever Unknown pour éviter biais
    subset = subset[subset["category"] != "Unknown"]

    h = shannon(subset["category"])

    results.append({
        "condition": condition,
        "shannon_entropy": h
    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="shannon_entropy",
    ascending=False
).reset_index(drop=True)

print("\nEntropie par condition :")
print(results_df.round(3))


# =========================
# 3. NORMALISATION (OPTIONNEL MAIS TOP)
# =========================

# max possible = log2(nombre de catégories)
n_categories = df["category"].nunique()
max_entropy = np.log2(n_categories)

results_df["normalized_entropy"] = results_df["shannon_entropy"] / max_entropy

print("\nEntropie normalisée :")
print(results_df.round(3))


# =========================
# 4. INTERPRÉTATION AUTO
# =========================

print("\nInterprétation :")

for _, row in results_df.iterrows():
    c = row["condition"]
    h = row["shannon_entropy"]

    if h > 2:
        level = "diversité élevée"
    elif h > 1:
        level = "diversité modérée"
    else:
        level = "faible diversité (bulle)"

    print(f"{c} → {h:.3f} → {level}")


# =========================
# 5. SAUVEGARDE
# =========================

results_df.to_csv(
    "shannon_entropy_by_condition.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nFichier créé : shannon_entropy_by_condition.csv")
