# ============================================================
# 15.6 — ANALYSE QUASI-CAUSALE SIMPLIFIÉE
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt


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

df["category_clean"] = df["category_clean"].fillna("Unknown")
df["condition"] = df["condition"].fillna("neutral")


# ============================================================
# 3. FONCTION DE PROBABILITÉ
# ============================================================

def compute_prob(data, condition, target_category):
    subset = data[data["condition"] == condition]

    if len(subset) == 0:
        return 0

    return (subset["category_clean"] == target_category).mean()


# ============================================================
# 4. PARAMÈTRES
# ============================================================

analyses = {
    "sport_history": "Sports",
    "gaming_history": "Gaming",
    "news_history": "News & Politics",
    "science_history": "Science & Technology",
}

intensities = [1, 3, 8]

rows = []


# ============================================================
# 5. CALCUL DES EFFETS
# ============================================================

for history_type, target_category in analyses.items():

    p_neutral = compute_prob(df, "neutral", target_category)

    for intensity in intensities:
        condition = f"{history_type}_{intensity}"

        p_condition = compute_prob(df, condition, target_category)
        difference = p_condition - p_neutral

        if p_neutral > 0:
            ratio = p_condition / p_neutral
        else:
            ratio = None

        rows.append({
            "history_type": history_type,
            "target_category": target_category,
            "intensity": intensity,
            "condition": condition,
            "p_condition": p_condition,
            "p_neutral": p_neutral,
            "difference": difference,
            "ratio": ratio
        })


results = pd.DataFrame(rows)


# ============================================================
# 6. AFFICHAGE DES RÉSULTATS
# ============================================================

print("\n==============================")
print("ANALYSE QUASI-CAUSALE")
print("==============================")

print(results.to_string(index=False))


# ============================================================
# 7. SAUVEGARDE CSV
# ============================================================

results.to_csv(
    "analyse_causale_simplifiee.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nFichier créé : analyse_causale_simplifiee.csv")


# ============================================================
# 8. GRAPHIQUES PAR CATÉGORIE
# ============================================================

for history_type, target_category in analyses.items():

    subset = results[results["history_type"] == history_type]

    plt.figure()
    plt.plot(
        subset["intensity"],
        subset["difference"],
        marker="o"
    )

    plt.title(f"Effet quasi-causal — {target_category}")
    plt.xlabel("Intensité de l'historique")
    plt.ylabel("Différence de probabilité vs neutral")
    plt.xticks(intensities)
    plt.grid(True)

    safe_name = history_type.replace("_history", "")

    filename = f"effet_causal_{safe_name}.png"

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print("Graphique créé :", filename)


# ============================================================
# 9. GRAPHIQUE GLOBAL
# ============================================================

plt.figure()

for history_type, target_category in analyses.items():
    subset = results[results["history_type"] == history_type]

    plt.plot(
        subset["intensity"],
        subset["difference"],
        marker="o",
        label=target_category
    )

plt.title("Effet quasi-causal par intensité")
plt.xlabel("Intensité de l'historique")
plt.ylabel("Différence de probabilité vs neutral")
plt.xticks(intensities)
plt.legend()
plt.grid(True)

plt.savefig("effet_causal_global.png", dpi=300, bbox_inches="tight")
plt.close()

print("Graphique créé : effet_causal_global.png")


# ============================================================
# 10. CONCLUSION AUTOMATIQUE
# ============================================================

print("\n==============================")
print("LECTURE RAPIDE")
print("==============================")

for history_type, target_category in analyses.items():
    subset = results[results["history_type"] == history_type]
    best_row = subset.loc[subset["difference"].idxmax()]

    print(
        f"{target_category} : effet maximal à intensité "
        f"{int(best_row['intensity'])} "
        f"(différence = {best_row['difference']:.3f})"
    )