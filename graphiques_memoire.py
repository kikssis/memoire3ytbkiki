# ============================================================
# GRAPHIQUES MÉMOIRE
# 1. Heatmap transitions
# 2. Barplot influence
# 3. Courbe entropie vs intensité
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. CHARGER DATASET
# ============================================================

df = pd.read_csv("dataset_clean.csv")

if "category_clean" not in df.columns:
    df["category_clean"] = df["category"]

df["category_clean"] = df["category_clean"].fillna("Unknown")
df["previous_category"] = df["previous_category"].fillna("Unknown")
df["condition"] = df["condition"].fillna("neutral")

print("Dataset chargé :", len(df))


# ============================================================
# 2. HEATMAP DES TRANSITIONS
# À mettre dans mémoire : 16 — Analyse des transitions
# ============================================================

transition = pd.crosstab(
    df["previous_category"],
    df["category_clean"],
    normalize="index"
)

# enlever Unknown si présent
transition = transition.drop(index="Unknown", errors="ignore")
transition = transition.drop(columns="Unknown", errors="ignore")

plt.figure(figsize=(12, 10))
plt.imshow(transition, aspect="auto")

plt.colorbar(label="Probabilité de transition")
plt.title("Heatmap des transitions entre catégories")
plt.xlabel("Catégorie recommandée")
plt.ylabel("Catégorie précédente")

plt.xticks(
    ticks=np.arange(len(transition.columns)),
    labels=transition.columns,
    rotation=90
)

plt.yticks(
    ticks=np.arange(len(transition.index)),
    labels=transition.index
)

plt.tight_layout()
plt.savefig("heatmap_transitions.png", dpi=300, bbox_inches="tight")
plt.close()

print("Graphique créé : heatmap_transitions.png")


# ============================================================
# 3. BARPLOT INFLUENCE
# À mettre dans mémoire : 15.5 ou 15.6 — Analyse d’influence
# ============================================================

target_map = {
    "sport_history": "Sports",
    "gaming_history": "Gaming",
    "news_history": "News & Politics",
    "science_history": "Science & Technology",
}

def compute_prob(data, condition, target_category):
    subset = data[data["condition"] == condition]
    if len(subset) == 0:
        return 0
    return (subset["category_clean"] == target_category).mean()

rows = []

for history_type, target in target_map.items():
    for intensity in [1, 3, 8]:
        condition = f"{history_type}_{intensity}"

        p_condition = compute_prob(df, condition, target)
        p_others = compute_prob(df[df["condition"] != condition], condition, target)

        # meilleure version : comparaison avec neutral
        p_neutral = compute_prob(df, "neutral", target)

        rows.append({
            "condition": condition,
            "history_type": history_type,
            "target_category": target,
            "intensity": intensity,
            "p_condition": p_condition,
            "p_neutral": p_neutral,
            "difference": p_condition - p_neutral
        })

influence = pd.DataFrame(rows)

influence.to_csv(
    "barplot_influence_data.csv",
    index=False,
    encoding="utf-8-sig"
)

plt.figure(figsize=(12, 6))

labels = influence["condition"]
values = influence["difference"]

plt.bar(labels, values)

plt.title("Effet de l’historique sur la catégorie cible")
plt.xlabel("Condition")
plt.ylabel("Différence de probabilité vs neutral")
plt.xticks(rotation=45, ha="right")
plt.axhline(0, linewidth=1)

plt.tight_layout()
plt.savefig("barplot_influence.png", dpi=300, bbox_inches="tight")
plt.close()

print("Graphique créé : barplot_influence.png")


# ============================================================
# 4. COURBE ENTROPIE VS INTENSITÉ
# À mettre dans mémoire : 18 — Entropie de Shannon
# ============================================================

def shannon_entropy(series):
    probs = series.value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs))

entropy_rows = []

for history_type in ["sport_history", "gaming_history", "news_history", "science_history"]:
    for intensity in [1, 3, 8]:
        condition = f"{history_type}_{intensity}"

        subset = df[df["condition"] == condition]

        if len(subset) == 0:
            continue

        entropy = shannon_entropy(subset["category_clean"])

        entropy_rows.append({
            "history_type": history_type,
            "intensity": intensity,
            "condition": condition,
            "shannon_entropy": entropy
        })

entropy_df = pd.DataFrame(entropy_rows)

entropy_df.to_csv(
    "entropie_vs_intensite.csv",
    index=False,
    encoding="utf-8-sig"
)

plt.figure(figsize=(10, 6))

for history_type in entropy_df["history_type"].unique():
    subset = entropy_df[entropy_df["history_type"] == history_type]

    plt.plot(
        subset["intensity"],
        subset["shannon_entropy"],
        marker="o",
        label=history_type
    )

plt.title("Évolution de l’entropie selon l’intensité de l’historique")
plt.xlabel("Intensité de l’historique")
plt.ylabel("Entropie de Shannon")
plt.xticks([1, 3, 8])
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("courbe_entropie_vs_intensite.png", dpi=300, bbox_inches="tight")
plt.close()

print("Graphique créé : courbe_entropie_vs_intensite.png")


# ============================================================
# 5. FIN
# ============================================================

print("\nTous les graphiques ont été créés :")
print("- heatmap_transitions.png")
print("- barplot_influence.png")
print("- courbe_entropie_vs_intensite.png")