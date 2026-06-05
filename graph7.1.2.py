import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# LECTURE DES DONNÉES
# ===============================

df = pd.read_csv("analyse_influence_15_5.csv")

# ===============================
# GRAPHIQUE ANALYSE D'INFLUENCE
# ===============================

plt.figure(figsize=(12, 6))

x = range(len(df))

plt.bar(
    x,
    df["ratio"]
)

plt.xticks(
    x,
    df["condition"],
    rotation=45,
    ha="right"
)

plt.ylabel("Ratio")
plt.xlabel("Condition")
plt.title("Analyse d'influence des catégories ciblées")

# ligne ratio = 1
plt.axhline(
    y=1,
    linestyle="--"
)

plt.tight_layout()

# sauvegarde
plt.savefig(
    "graphique_analyse_influence.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nGraphique créé : graphique_analyse_influence.png")