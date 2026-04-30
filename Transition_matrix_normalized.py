
import pandas as pd

df = pd.read_csv("dataset_clean.csv")

print("\n==============================")
print("16 — ANALYSE DES TRANSITIONS")
print("==============================")

# enlever les valeurs inutiles
df_trans = df[
    (df["previous_category"] != "Unknown") &
    (df["category"] != "Unknown")
].copy()


# =========================
# 1. MATRICE BRUTE
# =========================

transition = pd.crosstab(
    df_trans["previous_category"],
    df_trans["category"]
)

print("\nTransitions (brut) :")
print(transition)


# =========================
# 2. MATRICE NORMALISÉE
# =========================

transition_norm = pd.crosstab(
    df_trans["previous_category"],
    df_trans["category"],
    normalize="index"
)

print("\nTransitions (probabilités) :")
print(transition_norm.round(3))


# =========================
# 3. TOP TRANSITIONS
# =========================

print("\nTop transitions les plus fréquentes :")

transition_long = transition_norm.stack().reset_index()
transition_long.columns = ["from", "to", "prob"]

top_transitions = transition_long.sort_values(
    by="prob",
    ascending=False
).head(10)

print(top_transitions.round(3))


# =========================
# 4. STABILITÉ (IMPORTANT)
# =========================

print("\nStabilité des catégories (rester dans la même catégorie) :")

stability = []

for cat in transition_norm.index:
    if cat in transition_norm.columns:
        prob_same = transition_norm.loc[cat, cat]
    else:
        prob_same = 0

    stability.append({
        "category": cat,
        "stay_probability": prob_same
    })

stability_df = pd.DataFrame(stability).sort_values(
    by="stay_probability",
    ascending=False
)

print(stability_df.round(3))


# =========================
# 5. SAUVEGARDE
# =========================

transition_norm.to_csv("transition_matrix_normalized.csv")
top_transitions.to_csv("top_transitions.csv", index=False)
stability_df.to_csv("stability_by_category.csv", index=False)

print("\nFichiers créés :")
print("- transition_matrix_normalized.csv")
print("- top_transitions.csv")
print("- stability_by_category.csv")
