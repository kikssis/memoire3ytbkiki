import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# ============================================================
# CHARGER LE DATASET
# ============================================================

df = pd.read_csv("dataset_clean.csv")

print("Dataset chargé :", len(df), "lignes")


# ============================================================
# PRÉPARATION DES VARIABLES
# ============================================================

# Si category_clean n'existe pas, on utilise category
if "category_clean" not in df.columns:
    df["category_clean"] = df["category"]

# Créer state_last_3 pour l'analyse 7.3
needed_cols = [
    "condition",
    "category_clean",
    "previous_category",
    "history_1",
    "history_2",
    "history_3"
]

for col in needed_cols:
    if col not in df.columns:
        df[col] = "Unknown"
    df[col] = df[col].fillna("Unknown")

df["state_last_3"] = (
    df["history_3"].astype(str)
    + " -> "
    + df["history_2"].astype(str)
    + " -> "
    + df["history_1"].astype(str)
)


# ============================================================
# FONCTION CHI-DEUX + CRAMER'S V
# ============================================================

def run_chi2(data, row_var, col_var, test_name, min_row_count=None):
    """
    Fait un test du chi-deux entre deux variables catégorielles.
    Calcule aussi la p-value et le V de Cramér.
    """

    temp = data.copy()

    # Retirer les valeurs Unknown
    temp = temp[
        (temp[row_var] != "Unknown") &
        (temp[col_var] != "Unknown")
    ]

    # Pour 7.3 : garder seulement les séquences observées au moins 5 fois
    if min_row_count is not None:
        counts = temp[row_var].value_counts()
        valid_rows = counts[counts >= min_row_count].index
        temp = temp[temp[row_var].isin(valid_rows)]

    # Tableau de contingence
    table = pd.crosstab(temp[row_var], temp[col_var])

    # Test chi-deux
    chi2, p_value, dof, expected = chi2_contingency(table)

    # Cramer's V
    n = table.to_numpy().sum()
    r, k = table.shape
    cramers_v = np.sqrt(chi2 / (n * min(r - 1, k - 1)))

    # Vérifier les cellules avec effectif attendu < 5
    expected_df = pd.DataFrame(
        expected,
        index=table.index,
        columns=table.columns
    )

    low_expected = (expected_df < 5).sum().sum()
    total_cells = expected_df.size
    low_expected_percent = (low_expected / total_cells) * 100

    print("\n===================================================")
    print(test_name)
    print("===================================================")
    print("Variables :", row_var, "x", col_var)
    print("Nombre d'observations :", n)
    print("Chi-deux :", round(chi2, 3))
    print("Degrés de liberté :", dof)
    print("P-value :", p_value)
    print("V de Cramér :", round(cramers_v, 3))
    print("Cellules attendues < 5 :", low_expected, "/", total_cells)
    print("Pourcentage cellules faibles :", round(low_expected_percent, 2), "%")

    if p_value < 0.001:
        conclusion = "Significatif, p < 0.001"
    elif p_value < 0.05:
        conclusion = "Significatif, p < 0.05"
    else:
        conclusion = "Non significatif"

    print("Conclusion :", conclusion)

    return {
        "test": test_name,
        "variables": f"{row_var} x {col_var}",
        "n": n,
        "chi2": chi2,
        "dof": dof,
        "p_value": p_value,
        "cramers_v": cramers_v,
        "low_expected_cells": low_expected,
        "total_cells": total_cells,
        "low_expected_percent": low_expected_percent,
        "conclusion": conclusion
    }


# ============================================================
# LANCER LES 3 TESTS
# ============================================================

results = []

# 7.1 — condition expérimentale x catégorie recommandée
results.append(
    run_chi2(
        data=df,
        row_var="condition",
        col_var="category_clean",
        test_name="7.1 — Condition expérimentale x catégorie recommandée"
    )
)

# 7.2 — catégorie précédente x catégorie actuelle
results.append(
    run_chi2(
        data=df,
        row_var="previous_category",
        col_var="category_clean",
        test_name="7.2 — Catégorie précédente x catégorie actuelle"
    )
)

# 7.3 — trois dernières catégories x catégorie actuelle
results.append(
    run_chi2(
        data=df,
        row_var="state_last_3",
        col_var="category_clean",
        test_name="7.3 — Trois dernières catégories x catégorie actuelle",
        min_row_count=5
    )
)


# ============================================================
# SAUVEGARDE DU RÉSUMÉ
# ============================================================

results_df = pd.DataFrame(results)

print("\n===================================================")
print("RÉSUMÉ DES TESTS")
print("===================================================")
print(results_df.round(4))

results_df.to_csv(
    "chi_square_results_7_1_7_2_7_3.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nFichier créé : chi_square_results_7_1_7_2_7_3.csv")