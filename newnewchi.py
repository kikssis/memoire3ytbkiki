import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact


# ============================================================
# CHARGER LE DATASET
# ============================================================

df = pd.read_csv("dataset_clean.csv")

print("Dataset chargé :", len(df), "lignes")


# ============================================================
# PRÉPARATION
# ============================================================

if "category_clean" not in df.columns:
    df["category_clean"] = df["category"]

for col in ["condition", "category_clean"]:
    if col not in df.columns:
        df[col] = "Unknown"
    df[col] = df[col].fillna("Unknown").astype(str)

df = df[
    (df["condition"] != "Unknown") &
    (df["category_clean"] != "Unknown")
].copy()


# ============================================================
# FONCTION INTERVALLE DE CONFIANCE WILSON 95 %
# ============================================================

def wilson_ci(successes, total, z=1.96):
    if total == 0:
        return np.nan, np.nan

    p = successes / total
    denominator = 1 + (z ** 2 / total)

    center = p + (z ** 2 / (2 * total))
    margin = z * np.sqrt((p * (1 - p) / total) + (z ** 2 / (4 * total ** 2)))

    low = (center - margin) / denominator
    high = (center + margin) / denominator

    return low, high


# ============================================================
# 7.2 — CHI-DEUX GLOBAL
# ============================================================

print("\n===================================================")
print("7.2 — Condition expérimentale × catégorie recommandée")
print("===================================================")

table_global = pd.crosstab(
    df["condition"],
    df["category_clean"]
)

chi2, p_value, dof, expected = chi2_contingency(table_global)

n = table_global.to_numpy().sum()
r, k = table_global.shape
cramers_v = np.sqrt(chi2 / (n * min(r - 1, k - 1)))

expected_df = pd.DataFrame(
    expected,
    index=table_global.index,
    columns=table_global.columns
)

low_expected = (expected_df < 5).sum().sum()
total_cells = expected_df.size
low_expected_percent = (low_expected / total_cells) * 100

print("\nNombre d'observations :", n)
print("Chi-deux :", round(chi2, 3))
print("Degrés de liberté :", dof)
print("P-value :", p_value)
print("V de Cramér :", round(cramers_v, 3))
print("Cellules attendues < 5 :", low_expected, "/", total_cells)
print("Pourcentage cellules faibles :", round(low_expected_percent, 2), "%")

if p_value < 0.001:
    conclusion_global = "Significatif, p < 0.001"
elif p_value < 0.05:
    conclusion_global = "Significatif, p < 0.05"
else:
    conclusion_global = "Non significatif"

print("Conclusion :", conclusion_global)

table_global.to_csv(
    "7_2_table_condition_category.csv",
    encoding="utf-8-sig"
)


# ============================================================
# 7.2.3 — TESTS CIBLÉS FISHER
# ============================================================

comparisons = {
    "Gaming": {
        "target_category": "Gaming",
        "conditions": [
            "gaming_history_1",
            "gaming_history_3",
            "gaming_history_8"
        ]
    },
    "Sports": {
        "target_category": "Sports",
        "conditions": [
            "sport_history_1",
            "sport_history_3",
            "sport_history_8"
        ]
    },
    "News & Politics": {
        "target_category": "News & Politics",
        "conditions": [
            "news_history_1",
            "news_history_3",
            "news_history_8"
        ]
    },
    "Science & Technology": {
        "target_category": "Science & Technology",
        "conditions": [
            "science_history_1",
            "science_history_3",
            "science_history_8"
        ]
    }
}

target_results = []

print("\n===================================================")
print("7.2.3 — Comparaisons ciblées avec la condition neutre")
print("===================================================")

for group_name, info in comparisons.items():
    target_category = info["target_category"]

    for condition in info["conditions"]:

        temp = df[
            df["condition"].isin(["neutral", condition])
        ].copy()

        if temp.empty:
            continue

        temp["is_target"] = (
            temp["category_clean"] == target_category
        ).astype(int)

        table = pd.crosstab(
            temp["condition"],
            temp["is_target"]
        )

        for col in [0, 1]:
            if col not in table.columns:
                table[col] = 0

        if "neutral" not in table.index or condition not in table.index:
            continue

        target_neutral = table.loc["neutral", 1]
        non_target_neutral = table.loc["neutral", 0]

        target_condition = table.loc[condition, 1]
        non_target_condition = table.loc[condition, 0]

        fisher_table = [
            [target_neutral, non_target_neutral],
            [target_condition, non_target_condition]
        ]

        odds_ratio, fisher_p = fisher_exact(fisher_table)

        n_neutral = target_neutral + non_target_neutral
        n_condition = target_condition + non_target_condition

        p_neutral = target_neutral / n_neutral if n_neutral > 0 else np.nan
        p_condition = target_condition / n_condition if n_condition > 0 else np.nan

        difference = p_condition - p_neutral

        ratio = (
            p_condition / p_neutral
            if p_neutral > 0
            else np.nan
        )

        ci_neutral_low, ci_neutral_high = wilson_ci(
            target_neutral,
            n_neutral
        )

        ci_condition_low, ci_condition_high = wilson_ci(
            target_condition,
            n_condition
        )

        if fisher_p < 0.001:
            conclusion = "Significatif, p < 0.001"
        elif fisher_p < 0.05:
            conclusion = "Significatif, p < 0.05"
        else:
            conclusion = "Non significatif"

        print("\n---------------------------------------------------")
        print("Catégorie étudiée :", target_category)
        print("Comparaison : neutral vs", condition)

        print("\nTableau 2x2 :")
        print(pd.DataFrame(
            fisher_table,
            index=["neutral", condition],
            columns=[target_category, "Non-" + target_category]
        ))

        print("\nProportion neutral :", round(p_neutral, 4))
        print("IC 95 % neutral :", round(ci_neutral_low, 4), "-", round(ci_neutral_high, 4))

        print("Proportion condition :", round(p_condition, 4))
        print("IC 95 % condition :", round(ci_condition_low, 4), "-", round(ci_condition_high, 4))

        print("Différence :", round(difference, 4))
        print("Ratio :", round(ratio, 4) if not np.isnan(ratio) else "Non calculable")
        print("Test utilisé : Fisher exact")
        print("P-value :", fisher_p)
        print("Conclusion :", conclusion)

        target_results.append({
            "target_category": target_category,
            "condition": condition,

            "neutral_target_count": target_neutral,
            "neutral_non_target_count": non_target_neutral,
            "condition_target_count": target_condition,
            "condition_non_target_count": non_target_condition,

            "n_neutral": n_neutral,
            "n_condition": n_condition,

            "p_neutral": p_neutral,
            "p_condition": p_condition,

            "ci_neutral_low": ci_neutral_low,
            "ci_neutral_high": ci_neutral_high,
            "ci_condition_low": ci_condition_low,
            "ci_condition_high": ci_condition_high,

            "difference": difference,
            "ratio": ratio,

            "test_used": "Fisher exact",
            "odds_ratio": odds_ratio,
            "p_value": fisher_p,
            "conclusion": conclusion
        })


# ============================================================
# SAUVEGARDES
# ============================================================

global_results = pd.DataFrame([{
    "test": "7.2 — Condition expérimentale x catégorie recommandée",
    "n": n,
    "chi2": chi2,
    "dof": dof,
    "p_value": p_value,
    "cramers_v": cramers_v,
    "low_expected_cells": low_expected,
    "total_cells": total_cells,
    "low_expected_percent": low_expected_percent,
    "conclusion": conclusion_global
}])

target_results_df = pd.DataFrame(target_results)

global_results.to_csv(
    "7_2_chi_square_global.csv",
    index=False,
    encoding="utf-8-sig"
)

target_results_df.to_csv(
    "7_2_3_tests_cibles_fisher.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===================================================")
print("FICHIERS CRÉÉS")
print("===================================================")
print("- 7_2_table_condition_category.csv")
print("- 7_2_chi_square_global.csv")
print("- 7_2_3_tests_cibles_fisher.csv")

print("\nPhrase méthodologique :")
print(
    "Le chi-carré global est conservé comme test exploratoire, "
    "car plusieurs cellules attendues sont faibles. "
    "Les comparaisons ciblées catégorie vs non-catégorie permettent "
    "d’interpréter plus prudemment les différences entre la condition neutre "
    "et les conditions expérimentales."
)