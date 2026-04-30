import pandas as pd
from scipy.stats import chi2_contingency

df = pd.read_csv("dataset_clean.csv")

print("\n==============================")
print("17 — TEST DU CHI-DEUX")
print("==============================")

# =========================
# 1. TABLE DE CONTINGENCE
# =========================

table = pd.crosstab(df["condition"], df["category"])

print("\nTable de contingence :")
print(table)


# =========================
# 2. TEST CHI-DEUX
# =========================

chi2, p, dof, expected = chi2_contingency(table)

print("\nRésultats du test :")
print(f"Chi2 = {chi2:.3f}")
print(f"p-value = {p:.6f}")
print(f"Degrés de liberté = {dof}")


# =========================
# 3. MATRICE ATTENDUE
# =========================

expected_df = pd.DataFrame(
    expected,
    index=table.index,
    columns=table.columns
)

print("\nFréquences attendues :")
print(expected_df.round(2))


# =========================
# 4. CONTRIBUTION (TRÈS IMPORTANT)
# =========================

contrib = (table - expected_df) ** 2 / expected_df

print("\nContribution au Chi2 :")
print(contrib.round(2))


# =========================
# 5. EFFET (Cramér’s V)
# =========================

n = table.values.sum()
min_dim = min(table.shape) - 1

cramers_v = (chi2 / (n * min_dim)) ** 0.5

print(f"\nCramér's V = {cramers_v:.3f}")


# =========================
# 6. INTERPRÉTATION AUTO
# =========================

print("\nInterprétation :")

if p < 0.001:
    print("Effet TRÈS significatif (p < 0.001)")
elif p < 0.01:
    print("Effet significatif (p < 0.01)")
elif p < 0.05:
    print("Effet modéré (p < 0.05)")
else:
    print("Pas d'effet significatif")

if cramers_v < 0.1:
    print("Effet très faible")
elif cramers_v < 0.3:
    print("Effet faible à modéré")
elif cramers_v < 0.5:
    print("Effet modéré")
else:
    print("Effet fort")


# =========================
# 7. SAUVEGARDE
# =========================

expected_df.to_csv("chi2_expected.csv")
contrib.to_csv("chi2_contributions.csv")

print("\nFichiers créés :")
print("- chi2_expected.csv")
print("- chi2_contributions.csv")

