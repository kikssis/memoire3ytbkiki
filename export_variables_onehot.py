import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


# ===============================
# 1. CHARGER LE DATASET
# ===============================

df = pd.read_csv("dataset_clean.csv")

print("Dataset chargé :", len(df), "lignes")


# ===============================
# 2. NETTOYAGE IDENTIQUE AU MODÈLE
# ===============================

data = df[df["next_category_clean"] != "Unknown"].copy()
data = data.dropna(subset=["next_category_clean"])

print("Lignes utilisées :", len(data))


# ===============================
# 3. VARIABLES UTILISÉES PAR LE MODÈLE
# ===============================

features = [
    "category_clean",
    "previous_category",
    "previous_2_category",
    "history_1",
    "history_2",
    "history_3",
    "dominant_last_3",
    "condition",
    "depth",
    "history_intensity",
    "history_video_count",
    "repeat_last_2",
    "repeat_last_3",
    "recent_diversity",
    "same_as_previous"
]

X = data[features]


# ===============================
# 4. PRÉPROCESSING IDENTIQUE AU MODÈLE
# ===============================

cat_features = [
    "category_clean",
    "previous_category",
    "previous_2_category",
    "history_1",
    "history_2",
    "history_3",
    "dominant_last_3",
    "condition"
]

num_features = [
    "depth",
    "history_intensity",
    "history_video_count",
    "repeat_last_2",
    "repeat_last_3",
    "recent_diversity",
    "same_as_previous"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_features),

        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), num_features)
    ]
)


# ===============================
# 5. CRÉER LES VARIABLES ONEHOT
# ===============================

preprocessor.fit(X)

feature_names = preprocessor.get_feature_names_out()


# ===============================
# 6. AFFICHER ET EXPORTER LES VARIABLES
# ===============================

variables_df = pd.DataFrame({
    "numero": range(1, len(feature_names) + 1),
    "variable_utilisee": feature_names
})

print("\nNombre total de variables utilisées après OneHot :")
print(len(feature_names))

print("\nListe des variables :")
print(variables_df.to_string(index=False))

variables_df.to_csv(
    "liste_variables_utilisees_onehot.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nFichier créé : liste_variables_utilisees_onehot.csv")