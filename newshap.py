# ============================================================
# ANALYSE SHAP - MODÈLE XGBOOST 8.2
# Version cohérente avec le modèle XGBoost final propre
# ============================================================

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier


# ============================================================
# 1. CHARGER LE DATASET
# ============================================================

df = pd.read_csv("dataset_clean.csv")

print("\nDataset chargé :", len(df), "lignes")


# ============================================================
# 2. NETTOYAGE POUR PRÉDICTION
# ============================================================

# On prédit next_category_clean.
# On enlève les lignes où la prochaine catégorie est inconnue.
data = df[df["next_category_clean"] != "Unknown"].copy()
data = data.dropna(subset=["next_category_clean"])

print("Lignes utilisées pour le modèle :", len(data))


# ============================================================
# 3. VARIABLES EXPLICATIVES
# ============================================================
# On ne met PAS dans les variables explicatives :
# - next_category
# - next_category_clean
# - next_url
# - is_next_target_category
# - target_category
# - target_match_score
#
# Ces variables sont trop proches de la réponse à prédire
# ou directement liées à la catégorie suivante.

features_cat = [
    "category_clean",
    "previous_category",
    "previous_2_category",
    "history_1",
    "history_2",
    "history_3",
    "dominant_last_3",
    "condition"
]

features_num = [
    "depth",
    "history_intensity",
    "history_video_count",
    "repeat_last_2",
    "repeat_last_3",
    "recent_diversity",
    "same_as_previous"
]

features = features_cat + features_num

# Sécuriser les colonnes catégorielles
for col in features_cat:
    if col not in data.columns:
        data[col] = "Unknown"
    data[col] = data[col].fillna("Unknown")

# Sécuriser les colonnes numériques
for col in features_num:
    if col not in data.columns:
        data[col] = 0
    data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

# Sécuriser session_id
if "session_id" not in data.columns:
    data["session_id"] = range(len(data))

X = data[features]
y_raw = data["next_category_clean"]
groups = data["session_id"]

print("\nClasses prédites avant encodage :")
print(sorted(y_raw.unique()))
print("Nombre de classes :", y_raw.nunique())


# ============================================================
# 4. ENCODAGE DE LA CIBLE
# ============================================================

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

classes = label_encoder.classes_

print("\nClasses encodées :")
for i, c in enumerate(classes):
    print(i, "=", c)


# ============================================================
# 5. SPLIT PAR SESSION
# ============================================================

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(gss.split(X, y, groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

y_train_labels = label_encoder.inverse_transform(y_train)
y_test_labels = label_encoder.inverse_transform(y_test)

print("\nTaille train :", len(X_train))
print("Taille test :", len(X_test))

print("\nDistribution train :")
print(pd.Series(y_train_labels).value_counts(normalize=True).round(3))

print("\nDistribution test :")
print(pd.Series(y_test_labels).value_counts(normalize=True).round(3))


# ============================================================
# 6. PRÉPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]),
            features_cat
        ),
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            features_num
        )
    ]
)


# ============================================================
# 7. TRANSFORMATION DES DONNÉES
# ============================================================

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

X_train_transformed = pd.DataFrame(
    X_train_transformed,
    columns=feature_names
)

X_test_transformed = pd.DataFrame(
    X_test_transformed,
    columns=feature_names
)

print("\nNombre de features après OneHot :")
print(len(feature_names))


# ============================================================
# 8. MODÈLE XGBOOST IDENTIQUE À 8.2
# ============================================================

model = XGBClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42
)

model.fit(X_train_transformed, y_train)

pred = model.predict(X_test_transformed)
proba = model.predict_proba(X_test_transformed)

pred_labels = label_encoder.inverse_transform(pred)


# ============================================================
# 9. ÉVALUATION DU MODÈLE
# ============================================================

print("\n==============================")
print("MODÈLE XGBOOST - VERSION SHAP COHÉRENTE 8.2")
print("==============================")

accuracy = accuracy_score(y_test_labels, pred_labels)

print("\nAccuracy :")
print(round(accuracy, 3))

print("\nClassification report :")
print(classification_report(
    y_test_labels,
    pred_labels,
    zero_division=0
))

print("\nConfusion matrix :")
print(confusion_matrix(y_test_labels, pred_labels))


# ============================================================
# 10. ANALYSE SHAP
# ============================================================

print("\n==============================")
print("ANALYSE SHAP")
print("==============================")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_transformed)

print("SHAP values calculées.")


# ============================================================
# 11. SUMMARY PLOT GLOBAL
# ============================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_test_transformed,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()
plt.savefig("shap_summary_global_xgboost_8_2.png", dpi=300, bbox_inches="tight")
plt.close()

print("Graphique créé : shap_summary_global_xgboost_8_2.png")


# ============================================================
# 12. IMPORTANCE GLOBALE DES VARIABLES SHAP
# ============================================================
# Pour un modèle multiclasses, SHAP peut retourner :
# - une liste de tableaux : un tableau par classe
# - ou un tableau 3D : observations x features x classes
#
# On calcule ici la moyenne des valeurs absolues SHAP
# sur toutes les observations et toutes les classes.

if isinstance(shap_values, list):
    shap_array = np.array(shap_values)
    mean_abs_shap = np.mean(np.abs(shap_array), axis=(0, 1))
else:
    if len(shap_values.shape) == 3:
        mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 2))
    else:
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

shap_importance = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap
}).sort_values(
    by="mean_abs_shap",
    ascending=False
)

shap_importance.to_csv(
    "shap_importance_globale_xgboost_8_2.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nTop 30 variables SHAP globales :")
print(shap_importance.head(30).to_string(index=False))

print("\nFichier créé : shap_importance_globale_xgboost_8_2.csv")


# ============================================================
# 13. SHAP PAR CATÉGORIE
# ============================================================

for class_index, class_name in enumerate(classes):

    print("\n==============================")
    print("Classe :", class_name)
    print("==============================")

    if isinstance(shap_values, list):
        shap_class = shap_values[class_index]
    else:
        if len(shap_values.shape) == 3:
            shap_class = shap_values[:, :, class_index]
        else:
            shap_class = shap_values

    mean_abs_class = np.mean(np.abs(shap_class), axis=0)

    shap_class_importance = pd.DataFrame({
        "category_predicted": class_name,
        "feature": feature_names,
        "mean_abs_shap": mean_abs_class
    }).sort_values(by="mean_abs_shap", ascending=False)

    safe_class_name = (
        class_name
        .replace(" ", "_")
        .replace("&", "and")
        .replace("/", "_")
    )

    output_csv = f"shap_importance_{safe_class_name}_xgboost_8_2.csv"

    shap_class_importance.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(shap_class_importance.head(15).to_string(index=False))
    print("Fichier créé :", output_csv)


# ============================================================
# 14. GRAPHIQUES SHAP PAR CLASSE IMPORTANTE
# ============================================================

classes_to_plot = [
    "People & Blogs",
    "Sports",
    "Gaming",
    "Entertainment",
    "News & Politics",
    "Science & Technology"
]

for class_name in classes_to_plot:
    if class_name not in classes:
        continue

    class_index = list(classes).index(class_name)

    if isinstance(shap_values, list):
        shap_class = shap_values[class_index]
    else:
        if len(shap_values.shape) == 3:
            shap_class = shap_values[:, :, class_index]
        else:
            shap_class = shap_values

    safe_class_name = (
        class_name
        .replace(" ", "_")
        .replace("&", "and")
        .replace("/", "_")
    )

    plt.figure()

    shap.summary_plot(
        shap_class,
        X_test_transformed,
        feature_names=feature_names,
        show=False
    )

    plt.tight_layout()
    plt.savefig(
        f"shap_summary_{safe_class_name}_xgboost_8_2.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("Graphique créé :", f"shap_summary_{safe_class_name}_xgboost_8_2.png")


# ============================================================
# 15. SAUVEGARDE DES PRÉDICTIONS
# ============================================================

predictions_df = pd.DataFrame({
    "true_category": y_test_labels,
    "predicted_category": pred_labels
})

predictions_df.to_csv(
    "shap_xgboost_8_2_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 16. FIN
# ============================================================

print("\n==============================")
print("ANALYSE SHAP TERMINÉE")
print("==============================")

print("\nFichiers principaux créés :")
print("- shap_summary_global_xgboost_8_2.png")
print("- shap_importance_globale_xgboost_8_2.csv")
print("- shap_importance_[classe]_xgboost_8_2.csv")
print("- shap_summary_[classe]_xgboost_8_2.png")
print("- shap_xgboost_8_2_predictions.csv")