# ============================================================
# ANALYSE SHAP - MODÈLE XGBOOST LÉGER
# Version mémoire optimisée
# ============================================================

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import gc

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score
)

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
# On ne met PAS :
# - next_category
# - next_category_clean
# - next_url
# - is_next_target_category
# - target_category
# - target_match_score
#
# Ces variables sont trop proches de la réponse à prédire.

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


# ============================================================
# 4. SÉCURISER LES COLONNES
# ============================================================

for col in features_cat:
    if col not in data.columns:
        data[col] = "Unknown"
    data[col] = data[col].fillna("Unknown")

for col in features_num:
    if col not in data.columns:
        data[col] = 0
    data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

if "session_id" not in data.columns:
    data["session_id"] = range(len(data))


X = data[features]
y_raw = data["next_category_clean"]
groups = data["session_id"]

print("\nClasses prédites avant encodage :")
print(sorted(y_raw.unique()))
print("Nombre de classes :", y_raw.nunique())


# ============================================================
# 5. ENCODAGE DE LA CIBLE
# ============================================================

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

classes = label_encoder.classes_

print("\nClasses encodées :")
for i, c in enumerate(classes):
    print(i, "=", c)


# ============================================================
# 6. SPLIT PAR SESSION
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
# 7. PRÉPROCESSING
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
# 8. TRANSFORMATION DES DONNÉES
# ============================================================

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

X_train_transformed = pd.DataFrame(
    X_train_transformed,
    columns=feature_names
).astype("float32")

X_test_transformed = pd.DataFrame(
    X_test_transformed,
    columns=feature_names
).astype("float32")

print("\nNombre de features après OneHot :")
print(len(feature_names))


# ============================================================
# 9. MODÈLE XGBOOST LÉGER POUR SHAP
# ============================================================
# Même cible et mêmes variables que le modèle principal,
# mais modèle plus léger pour éviter les problèmes mémoire.

model = XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
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
# 10. ÉVALUATION DU MODÈLE
# ============================================================

print("\n==============================")
print("MODÈLE XGBOOST LÉGER - VERSION SHAP")
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
# 11. TOP-K ACCURACY
# ============================================================

top3 = top_k_accuracy_score(
    y_test,
    proba,
    k=3,
    labels=list(range(len(label_encoder.classes_)))
)

top5 = top_k_accuracy_score(
    y_test,
    proba,
    k=5,
    labels=list(range(len(label_encoder.classes_)))
)

print("\n==============================")
print("TOP-K ACCURACY")
print("==============================")
print("Top-3 accuracy :", round(top3, 3))
print("Top-5 accuracy :", round(top5, 3))


# ============================================================
# 12. BASELINES
# ============================================================

print("\n==============================")
print("BASELINES")
print("==============================")

train_labels_series = pd.Series(y_train_labels)

# Baseline Top-1 : toujours prédire la classe la plus fréquente
most_common_label = train_labels_series.mode()[0]
baseline_common = [most_common_label] * len(y_test_labels)
baseline_common_acc = accuracy_score(y_test_labels, baseline_common)

print("\nBaseline Top-1 most common :")
print("Classe prédite :", most_common_label)
print("Accuracy :", round(baseline_common_acc, 3))

# Baseline Top-3 et Top-5
top3_baseline_classes = train_labels_series.value_counts().head(3).index.tolist()
top5_baseline_classes = train_labels_series.value_counts().head(5).index.tolist()

baseline_top3_acc = sum(
    true_label in top3_baseline_classes
    for true_label in y_test_labels
) / len(y_test_labels)

baseline_top5_acc = sum(
    true_label in top5_baseline_classes
    for true_label in y_test_labels
) / len(y_test_labels)

print("\nBaseline Top-3 :")
print("Classes proposées :", top3_baseline_classes)
print("Accuracy :", round(baseline_top3_acc, 3))

print("\nBaseline Top-5 :")
print("Classes proposées :", top5_baseline_classes)
print("Accuracy :", round(baseline_top5_acc, 3))

print("\nGain Top-1 du modèle :", round(accuracy - baseline_common_acc, 3))
print("Gain Top-3 du modèle :", round(top3 - baseline_top3_acc, 3))
print("Gain Top-5 du modèle :", round(top5 - baseline_top5_acc, 3))


# ============================================================
# 13. ANALYSE SHAP SUR ÉCHANTILLON
# ============================================================

print("\n==============================")
print("ANALYSE SHAP")
print("==============================")

sample_size = min(200, len(X_test_transformed))

X_shap = X_test_transformed.sample(
    n=sample_size,
    random_state=42
).astype("float32")

print("Nombre de lignes utilisées pour SHAP :", len(X_shap))

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)

print("SHAP values calculées.")


# ============================================================
# 14. SUMMARY PLOT GLOBAL
# ============================================================

try:
    plt.figure(figsize=(10, 7))

    shap.summary_plot(
        shap_values,
        X_shap,
        feature_names=feature_names,
        show=False,
        max_display=25
    )

    plt.tight_layout()
    plt.savefig(
        "shap_summary_global_xgboost_light.png",
        dpi=120,
        bbox_inches="tight"
    )
    plt.close("all")
    gc.collect()

    print("Graphique créé : shap_summary_global_xgboost_light.png")

except MemoryError:
    print("Mémoire insuffisante pour créer le graphique SHAP global.")
    plt.close("all")
    gc.collect()


# ============================================================
# 15. IMPORTANCE GLOBALE DES VARIABLES SHAP
# ============================================================

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
    "shap_importance_globale_xgboost_light.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nTop 30 variables SHAP globales :")
print(shap_importance.head(30).to_string(index=False))

print("\nFichier créé : shap_importance_globale_xgboost_light.csv")


# ============================================================
# 16. SHAP PAR CATÉGORIE
# ============================================================
# On crée les CSV pour toutes les classes.

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

    output_csv = f"shap_importance_{safe_class_name}_xgboost_light.csv"

    shap_class_importance.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(shap_class_importance.head(15).to_string(index=False))
    print("Fichier créé :", output_csv)


# ============================================================
# 17. GRAPHIQUES SHAP PAR CLASSE PRINCIPALE
# ============================================================
# Pour éviter les erreurs mémoire, on limite les graphiques
# aux catégories principales uniquement.

classes_to_plot = [
    "People & Blogs",
    "Sports",
    "Gaming"
]

for class_name in classes_to_plot:

    if class_name not in classes:
        continue

    print("\nCréation graphique SHAP pour :", class_name)

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

    try:
        plt.figure(figsize=(10, 7))

        shap.summary_plot(
            shap_class,
            X_shap,
            feature_names=feature_names,
            show=False,
            max_display=20
        )

        plt.tight_layout()
        plt.savefig(
            f"shap_summary_{safe_class_name}_xgboost_light.png",
            dpi=120,
            bbox_inches="tight"
        )
        plt.close("all")
        gc.collect()

        print("Graphique créé :", f"shap_summary_{safe_class_name}_xgboost_light.png")

    except MemoryError:
        print("Mémoire insuffisante pour créer le graphique :", class_name)
        plt.close("all")
        gc.collect()


# ============================================================
# 18. SAUVEGARDE DES PRÉDICTIONS ET MÉTRIQUES
# ============================================================

predictions_df = pd.DataFrame({
    "true_category": y_test_labels,
    "predicted_category": pred_labels
})

predictions_df.to_csv(
    "shap_xgboost_light_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

metrics_df = pd.DataFrame({
    "metric": [
        "accuracy",
        "baseline_top1_most_common",
        "top3_accuracy",
        "baseline_top3",
        "top5_accuracy",
        "baseline_top5",
        "gain_top1",
        "gain_top3",
        "gain_top5"
    ],
    "value": [
        accuracy,
        baseline_common_acc,
        top3,
        baseline_top3_acc,
        top5,
        baseline_top5_acc,
        accuracy - baseline_common_acc,
        top3 - baseline_top3_acc,
        top5 - baseline_top5_acc
    ]
})

metrics_df.to_csv(
    "shap_xgboost_light_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 19. FIN
# ============================================================

print("\n==============================")
print("ANALYSE SHAP TERMINÉE")
print("==============================")

print("\nFichiers principaux créés :")
print("- shap_summary_global_xgboost_light.png")
print("- shap_importance_globale_xgboost_light.csv")
print("- shap_importance_[classe]_xgboost_light.csv")
print("- shap_summary_People_and_Blogs_xgboost_light.png")
print("- shap_summary_Sports_xgboost_light.png")
print("- shap_summary_Gaming_xgboost_light.png")
print("- shap_xgboost_light_predictions.csv")
print("- shap_xgboost_light_metrics.csv")