import numpy as np
import pandas as pd
import joblib

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


# ===============================
# 1. CHARGER LE DATASET
# ===============================

df = pd.read_csv("dataset_clean.csv")

print("\nDataset chargé :", len(df), "lignes")


# ===============================
# 2. NETTOYAGE POUR PRÉDICTION
# ===============================

# On enlève les lignes où la prochaine catégorie est inconnue
data = df[df["next_category_clean"] != "Unknown"].copy()
data = data.dropna(subset=["next_category_clean"])

print("Lignes utilisées pour le modèle :", len(data))


# ===============================
# 3. VARIABLES EXPLICATIVES
# ===============================
# On prédit next_category_clean.
# On ne met PAS dans les features :
# - next_category
# - next_category_clean
# - next_url
# - is_next_target_category
# - target_category
# - target_match_score
# car ces variables sont trop proches de la réponse à prédire.

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
y_raw = data["next_category_clean"]
groups = data["session_id"]


# ===============================
# 4. ENCODAGE DE LA CIBLE
# ===============================

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

print("\nClasses prédites :")
print(list(label_encoder.classes_))


# ===============================
# 5. SPLIT PAR SESSION
# ===============================

gss = GroupShuffleSplit(
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


# ===============================
# 6. PRÉPROCESSING
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
# 7. MODÈLE XGBOOST
# ===============================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42
    ))
])


# ===============================
# 8. ENTRAÎNEMENT
# ===============================

model.fit(X_train, y_train)


# ===============================
# 9. PRÉDICTIONS
# ===============================

pred = model.predict(X_test)
proba = model.predict_proba(X_test)

pred_labels = label_encoder.inverse_transform(pred)


# ===============================
# 10. RÉSULTATS DU MODÈLE
# ===============================

print("\n==============================")
print("MODÈLE CATÉGORIE XGBOOST FINAL PROPRE")
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


# ===============================
# 11. TOP-K ACCURACY DU MODÈLE
# ===============================

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
print("TOP-K ACCURACY DU MODÈLE")
print("==============================")

print("Top-3 accuracy modèle :", round(top3, 3))
print("Top-5 accuracy modèle :", round(top5, 3))


# ===============================
# 12. TOP-3 DÉTAILLÉ PAR LIGNE
# ===============================

top3_indices = np.argsort(proba, axis=1)[:, -3:][:, ::-1]

top3_labels = [
    label_encoder.inverse_transform(row)
    for row in top3_indices
]

top3_results = []

for i in range(len(y_test_labels)):
    top3_results.append({
        "true_category": y_test_labels[i],
        "predicted_category": pred_labels[i],
        "top_1": top3_labels[i][0],
        "top_2": top3_labels[i][1],
        "top_3": top3_labels[i][2],
        "true_in_top3": y_test_labels[i] in top3_labels[i]
    })

top3_results_df = pd.DataFrame(top3_results)


# ===============================
# 13. BASELINES TOP-1
# ===============================

print("\n==============================")
print("BASELINES TOP-1")
print("==============================")

# Baseline 1 : toujours prédire la catégorie la plus fréquente du train
most_common_encoded = pd.Series(y_train).mode()[0]
most_common_label = label_encoder.inverse_transform([most_common_encoded])[0]

baseline_common = [most_common_label] * len(y_test_labels)
baseline_common_acc = accuracy_score(y_test_labels, baseline_common)

print("\nBaseline most common :")
print("Classe prédite :", most_common_label)
print("Accuracy :", round(baseline_common_acc, 3))


# Baseline 2 : prédire la catégorie précédente
baseline_previous = X_test["previous_category"].fillna("Unknown")

baseline_previous_acc = accuracy_score(
    y_test_labels,
    baseline_previous
)

print("\nBaseline previous category :")
print("Accuracy :", round(baseline_previous_acc, 3))


# Baseline 3 : prédire la catégorie actuelle
baseline_current = X_test["category_clean"].fillna("Unknown")

baseline_current_acc = accuracy_score(
    y_test_labels,
    baseline_current
)

print("\nBaseline current category :")
print("Accuracy :", round(baseline_current_acc, 3))


# ===============================
# 14. BASELINES TOP-3 ET TOP-5
# ===============================
# Ici, la baseline Top-K consiste à toujours proposer
# les K catégories les plus fréquentes dans le train set.
# Exemple Top-3 : toujours proposer les 3 catégories les plus fréquentes.
# Exemple Top-5 : toujours proposer les 5 catégories les plus fréquentes.

print("\n==============================")
print("BASELINES TOP-K")
print("==============================")

train_labels_series = pd.Series(y_train_labels)

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

print("\nGain Top-3 du modèle par rapport à la baseline Top-3 :")
print(round(top3 - baseline_top3_acc, 3))

print("\nGain Top-5 du modèle par rapport à la baseline Top-5 :")
print(round(top5 - baseline_top5_acc, 3))


# ===============================
# 15. IMPORTANCE DES VARIABLES
# ===============================

classifier = model.named_steps["classifier"]
preprocessor_fitted = model.named_steps["preprocessor"]

feature_names = preprocessor_fitted.get_feature_names_out()
importances = classifier.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(
    by="importance",
    ascending=False
)

print("\n==============================")
print("TOP 30 VARIABLES IMPORTANTES")
print("==============================")
print(importance_df.head(30).round(4))


# ===============================
# 16. FEATURES RÉELLEMENT UTILISÉES
# ===============================

print("\n==============================")
print("FEATURES DONNÉES AU MODÈLE")
print("==============================")

print(features)

print("\nNombre de features après OneHot :")
print(len(feature_names))


# ===============================
# 17. SAUVEGARDES
# ===============================

importance_df.to_csv(
    "xgboost_categorie_feature_importance_final_propre.csv",
    index=False,
    encoding="utf-8-sig"
)

results_df = pd.DataFrame({
    "true_category": y_test_labels,
    "predicted_category": pred_labels
})

results_df.to_csv(
    "xgboost_categorie_predictions_final_propre.csv",
    index=False,
    encoding="utf-8-sig"
)

top3_results_df.to_csv(
    "xgboost_categorie_top3_predictions_final_propre.csv",
    index=False,
    encoding="utf-8-sig"
)

# Résumé des métriques
metrics_df = pd.DataFrame({
    "metric": [
        "model_accuracy",
        "baseline_most_common",
        "baseline_previous_category",
        "baseline_current_category",
        "model_top3",
        "baseline_top3",
        "model_top5",
        "baseline_top5",
        "gain_top1_vs_most_common",
        "gain_top3",
        "gain_top5"
    ],
    "value": [
        accuracy,
        baseline_common_acc,
        baseline_previous_acc,
        baseline_current_acc,
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
    "xgboost_categorie_metrics_final_propre.csv",
    index=False,
    encoding="utf-8-sig"
)

joblib.dump(model, "xgboost_categorie_model_final_propre.pkl")
joblib.dump(label_encoder, "xgboost_categorie_label_encoder_final_propre.pkl")

print("\nFichiers créés :")
print("- xgboost_categorie_feature_importance_final_propre.csv")
print("- xgboost_categorie_predictions_final_propre.csv")
print("- xgboost_categorie_top3_predictions_final_propre.csv")
print("- xgboost_categorie_metrics_final_propre.csv")
print("- xgboost_categorie_model_final_propre.pkl")
print("- xgboost_categorie_label_encoder_final_propre.pkl")