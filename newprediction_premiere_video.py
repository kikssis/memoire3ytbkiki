import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier


# ===============================
# 1. CHARGER DATASET
# ===============================

df = pd.read_csv("dataset_clean.csv")

print("\nDataset chargé :", len(df))


# ===============================
# 2. PREMIÈRE VIDÉO PAR SESSION
# ===============================

# Trier les vidéos par session et par profondeur
df = df.sort_values(by=["session_id", "depth"]).reset_index(drop=True)

# Garder la première vidéo de chaque session
data = df.groupby("session_id").first().reset_index()

# Enlever les catégories inconnues
data = data[data["category_clean"] != "Unknown"].copy()

print("\nPremières vidéos avant filtre classes rares :", len(data))

print("\nRépartition initiale des catégories :")
print(data["category_clean"].value_counts())


# ===============================
# 3. SUPPRESSION DES CLASSES TROP RARES
# ===============================

# On garde seulement les catégories avec au moins 5 observations
# Cela permet d'utiliser stratify=y plus proprement
counts = data["category_clean"].value_counts()
valid_classes = counts[counts >= 5].index

data = data[data["category_clean"].isin(valid_classes)].copy()

print("\nPremières vidéos utilisées après filtre classes rares :", len(data))

print("\nRépartition finale des catégories :")
print(data["category_clean"].value_counts())

print("\nDistribution finale des catégories en proportion :")
print(data["category_clean"].value_counts(normalize=True).round(3))


# ===============================
# 4. BASELINE GLOBALE DESCRIPTIVE
# ===============================

global_majority_class = data["category_clean"].value_counts().idxmax()
global_baseline = data["category_clean"].value_counts(normalize=True).max()

print("\n==============================")
print("BASELINE GLOBALE DESCRIPTIVE")
print("==============================")
print("Classe majoritaire globale :", global_majority_class)
print("Baseline globale :", round(global_baseline, 3))


# ===============================
# 5. FEATURES SANS FUITE D'INFORMATION
# ===============================

# Attention :
# is_current_target_category est supprimée car elle dépend de la catégorie actuelle,
# donc elle donne indirectement la réponse au modèle.

features = [
    "condition",
    "history_category",
    "history_intensity",
    "history_video_count",
    "target_category"
]

X = data[features].reset_index(drop=True)
y_raw = data["category_clean"].reset_index(drop=True)


# ===============================
# 6. ENCODAGE TARGET
# ===============================

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

print("\nClasses prédites :")
print(list(label_encoder.classes_))


# ===============================
# 7. SPLIT TRAIN / TEST STRATIFIÉ
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

y_train_labels = label_encoder.inverse_transform(y_train)
y_test_labels = label_encoder.inverse_transform(y_test)

print("\nTaille train :", len(X_train))
print("Taille test :", len(X_test))

print("\nDistribution train :")
print(pd.Series(y_train_labels).value_counts(normalize=True).round(3))

print("\nDistribution test :")
print(pd.Series(y_test_labels).value_counts(normalize=True).round(3))


# ===============================
# 8. PREPROCESSING
# ===============================

cat_features = [
    "condition",
    "history_category",
    "target_category"
]

num_features = [
    "history_intensity",
    "history_video_count"
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
# 9. MODÈLE XGBOOST
# ===============================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42
    ))
])


# ===============================
# 10. ENTRAÎNEMENT
# ===============================

model.fit(X_train, y_train)


# ===============================
# 11. PRÉDICTIONS
# ===============================

pred = model.predict(X_test)

pred_labels = label_encoder.inverse_transform(pred)


# ===============================
# 12. RÉSULTATS DU MODÈLE
# ===============================

print("\n==============================")
print("MODÈLE PREMIÈRE VIDÉO - XGBOOST")
print("==============================")

model_accuracy = accuracy_score(y_test_labels, pred_labels)

print("\nAccuracy du modèle :")
print(round(model_accuracy, 3))

print("\nClassification report :")
print(classification_report(
    y_test_labels,
    pred_labels,
    zero_division=0
))

print("\nConfusion matrix :")
print(confusion_matrix(y_test_labels, pred_labels))


# ===============================
# 13. BASELINE SUR LE JEU DE TEST
# ===============================

# Classe majoritaire apprise sur le train
most_common_train_encoded = pd.Series(y_train).mode()[0]
most_common_train_label = label_encoder.inverse_transform([most_common_train_encoded])[0]

# Baseline : prédire toujours cette classe sur le test
baseline_pred = [most_common_train_label] * len(y_test_labels)

baseline_accuracy = accuracy_score(y_test_labels, baseline_pred)

print("\n==============================")
print("BASELINE TEST")
print("==============================")
print("Classe prédite par la baseline :", most_common_train_label)
print("Accuracy baseline test :", round(baseline_accuracy, 3))

print("\nGain du modèle par rapport à la baseline test :")
print(round(model_accuracy - baseline_accuracy, 3))


# ===============================
# 14. BASELINE SI ON PRÉDIT PEOPLE & BLOGS
# ===============================

people_baseline_pred = ["People & Blogs"] * len(y_test_labels)
people_baseline_accuracy = accuracy_score(y_test_labels, people_baseline_pred)

print("\n==============================")
print("BASELINE PEOPLE & BLOGS SUR TEST")
print("==============================")
print("Accuracy si on prédit toujours People & Blogs :",
      round(people_baseline_accuracy, 3))


# ===============================
# 15. SAUVEGARDE
# ===============================

joblib.dump(model, "xgboost_first_video_clean.pkl")
joblib.dump(label_encoder, "label_encoder_first_video_clean.pkl")

print("\nModèle sauvegardé :")
print("- xgboost_first_video_clean.pkl")
print("- label_encoder_first_video_clean.pkl")