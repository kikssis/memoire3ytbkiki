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

df = df.sort_values(by=["session_id", "depth"]).reset_index(drop=True)

data = df.groupby("session_id").first().reset_index()

data = data[data["category_clean"] != "Unknown"].copy()

# Supprimer classes trop rares
counts = data["category_clean"].value_counts()
valid_classes = counts[counts >= 2].index
data = data[data["category_clean"].isin(valid_classes)].copy()

print("Premières vidéos utilisées :", len(data))

print("\nRépartition des catégories :")
print(data["category_clean"].value_counts())


# ===============================
# 3. FEATURES (AMÉLIORÉES MAIS SAFE)
# ===============================

features = [
    "condition",
    "history_category",
    "history_intensity",
    "history_video_count",

    # ajout utiles et légitimes
    "target_category",
    "is_current_target_category"
]

X = data[features].reset_index(drop=True)
y_raw = data["category_clean"].reset_index(drop=True)


# ===============================
# 4. ENCODAGE TARGET
# ===============================

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

print("\nClasses prédites :")
print(list(label_encoder.classes_))


# ===============================
# 5. SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ===============================
# 6. PREPROCESSING
# ===============================

cat_features = [
    "condition",
    "history_category",
    "target_category"
]

num_features = [
    "history_intensity",
    "history_video_count",
    "is_current_target_category"
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
# 7. MODÈLE
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
# 8. TRAIN
# ===============================

model.fit(X_train, y_train)


# ===============================
# 9. PRÉDICTIONS
# ===============================

pred = model.predict(X_test)

y_test_labels = label_encoder.inverse_transform(y_test)
pred_labels = label_encoder.inverse_transform(pred)


# ===============================
# 10. RÉSULTATS
# ===============================

print("\n==============================")
print("MODÈLE PREMIÈRE VIDÉO (OPTIMISÉ)")
print("==============================")

print("\nAccuracy :")
print(round(accuracy_score(y_test_labels, pred_labels), 3))

print("\nClassification report :")
print(classification_report(
    y_test_labels,
    pred_labels,
    zero_division=0
))

print("\nConfusion matrix :")
print(confusion_matrix(y_test_labels, pred_labels))


# ===============================
# 11. BASELINE
# ===============================

most_common_encoded = pd.Series(y_train).mode()[0]
most_common_label = label_encoder.inverse_transform([most_common_encoded])[0]

baseline = [most_common_label] * len(y_test_labels)

print("\n==============================")
print("BASELINE")
print("==============================")

print("Classe prédite :", most_common_label)
print("Accuracy :", round(
    accuracy_score(y_test_labels, baseline), 3
))


# ===============================
# 12. SAVE
# ===============================

joblib.dump(model, "xgboost_first_video.pkl")
joblib.dump(label_encoder, "label_encoder_first_video.pkl")

print("\nModèle sauvegardé")