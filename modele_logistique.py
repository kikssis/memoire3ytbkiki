import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# ===============================
# LOAD DATA
# ===============================

df = pd.read_csv("dataset_clean.csv")

df = df[df["next_category_clean"] != "Unknown"].copy()


# ===============================
# FEATURES
# ===============================

features = [
   "previous_category",
   "history_1",
   "history_2",
   "history_3",
   "condition",
   "depth"
]

X = df[features]
y = df["next_category_clean"]


# ===============================
# SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
   X, y,
   test_size=0.2,
   random_state=42
)


# ===============================
# PREPROCESSING
# ===============================

cat_features = features

preprocessor = ColumnTransformer(
   transformers=[
       ("cat", Pipeline([
           ("imputer", SimpleImputer(strategy="most_frequent")),
           ("onehot", OneHotEncoder(handle_unknown="ignore"))
       ]), cat_features)
   ]
)


# ===============================
# MODELE LOGISTIQUE
# ===============================

model = Pipeline([
   ("preprocessor", preprocessor),
   ("classifier", LogisticRegression(max_iter=1000))
])


# ===============================
# TRAIN
# ===============================

model.fit(X_train, y_train)


# ===============================
# RESULTATS
# ===============================

pred = model.predict(X_test)

print("\n==============================")
print("RÉGRESSION LOGISTIQUE")
print("==============================")

print(classification_report(y_test, pred))
