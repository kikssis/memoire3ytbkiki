import glob
import os
import pandas as pd


# =========================
# 1. CHARGER TOUS LES CSV BRUTS
# =========================

files = (
    glob.glob(r"condition1_neutre_*.csv")
    + glob.glob(r"collecte_sport_history_*.csv")
    + glob.glob(r"collecte_news_history_*.csv")
    + glob.glob(r"collecte_science_history_*.csv")
    + glob.glob(r"collecte_gaming_history_*.csv")
    + glob.glob(r"collecte_music_history_*.csv")
    + glob.glob(r"collecte_people_blog_history_*.csv")
)

files = [
    f for f in files
    if not f.startswith("dataset_")
    and not f.startswith("analyse_")
    and not f.startswith("Analyse_")
]

files = sorted(list(set(files)))

print("Fichiers utilisés :")
for f in files:
    print("-", f)

if len(files) == 0:
    print("\nAucun fichier CSV brut trouvé.")
    raise SystemExit


# =========================
# 2. LECTURE + FUSION
# =========================

dfs = []

for file in files:
    try:
        df_temp = pd.read_csv(file)
        df_temp["source_file"] = os.path.basename(file)
        dfs.append(df_temp)
    except Exception as e:
        print("Erreur lecture :", file, e)

if len(dfs) == 0:
    print("\nAucun CSV lisible.")
    raise SystemExit

df = pd.concat(dfs, ignore_index=True)

print("\nFusion OK")
print("Nombre de lignes brutes :", len(df))
print("Colonnes trouvées :")
print(df.columns.tolist())


# =========================
# 3. COLONNES OBLIGATOIRES
# =========================

required_columns = {
    "session_id": "unknown_session",
    "timestamp": "",
    "condition": "neutral",
    "history_category": "neutral",
    "history_intensity": 0,
    "history_video_count": 0,
    "video_id": "",
    "url": "",
    "category": "Unknown",
    "depth": 0,
    "total_seen_depth": 0,
    "watch_time": 0,
    "is_sponsored": 0,
    "include_in_prediction": 1,
    "sponsored_skipped_before": 0,
}

for col, default_value in required_columns.items():
    if col not in df.columns:
        df[col] = default_value
    else:
        df[col] = df[col].fillna(default_value)


# =========================
# 4. NETTOYAGE TYPES
# =========================

text_cols = [
    "session_id",
    "timestamp",
    "condition",
    "history_category",
    "video_id",
    "url",
    "category",
]

for col in text_cols:
    df[col] = df[col].astype(str)

numeric_cols = [
    "history_intensity",
    "history_video_count",
    "depth",
    "total_seen_depth",
    "watch_time",
    "is_sponsored",
    "include_in_prediction",
    "sponsored_skipped_before",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)


# =========================
# 5. CORRIGER CONDITION / HISTORIQUE
# =========================

df.loc[df["condition"].isin(["", "nan", "None"]), "condition"] = "neutral"

df.loc[df["condition"].str.contains("neutral", case=False, na=False), "condition"] = "neutral"

df.loc[df["condition"] == "neutral", "history_category"] = "neutral"
df.loc[df["condition"] == "neutral", "history_intensity"] = 0
df.loc[df["condition"] == "neutral", "history_video_count"] = 0

df.loc[df["condition"].str.contains("sport", case=False, na=False), "history_category"] = "sport"
df.loc[df["condition"].str.contains("music", case=False, na=False), "history_category"] = "music"
df.loc[df["condition"].str.contains("gaming", case=False, na=False), "history_category"] = "gaming"
df.loc[df["condition"].str.contains("news", case=False, na=False), "history_category"] = "news"
df.loc[df["condition"].str.contains("science", case=False, na=False), "history_category"] = "science"
df.loc[df["condition"].str.contains("people_blog", case=False, na=False), "history_category"] = "people_blog"

df.loc[df["condition"].str.endswith("_1", na=False), "history_intensity"] = 1
df.loc[df["condition"].str.endswith("_3", na=False), "history_intensity"] = 3
df.loc[df["condition"].str.endswith("_8", na=False), "history_intensity"] = 8

df.loc[df["history_video_count"] == 0, "history_video_count"] = df["history_intensity"]


# =========================
# 6. NETTOYAGE CATÉGORIE
# =========================

df["category"] = (
    df["category"]
    .replace("", "Unknown")
    .replace("nan", "Unknown")
    .replace("None", "Unknown")
    .str.strip()
)

df["category"] = df["category"].replace({
    "People &amp; Blogs": "People & Blogs",
    "News &amp; Politics": "News & Politics",
    "Science &amp; Technology": "Science & Technology",
})


# =========================
# 7. TRI IMPORTANT
# =========================

df = df.sort_values(
    by=["session_id", "depth", "timestamp"],
    ascending=[True, True, True]
).reset_index(drop=True)


# =========================
# 8. RECONSTRUIRE HISTORIQUE
# =========================

df["previous_url"] = df.groupby("session_id")["url"].shift(1)
df["previous_2_url"] = df.groupby("session_id")["url"].shift(2)

df["previous_category"] = df.groupby("session_id")["category"].shift(1)
df["previous_2_category"] = df.groupby("session_id")["category"].shift(2)

df["history_1"] = df.groupby("session_id")["category"].shift(1)
df["history_2"] = df.groupby("session_id")["category"].shift(2)
df["history_3"] = df.groupby("session_id")["category"].shift(3)

history_cols = [
    "previous_url",
    "previous_2_url",
    "previous_category",
    "previous_2_category",
    "history_1",
    "history_2",
    "history_3",
]

df[history_cols] = df[history_cols].fillna("Unknown")


# =========================
# 9. VARIABLE CIBLE
# =========================

df["next_category"] = df.groupby("session_id")["category"].shift(-1)
df["next_url"] = df.groupby("session_id")["url"].shift(-1)

df["next_category"] = df["next_category"].fillna("Unknown")
df["next_url"] = df["next_url"].fillna("Unknown")


# =========================
# 10. REGROUPEMENT CATÉGORIES RARES
# =========================

threshold = 10

counts = df["category"].value_counts()
rare_categories = counts[counts < threshold].index

df["category_clean"] = df["category"].replace(rare_categories, "Other")

df["next_category_clean"] = df.groupby("session_id")["category_clean"].shift(-1)
df["next_category_clean"] = df["next_category_clean"].fillna("Unknown")


# =========================
# 11. TARGET CATEGORY
# =========================

def get_target_category(condition):
    condition = str(condition).lower()

    if "sport" in condition:
        return "Sports"
    if "music" in condition:
        return "Music"
    if "gaming" in condition:
        return "Gaming"
    if "news" in condition:
        return "News & Politics"
    if "science" in condition:
        return "Science & Technology"
    if "people_blog" in condition:
        return "People & Blogs"
    return "neutral"


df["target_category"] = df["condition"].apply(get_target_category)


# =========================
# 12. FEATURES POUR MODÈLE
# =========================

df["repeat_last_2"] = (
    df["history_1"] == df["history_2"]
).astype(int)

df["repeat_last_3"] = (
    (df["history_1"] == df["history_2"])
    & (df["history_2"] == df["history_3"])
).astype(int)

df["recent_diversity"] = df[
    ["history_1", "history_2", "history_3"]
].nunique(axis=1)

df["same_as_previous"] = (
    df["category"] == df["previous_category"]
).astype(int)

df["dominant_last_3"] = df[
    ["history_1", "history_2", "history_3"]
].mode(axis=1)[0]

df["target_match_score"] = (
    (df["history_1"] == df["target_category"]).astype(int)
    + (df["history_2"] == df["target_category"]).astype(int)
    + (df["history_3"] == df["target_category"]).astype(int)
)


# =========================
# 13. LIEN AVEC CONDITION
# =========================

df["is_current_target_category"] = (
    df["category"].str.lower() == df["target_category"].str.lower()
).astype(int)

df["is_previous_target_category"] = (
    df["previous_category"].str.lower() == df["target_category"].str.lower()
).astype(int)

df["is_next_target_category"] = (
    df["next_category"].str.lower() == df["target_category"].str.lower()
).astype(int)


# =========================
# 14. FILTRAGE FINAL
# =========================

df = df[df["include_in_prediction"] == 1].copy()
df = df[df["is_sponsored"] == 0].copy()

df = df.sort_values(
    by=["session_id", "depth", "timestamp"],
    ascending=[True, True, True]
).reset_index(drop=True)


# =========================
# 15. ORDRE FINAL DES COLONNES
# =========================

final_columns = [
    "session_id",
    "timestamp",
    "condition",
    "history_category",
    "history_intensity",
    "history_video_count",

    "video_id",
    "url",
    "previous_url",
    "previous_2_url",
    "next_url",

    "category",
    "previous_category",
    "previous_2_category",
    "next_category",

    "category_clean",
    "next_category_clean",

    "history_1",
    "history_2",
    "history_3",
    "dominant_last_3",

    "depth",
    "total_seen_depth",
    "watch_time",
    "is_sponsored",
    "include_in_prediction",
    "sponsored_skipped_before",

    "repeat_last_2",
    "repeat_last_3",
    "recent_diversity",
    "same_as_previous",
    "target_match_score",

    "target_category",
    "is_current_target_category",
    "is_previous_target_category",
    "is_next_target_category",

    "source_file",
]

final_columns = [col for col in final_columns if col in df.columns]
df = df[final_columns]


# =========================
# 16. SAUVEGARDE
# =========================

output = "dataset_clean.csv"

df.to_csv(output, index=False, encoding="utf-8-sig")

print("\nDATASET FINAL CRÉÉ :", output)
print("Nombre de lignes finales :", len(df))


# =========================
# 17. CHECKS
# =========================

print("\nRépartition des conditions :")
print(df["condition"].value_counts(dropna=False))

print("\nRépartition history_category :")
print(df["history_category"].value_counts(dropna=False))

print("\nRépartition history_intensity :")
print(df["history_intensity"].value_counts(dropna=False))

print("\nRépartition category :")
print(df["category"].value_counts(dropna=False))

print("\nRépartition category_clean :")
print(df["category_clean"].value_counts(dropna=False))

print("\nNouvelles features :")
print(df[[
    "history_1",
    "history_2",
    "history_3",
    "dominant_last_3",
    "target_category",
    "target_match_score"
]].head(10))

print("\nColonnes finales :")
print(df.columns.tolist())

print("\nAperçu :")
print(df.head(10))