import csv
import glob
import re
import time
from datetime import datetime
from urllib.request import Request, urlopen

import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains


# ============================
# PARAMÈTRES
# ============================

NB_SESSIONS = 15

NB_VALID_VIDEOS = 15
MAX_TOTAL_VIDEOS = 60

WATCH_TIME_COLLECT = 1.5
WATCH_TIME_SPONSORED = 2

WAIT_BEFORE_EXTRACTION = 1.5
WAIT_AFTER_NEXT = 2
PAUSE_BETWEEN_SESSIONS = 3

OUTPUT_PREFIX = "condition1_neutre"
FINAL_DATASET = "dataset_condition1.csv"


# ============================
# FONCTIONS UTILES
# ============================

def extract_video_id(url):
    if url and "/shorts/" in url:
        return url.split("/shorts/")[-1].split("?")[0]
    return None


def get_category(url):
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        html = urlopen(req, timeout=10).read().decode("utf-8", errors="ignore")

        match = re.search(r'"category":"([^"]+)"', html)

        if match:
            return match.group(1).replace("\\u0026", "&")

    except Exception as e:
        print("Erreur catégorie :", e)

    return None


def is_sponsored(driver):
    try:
        selectors = [
            '//span[normalize-space(text())="Sponsorisé"]',
            '//span[normalize-space(text())="Sponsored"]',
            '//div[normalize-space(text())="Sponsorisé"]',
            '//div[normalize-space(text())="Sponsored"]',
        ]

        for selector in selectors:
            elements = driver.find_elements(By.XPATH, selector)

            for element in elements:
                try:
                    if element.is_displayed():
                        return True
                except Exception:
                    continue

        return False

    except Exception:
        return False


# ============================
# COOKIES
# ============================

def accept_cookies(driver):
    time.sleep(3)

    selectors = [
        '//button[contains(., "Tout accepter")]',
        '//button[contains(., "Accept all")]',
        '//button[contains(., "Accepter")]',
        '//button[contains(., "Accept")]',
    ]

    for selector in selectors:
        try:
            button = driver.find_element(By.XPATH, selector)
            driver.execute_script("arguments[0].click();", button)
            print("Cookies acceptés.")
            time.sleep(2)
            return True
        except Exception:
            continue

    print("Pas de bouton cookies trouvé.")
    return False


# ============================
# SHORTS
# ============================

def click_shorts(driver):
    selectors = [
        '//a[contains(@href, "/shorts")]',
        '//*[normalize-space(text())="Shorts"]',
        '//a[contains(@title, "Shorts")]',
        '//a[contains(@aria-label, "Shorts")]',
    ]

    for selector in selectors:
        try:
            button = driver.find_element(By.XPATH, selector)
            driver.execute_script("arguments[0].click();", button)
            print("Shorts ouvert.")
            time.sleep(3)
            return True
        except Exception:
            continue

    print("Impossible de cliquer sur Shorts.")
    return False


# ============================
# NAVIGATION
# ============================

def next_video(driver):
    old_url = driver.current_url

    for attempt in range(1, 5):
        try:
            driver.execute_script("document.body.click();")
            time.sleep(0.3)

            ActionChains(driver).send_keys(Keys.ARROW_DOWN).perform()
            print(f"Tentative flèche bas {attempt}/4")
            time.sleep(2)

            if driver.current_url != old_url and "/shorts/" in driver.current_url:
                print("Vidéo suivante.")
                time.sleep(WAIT_AFTER_NEXT)
                return True

            ActionChains(driver).send_keys(Keys.PAGE_DOWN).perform()
            print(f"Tentative page down {attempt}/4")
            time.sleep(2)

            if driver.current_url != old_url and "/shorts/" in driver.current_url:
                print("Vidéo suivante.")
                time.sleep(WAIT_AFTER_NEXT)
                return True

        except Exception as e:
            print("Erreur navigation :", e)

    print("Impossible de passer à la vidéo suivante.")
    return False


# ============================
# COLLECTE
# ============================

def collect(driver, session_id):
    rows = []

    valid_count = 0
    total_seen = 0
    sponsored_skipped = 0

    valid_url_history = []
    valid_category_history = []

    while valid_count < NB_VALID_VIDEOS and total_seen < MAX_TOTAL_VIDEOS:
        total_seen += 1
        time.sleep(WAIT_BEFORE_EXTRACTION)

        url = driver.current_url
        video_id = extract_video_id(url)
        category = get_category(url)

        if is_sponsored(driver):
            sponsored_skipped += 1

            print("\nVidéo ignorée car sponsorisée")
            print("URL :", url)
            print("VIDEO ID :", video_id)
            print("CATEGORY :", category)
            print("SPONSORED SKIPPED :", sponsored_skipped)

            time.sleep(WATCH_TIME_SPONSORED)

            if not next_video(driver):
                break

            continue

        valid_count += 1

        previous_url = valid_url_history[-1] if len(valid_url_history) >= 1 else None
        previous_2_url = valid_url_history[-2] if len(valid_url_history) >= 2 else None

        previous_category = valid_category_history[-1] if len(valid_category_history) >= 1 else None
        previous_2_category = valid_category_history[-2] if len(valid_category_history) >= 2 else None

        row = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "condition": "neutral",
            "history_category": "neutral",
            "history_intensity": 0,
            "history_video_count": 0,
            "video_id": video_id,
            "url": url,
            "previous_url": previous_url,
            "previous_2_url": previous_2_url,
            "category": category,
            "previous_category": previous_category,
            "previous_2_category": previous_2_category,
            "depth": valid_count,
            "total_seen_depth": total_seen,
            "watch_time": WATCH_TIME_COLLECT,
            "is_sponsored": 0,
            "include_in_prediction": 1,
            "sponsored_skipped_before": sponsored_skipped,
        }

        rows.append(row)

        valid_url_history.append(url)
        valid_category_history.append(category)

        print(f"\nVidéo valide {valid_count}/{NB_VALID_VIDEOS}")
        print("URL :", url)
        print("VIDEO ID :", video_id)
        print("CATEGORY :", category)
        print("PREVIOUS CATEGORY :", previous_category)
        print("PREVIOUS 2 CATEGORY :", previous_2_category)
        print("TOTAL SEEN DEPTH :", total_seen)
        print("SPONSORED SKIPPED BEFORE :", sponsored_skipped)

        time.sleep(WATCH_TIME_COLLECT)

        if valid_count < NB_VALID_VIDEOS:
            if not next_video(driver):
                break

    return rows


# ============================
# UNE SESSION
# ============================

def run_session(session_number):
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{OUTPUT_PREFIX}_{session_id}.csv"

    options = Options()
    options.add_argument("--incognito")
    options.add_argument("--start-maximized")

    driver = webdriver.Chrome(options=options)

    try:
        print("\n==============================")
        print(f"SESSION {session_number}/{NB_SESSIONS}")
        print("SESSION ID :", session_id)
        print("CONDITION : neutral")
        print("==============================")

        driver.get("https://www.youtube.com/")
        time.sleep(4)

        accept_cookies(driver)

        if not click_shorts(driver):
            print("STOP : impossible d'ouvrir Shorts.")
            return

        rows = collect(driver=driver, session_id=session_id)

        if rows:
            pd.DataFrame(rows).to_csv(
                output_file,
                index=False,
                encoding="utf-8-sig"
            )

            print("\nFichier session créé :", output_file)
            print("Vidéos valides collectées :", len(rows))

        else:
            print("\nAucune donnée valide collectée.")

    finally:
        try:
            driver.quit()
        except Exception:
            pass


# ============================
# FUSION
# ============================

def merge_csv_files():
    files = glob.glob(f"{OUTPUT_PREFIX}_*.csv")

    if not files:
        print("\nAucun fichier CSV à fusionner.")
        return

    dfs = []

    print("\nFichiers fusionnés :")

    for file in files:
        try:
            print("-", file)
            dfs.append(pd.read_csv(file))
        except Exception as e:
            print("Erreur lecture :", file, e)

    if not dfs:
        print("\nAucun CSV lisible.")
        return

    final_df = pd.concat(dfs, ignore_index=True)

    final_df.to_csv(
        FINAL_DATASET,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nFusion terminée.")
    print("Fichier créé :", FINAL_DATASET)
    print("Nombre total de lignes :", len(final_df))
    print("Nombre de sessions :", final_df["session_id"].nunique())

    if "category" in final_df.columns:
        print("\nRépartition des catégories recommandées :")
        print(final_df["category"].value_counts(dropna=False))


# ============================
# MAIN
# ============================

def main():
    try:
        for session_number in range(1, NB_SESSIONS + 1):
            run_session(session_number)

            if session_number < NB_SESSIONS:
                print(
                    f"\nPause de {PAUSE_BETWEEN_SESSIONS} secondes "
                    f"avant la session suivante."
                )
                time.sleep(PAUSE_BETWEEN_SESSIONS)

        merge_csv_files()

    except KeyboardInterrupt:
        print("\nArrêt manuel détecté.")
        print("Fusion des CSV déjà créés...")
        merge_csv_files()
        print("Arrêt propre terminé.")


if __name__ == "__main__":
    main()