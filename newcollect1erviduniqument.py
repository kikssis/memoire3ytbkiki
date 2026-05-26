import glob
import re
import time
import random
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

BASE_DATASET = r"C:\Users\defre\OneDrive\Desktop\memoire3_youtube\dataset_clean.csv"
FINAL_DATASET_2 = r"C:\Users\defre\OneDrive\Desktop\memoire3_youtube\dataset_clean_2.csv"

NB_SESSIONS = 80
NB_VALID_VIDEOS = 1
MAX_TOTAL_VIDEOS = 20

HISTORY_INTENSITIES = [1, 3, 8]

WATCH_TIME_HISTORY = 5
WATCH_TIME_COLLECT = 1.5
WATCH_TIME_SPONSORED = 2

WAIT_BEFORE_EXTRACTION = 3
WAIT_AFTER_NEXT = 2
PAUSE_BETWEEN_SESSIONS = 3

OUTPUT_PREFIX = "first_video"


SPORT_URLS = [
    "https://www.youtube.com/shorts/O88fwmKRBEY",
    "https://www.youtube.com/shorts/997RwWpxRVI",
    "https://www.youtube.com/shorts/CPvoejF9k2o",
    "https://www.youtube.com/shorts/JNYz4YdOgFk",
    "https://www.youtube.com/shorts/2j3oXXC577E",
    "https://www.youtube.com/shorts/VY2tvycMQRw",
    "https://www.youtube.com/shorts/tRhl9ucHckU",
    "https://www.youtube.com/shorts/OmM-_LN6za8",
]

GAMING_URLS = [
    "https://www.youtube.com/shorts/3d3Z_lb90No",
    "https://www.youtube.com/shorts/OHj3VqTQgPU",
    "https://www.youtube.com/shorts/2-67twDuY-Q",
    "https://www.youtube.com/shorts/TLzHydYT00g",
    "https://www.youtube.com/shorts/nQW-9gXiAfU",
    "https://www.youtube.com/shorts/XMT5tPb2bek",
    "https://www.youtube.com/shorts/13cNpAy-BTs",
    "https://www.youtube.com/shorts/j5CMZQrx_zQ",
]

NEWS_URLS = [
    "https://www.youtube.com/shorts/fo9JEb8mAak",
    "https://www.youtube.com/shorts/GqhuK3Vydwc",
    "https://www.youtube.com/shorts/NboNpaRQow8",
    "https://www.youtube.com/shorts/qqRmkxzXH00",
    "https://www.youtube.com/shorts/BXLxH8Uv2Sw",
    "https://www.youtube.com/shorts/m4kdhs0afIk",
    "https://www.youtube.com/shorts/sw5sOjzSBXM",
    "https://www.youtube.com/shorts/J5UoXbM9B10",
]

SCIENCE_URLS = [
    "https://www.youtube.com/shorts/VxPFxDiAUko",
    "https://www.youtube.com/shorts/p_hmH6rHr4E",
    "https://www.youtube.com/shorts/OHMYzz2eu4w",
    "https://www.youtube.com/shorts/1vH7ZKskgbc",
    "https://www.youtube.com/shorts/yq44_HaOv8s",
    "https://www.youtube.com/shorts/1QtV7Gpb5XQ",
    "https://www.youtube.com/shorts/gJOAj4kjMGY",
    "https://www.youtube.com/shorts/j9MqcRAtqeg",
]

CONDITIONS = {
    "sport": SPORT_URLS,
    "gaming": GAMING_URLS,
    "news": NEWS_URLS,
    "science": SCIENCE_URLS,
}


# ============================
# OUTILS
# ============================

def create_driver():
    options = Options()
    options.add_argument("--incognito")
    options.add_argument("--start-maximized")

    # Réduction des risques Out of Memory
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(60)
    return driver


def extract_video_id(url):
    if url and "/shorts/" in url:
        return url.split("/shorts/")[-1].split("?")[0]
    return None


def get_raw_html(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/127.0.0.0 Safari/537.36"
        )
    }

    req = Request(url, headers=headers)

    with urlopen(req, timeout=15) as response:
        return response.read().decode("utf-8", errors="ignore")


def get_category(url):
    try:
        html = get_raw_html(url)

        patterns = [
            r'"category":"([^"]+)"',
            r'"category":\{"simpleText":"([^"]+)"\}',
            r'<meta itemprop="genre" content="([^"]+)"',
        ]

        for pattern in patterns:
            match = re.search(pattern, html)
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
            for element in driver.find_elements(By.XPATH, selector):
                try:
                    if element.is_displayed():
                        return True
                except Exception:
                    continue

        return False

    except Exception:
        return False


def safe_get(driver, url, retries=3):
    for attempt in range(1, retries + 1):
        try:
            driver.get(url)
            time.sleep(random.uniform(3, 5))
            return True
        except Exception as e:
            print(f"Erreur chargement {attempt}/{retries} :", url)
            print(e)
            time.sleep(5)

    print("URL ignorée après plusieurs essais :", url)
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

def open_shorts_directly(driver):
    try:
        driver.get("https://www.youtube.com/shorts")
        time.sleep(6)

        if "/shorts/" in driver.current_url or "/shorts" in driver.current_url:
            print("Shorts ouvert directement.")
            return True

        print("Shorts ouvert mais URL inattendue :", driver.current_url)
        return True

    except Exception as e:
        print("Impossible d'ouvrir Shorts directement :", e)
        return False


def open_new_shorts_tab(driver):
    old_tab = driver.current_window_handle

    driver.execute_script("window.open('https://www.youtube.com/shorts', '_blank');")
    time.sleep(6)

    new_tab = driver.window_handles[-1]
    driver.switch_to.window(new_tab)

    try:
        driver.switch_to.window(old_tab)
        driver.close()
        driver.switch_to.window(new_tab)
        print("Ancien onglet fermé. Shorts ouvert dans nouvel onglet.")
        time.sleep(3)
    except Exception as e:
        print("Impossible de fermer ancien onglet :", e)
        driver.switch_to.window(new_tab)

    return True


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
# HISTORIQUE
# ============================

def watch_history(driver, urls, history_category, history_intensity):
    history_urls = urls[:history_intensity]

    for i, url in enumerate(history_urls, start=1):
        print(f"Historique {history_category} {i}/{len(history_urls)} : {url}")

        success = safe_get(driver, url)

        if not success:
            print("Vidéo historique ignorée.")
            continue

        if i == 1:
            accept_cookies(driver)

        time.sleep(WATCH_TIME_HISTORY)
        time.sleep(random.uniform(1, 3))


# ============================
# COLLECTE PREMIÈRE VIDÉO
# ============================

def collect_first_video(driver, session_id, condition_name, history_category, history_intensity):
    rows = []

    valid_count = 0
    total_seen = 0
    sponsored_skipped = 0

    while valid_count < NB_VALID_VIDEOS and total_seen < MAX_TOTAL_VIDEOS:
        total_seen += 1
        time.sleep(WAIT_BEFORE_EXTRACTION)

        url = driver.current_url
        video_id = extract_video_id(url)
        category = get_category(url)

        if category is None:
            print("Catégorie non trouvée pour :", url)

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

        row = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "condition": condition_name,
            "history_category": history_category,
            "history_intensity": history_intensity,
            "history_video_count": history_intensity,
            "video_id": video_id,
            "url": url,
            "category": category,
            "depth": 1,
            "total_seen_depth": total_seen,
            "watch_time": WATCH_TIME_COLLECT,
            "is_sponsored": 0,
            "include_in_prediction": 1,
            "sponsored_skipped_before": sponsored_skipped,
            "source_file": None,
        }

        rows.append(row)

        print("\nPremière vidéo valide collectée")
        print("URL :", url)
        print("VIDEO ID :", video_id)
        print("CATEGORY :", category)
        print("CONDITION :", condition_name)

    return rows


# ============================
# SESSIONS
# ============================

def count_completed_sessions(condition_name):
    files = glob.glob(f"{OUTPUT_PREFIX}_{condition_name}_*.csv")
    return len(files)


def run_neutral_session(session_number):
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    condition_name = "neutral"
    output_file = f"{OUTPUT_PREFIX}_{condition_name}_{session_id}.csv"

    driver = create_driver()

    try:
        print("\n==============================")
        print(f"SESSION NEUTRE {session_number}/{NB_SESSIONS}")
        print("SESSION ID :", session_id)
        print("==============================")

        driver.get("https://www.youtube.com/")
        time.sleep(4)

        accept_cookies(driver)

        if not open_shorts_directly(driver):
            return False

        rows = collect_first_video(
            driver=driver,
            session_id=session_id,
            condition_name="neutral",
            history_category="neutral",
            history_intensity=0
        )

        if rows:
            for row in rows:
                row["source_file"] = output_file

            pd.DataFrame(rows).to_csv(output_file, index=False, encoding="utf-8-sig")
            print("Fichier créé :", output_file)
            return True

        return False

    except Exception as e:
        print("ERREUR SESSION NEUTRE :", e)
        return False

    finally:
        try:
            driver.quit()
        except Exception:
            pass


def run_history_session(session_number, history_category, history_intensity, urls):
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    condition_name = f"{history_category}_history_{history_intensity}"
    output_file = f"{OUTPUT_PREFIX}_{condition_name}_{session_id}.csv"

    driver = create_driver()

    try:
        print("\n==============================")
        print(f"SESSION {condition_name} {session_number}/{NB_SESSIONS}")
        print("SESSION ID :", session_id)
        print("==============================")

        watch_history(driver, urls, history_category, history_intensity)

        if not open_new_shorts_tab(driver):
            return False

        rows = collect_first_video(
            driver=driver,
            session_id=session_id,
            condition_name=condition_name,
            history_category=history_category,
            history_intensity=history_intensity
        )

        if rows:
            for row in rows:
                row["source_file"] = output_file

            pd.DataFrame(rows).to_csv(output_file, index=False, encoding="utf-8-sig")
            print("Fichier créé :", output_file)
            return True

        return False

    except Exception as e:
        print("ERREUR SESSION :", e)
        return False

    finally:
        try:
            driver.quit()
        except Exception:
            pass


# ============================
# DATASET 2
# ============================

def create_dataset_2():
    new_files = glob.glob(f"{OUTPUT_PREFIX}_*.csv")

    if not new_files:
        print("Aucun nouveau fichier à fusionner.")
        return

    new_dfs = []

    for file in new_files:
        try:
            temp = pd.read_csv(file)
            temp["source_file"] = file
            new_dfs.append(temp)
        except Exception as e:
            print("Erreur lecture :", file, e)

    if not new_dfs:
        print("Aucun nouveau CSV lisible.")
        return

    new_df = pd.concat(new_dfs, ignore_index=True)

    base_df = pd.read_csv(BASE_DATASET)

    combined = pd.concat([base_df, new_df], ignore_index=True, sort=False)

    combined["category"] = combined["category"].fillna("Unknown")

    counts = combined["category"].value_counts()
    rare_categories = counts[counts < 10].index

    combined["category_clean"] = combined["category"].replace(rare_categories, "Other")

    combined["target_category"] = combined["history_category"].map({
        "neutral": "neutral",
        "sport": "Sports",
        "gaming": "Gaming",
        "news": "News & Politics",
        "science": "Science & Technology",
    }).fillna("Unknown")

    combined["include_in_prediction"] = combined["include_in_prediction"].fillna(1)

    combined.to_csv(FINAL_DATASET_2, index=False, encoding="utf-8-sig")

    print("\nDATASET 2 créé :", FINAL_DATASET_2)
    print("Nombre total de lignes :", len(combined))
    print("Nombre de nouvelles lignes première vidéo :", len(new_df))
    print("\nRépartition nouvelles conditions :")
    print(new_df["condition"].value_counts(dropna=False))
    print("\nRépartition nouvelles catégories :")
    print(new_df["category"].value_counts(dropna=False))


# ============================
# MAIN
# ============================

def main():
    try:
        completed_neutral = count_completed_sessions("neutral")

        print("\n################################")
        print("CONDITION NEUTRE")
        print("Sessions déjà faites :", completed_neutral)
        print("################################")

        for session_number in range(completed_neutral + 1, NB_SESSIONS + 1):
            run_neutral_session(session_number)
            time.sleep(PAUSE_BETWEEN_SESSIONS)

        for history_category, urls in CONDITIONS.items():
            for history_intensity in HISTORY_INTENSITIES:

                condition_name = f"{history_category}_history_{history_intensity}"
                completed = count_completed_sessions(condition_name)

                print("\n################################")
                print("CONDITION :", condition_name)
                print("Sessions déjà faites :", completed)
                print("################################")

                if completed >= NB_SESSIONS:
                    continue

                for session_number in range(completed + 1, NB_SESSIONS + 1):
                    run_history_session(
                        session_number=session_number,
                        history_category=history_category,
                        history_intensity=history_intensity,
                        urls=urls
                    )

                    time.sleep(PAUSE_BETWEEN_SESSIONS)

        create_dataset_2()

    except KeyboardInterrupt:
        print("\nArrêt manuel détecté.")
        print("Création du dataset 2 avec les fichiers déjà existants...")
        create_dataset_2()
        print("Arrêt propre terminé.")


if __name__ == "__main__":
    main()