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

HISTORY_INTENSITIES = [1, 3, 8]

NB_SESSIONS = 15
HISTORY_CATEGORY = "gaming"

NB_VALID_VIDEOS = 15
MAX_TOTAL_VIDEOS = 60

WATCH_TIME_HISTORY = 5
WATCH_TIME_COLLECT = 1.5
WATCH_TIME_SPONSORED = 2

WAIT_BEFORE_EXTRACTION = 1.5
WAIT_AFTER_NEXT = 2
PAUSE_BETWEEN_SESSIONS = 3


# ============================
# OUTILS
# ============================

def extract_video_id(url):
    if url and "/shorts/" in url:
        return url.split("/shorts/")[-1].split("?")[0]
    return None


def safe_get(driver, url, retries=3):
    for attempt in range(1, retries + 1):
        try:
            driver.get(url)
            time.sleep(random.uniform(2, 4))
            return True
        except Exception as e:
            print(f"Erreur chargement {attempt}/{retries} :", url)
            print(e)
            time.sleep(5)

    print("URL ignorée après plusieurs essais :", url)
    return False


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


def open_new_shorts_tab(driver):
    old_tab = driver.current_window_handle

    driver.execute_script("window.open('https://www.youtube.com/', '_blank');")
    time.sleep(4)

    new_tab = driver.window_handles[-1]
    driver.switch_to.window(new_tab)

    if not click_shorts(driver):
        return False

    try:
        driver.switch_to.window(old_tab)
        driver.close()
        driver.switch_to.window(new_tab)
        print("Ancien onglet fermé.")
        time.sleep(2)
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

def watch_history(driver, history_intensity):
    if len(GAMING_URLS) < history_intensity:
        raise ValueError(
            f"Pas assez d'URLs gaming : {len(GAMING_URLS)} disponibles, "
            f"mais {history_intensity} demandées."
        )

    history_urls = GAMING_URLS[:history_intensity]

    for i, url in enumerate(history_urls, start=1):
        print(f"Historique {HISTORY_CATEGORY} {i}/{len(history_urls)} : {url}")

        success = safe_get(driver, url)

        if not success:
            print("Vidéo historique ignorée.")
            continue

        if i == 1:
            accept_cookies(driver)

        time.sleep(WATCH_TIME_HISTORY)
        time.sleep(random.uniform(1, 3))


# ============================
# COLLECTE
# ============================

def collect(driver, session_id, condition_name, history_intensity):
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
            "history_category": HISTORY_CATEGORY,
            "history_intensity": history_intensity,
            "history_video_count": history_intensity,
            "video_id": video_id,
            "url": url,
            "category": category,
            "depth": valid_count,
            "total_seen_depth": total_seen,
            "watch_time": WATCH_TIME_COLLECT,
            "is_sponsored": 0,
            "include_in_prediction": 1,
            "sponsored_skipped_before": sponsored_skipped,
        }

        rows.append(row)

        print(f"\nVidéo valide {valid_count}/{NB_VALID_VIDEOS}")
        print("URL :", url)
        print("VIDEO ID :", video_id)
        print("CATEGORY :", category)
        print("HISTORY CATEGORY :", HISTORY_CATEGORY)
        print("HISTORY INTENSITY :", history_intensity)
        print("TOTAL SEEN DEPTH :", total_seen)
        print("SPONSORED SKIPPED BEFORE :", sponsored_skipped)

        time.sleep(WATCH_TIME_COLLECT)

        if valid_count < NB_VALID_VIDEOS:
            if not next_video(driver):
                break

    return rows


# ============================
# SESSION
# ============================

def run_session(session_number, history_intensity):
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    condition_name = f"{HISTORY_CATEGORY}_history_{history_intensity}"
    output_prefix = f"collecte_{HISTORY_CATEGORY}_history_{history_intensity}"
    output_file = f"{output_prefix}_{session_id}.csv"

    options = Options()
    options.add_argument("--incognito")
    options.add_argument("--start-maximized")

    driver = webdriver.Chrome(options=options)

    try:
        driver.set_page_load_timeout(60)

        print("\n==============================")
        print(f"SESSION {session_number}/{NB_SESSIONS}")
        print("SESSION ID :", session_id)
        print("CONDITION :", condition_name)
        print("HISTORY CATEGORY :", HISTORY_CATEGORY)
        print("HISTORY INTENSITY :", history_intensity)
        print("==============================")

        watch_history(driver, history_intensity)

        if not open_new_shorts_tab(driver):
            print("STOP : impossible d'ouvrir Shorts.")
            return False

        rows = collect(
            driver=driver,
            session_id=session_id,
            condition_name=condition_name,
            history_intensity=history_intensity
        )

        if rows:
            pd.DataFrame(rows).to_csv(
                output_file,
                index=False,
                encoding="utf-8-sig"
            )

            print("\nFichier session créé :", output_file)
            print("Vidéos valides collectées :", len(rows))
            return True

        else:
            print("\nAucune donnée valide collectée.")
            return False

    except Exception as e:
        print("\nERREUR SESSION :", e)
        return False

    finally:
        try:
            driver.quit()
        except Exception:
            pass


# ============================
# REPRISE AUTOMATIQUE
# ============================

def count_completed_sessions(history_intensity):
    pattern = f"collecte_{HISTORY_CATEGORY}_history_{history_intensity}_*.csv"
    files = glob.glob(pattern)
    return len(files)


# ============================
# FUSION
# ============================

def merge_csv_files():
    files = glob.glob(f"collecte_{HISTORY_CATEGORY}_history_*.csv")

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

    final_dataset = f"dataset_{HISTORY_CATEGORY}_history_all_intensities.csv"

    final_df.to_csv(
        final_dataset,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nFusion terminée.")
    print("Fichier créé :", final_dataset)
    print("Nombre total de lignes :", len(final_df))
    print("Nombre de sessions :", final_df["session_id"].nunique())

    if "history_intensity" in final_df.columns:
        print("\nRépartition des intensités d'historique :")
        print(final_df["history_intensity"].value_counts(dropna=False))

    if "condition" in final_df.columns:
        print("\nRépartition des conditions :")
        print(final_df["condition"].value_counts(dropna=False))

    if "category" in final_df.columns:
        print("\nRépartition des catégories recommandées :")
        print(final_df["category"].value_counts(dropna=False))


# ============================
# MAIN
# ============================

def main():
    try:
        for history_intensity in HISTORY_INTENSITIES:
            completed_sessions = count_completed_sessions(history_intensity)

            print("\n################################")
            print(f"HISTORIQUE {history_intensity} VIDÉO(S)")
            print("Sessions déjà faites :", completed_sessions)
            print("################################")

            if completed_sessions >= NB_SESSIONS:
                print(f"Déjà terminé pour intensité {history_intensity}.")
                continue

            for session_number in range(completed_sessions + 1, NB_SESSIONS + 1):
                success = run_session(session_number, history_intensity)

                if not success:
                    print("\nSession échouée mais le script continue.")
                    time.sleep(10)

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