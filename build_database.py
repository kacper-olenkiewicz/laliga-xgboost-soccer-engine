import pandas as pd
import requests
import sys
import pickle
import time
from datetime import datetime
from features import read_season, pick_odds, build_features

BASE_URL = "https://www.football-data.co.uk/mmz4281/{}/SP1.csv"
# Od sezonu 2000/01: pierwsze sezony służą tylko do "rozgrzania" Elo i Dixona-Colesa
# (model trenowany jest od 2005/06 - patrz TRAIN_FROM w update_website.py)
FIRST_SEASON = 2000
STATE_FILE = 'team_state.pkl'

def current_season_start(today=None):
    # Sezon zaczyna się w lipcu/sierpniu: październik 2026 -> sezon 2026/27
    today = today or datetime.now()
    return today.year if today.month >= 7 else today.year - 1

def season_codes(today=None):
    start = current_season_start(today)
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(FIRST_SEASON, start + 1)]

def download_data():

    all_data = []
    headers = {"User-Agent": "Mozilla/5.0"}
    session = requests.Session()

    for code in season_codes():
        url = BASE_URL.format(code)
        try:
            r = session.get(url, headers=headers, timeout=30)
            if r.status_code == 404:
                print(f"Brak pliku dla sezonu {code} (404) - pomijam.")
                continue
            r.raise_for_status()

            try:
                text = r.content.decode('utf-8-sig')
            except UnicodeDecodeError:
                text = r.content.decode('latin-1')

            df = read_season(text, code) if text.strip() else pd.DataFrame()
            if df.empty:
                print(f"Brak meczów w sezonie {code} - pomijam.")
                continue

            all_data.append(df)
            print(f"Sezon {code}: {len(df)} meczów")
        except Exception as e:
            print(f"Błąd przy pobieraniu {url}: {e}")

    if not all_data:
        return pd.DataFrame()

    full_df = pd.concat(all_data, ignore_index=True)
    return full_df.sort_values(['Date', 'HomeTeam'], kind='mergesort').reset_index(drop=True)

def main():
    t0 = time.time()
    df = download_data()

    if df.empty:
        print("BŁĄD: Nie udało się pobrać danych.")
        sys.exit(1)

    print(f"Pobrano {len(df)} meczów w {time.time() - t0:.0f} s. Liczę cechy...")
    t1 = time.time()
    df_processed, state = build_features(pick_odds(df))

    df_processed.to_csv('processed_data.csv', index=False)
    # Stan drużyn na dziś (Elo, EWM, Dixon-Coles) - do cech nadchodzących meczów w update_website.py
    with open(STATE_FILE, 'wb') as f:
        pickle.dump(state, f)
    print(f"Udało się pobrać dane: {len(df_processed)} meczów do {state.last_date.date()} "
          f"(cechy: {time.time() - t1:.0f} s)")

if __name__ == "__main__":
    main()
