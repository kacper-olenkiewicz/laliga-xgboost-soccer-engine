import pandas as pd
import requests
import io
import sys
import numpy as np
from datetime import datetime
from utils import check_form, calculate_stats


BASE_URL = "https://www.football-data.co.uk/mmz4281/{}/SP1.csv"
N_SEASONS = 7

def current_season_start(today=None):
    # Sezon zaczyna się w lipcu/sierpniu: październik 2026 -> sezon 2026/27
    today = today or datetime.now()
    return today.year if today.month >= 7 else today.year - 1

def season_codes(n=N_SEASONS, today=None):
    start = current_season_start(today)
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(start - n + 1, start + 1)]

def download_data():

    all_data = []
    headers = {"User-Agent": "Mozilla/5.0"}

    for code in season_codes():
        url = BASE_URL.format(code)
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 404:
                print(f"Brak pliku dla sezonu {code} (404) - pomijam.")
                continue
            r.raise_for_status()

            try:
                text = r.content.decode('utf-8-sig')
            except UnicodeDecodeError:
                text = r.content.decode('latin-1')
            if not text.strip():
                print(f"Pusty plik dla sezonu {code} - pomijam.")
                continue

            df = pd.read_csv(io.StringIO(text))
            df.columns = [c.strip() for c in df.columns]
            df = df.dropna(subset=['HomeTeam', 'Date'])
            if df.empty:
                print(f"Brak meczów w sezonie {code} - pomijam.")
                continue

            df['Season'] = code
            all_data.append(df)
            print(f"Sezon {code}: {len(df)} meczów")
        except Exception as e:
            print(f"Błąd przy pobieraniu {url}: {e}")

    if not all_data:
        return pd.DataFrame()

    full_df = pd.concat(all_data, ignore_index=True)


    full_df['Date'] = pd.to_datetime(full_df['Date'], dayfirst=True, errors='coerce')
    full_df = full_df.dropna(subset=['Date'])
    full_df = full_df.sort_values('Date').reset_index(drop=True)
    return full_df

def process_history(df):   
    cols_to_add = [
        'HomeGaz', 'AwayGaz', 
        'HomeScoredAvg', 'HomeLosedAvg', 'HomeShotsAvg', 'HomeSTargetAvg', 'HomeCornersAvg',
        'AwayScoredAvg', 'AwayLosedAvg', 'AwayShotsAvg', 'AwaySTargetAvg', 'AwayCornersAvg',
        'EffDiff', 'HomeClean_sheets', 'AwayClean_sheets', 'MarketDiff'
    ]
    
    # Wszystkie nowe kolumny dodane naraz (unika fragmentacji DataFrame i PerformanceWarning)
    df = pd.concat([df, pd.DataFrame(0.0, index=df.index, columns=cols_to_add)], axis=1)

    
    for index, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        date = row['Date']
        

        
        h_gaz = check_form(home, df, date)
        a_gaz = check_form(away, df, date)
        
        h_stats = calculate_stats(home, df, date)
        a_stats = calculate_stats(away, df, date)

        
        df.at[index, 'HomeGaz'] = h_gaz
        df.at[index, 'AwayGaz'] = a_gaz

        
        df.at[index, 'HomeScoredAvg'] = h_stats['scored']
        df.at[index, 'HomeLosedAvg'] = h_stats['losed']
        df.at[index, 'HomeShotsAvg'] = h_stats['shots']
        df.at[index, 'HomeSTargetAvg'] = h_stats['shots_target']
        df.at[index, 'HomeCornersAvg'] = h_stats['corners']
        df.at[index, 'HomeClean_sheets'] = h_stats['clean_sheets']
        
        df.at[index, 'AwayScoredAvg'] = a_stats['scored']
        df.at[index, 'AwayLosedAvg'] = a_stats['losed']
        df.at[index, 'AwayShotsAvg'] = a_stats['shots']
        df.at[index, 'AwaySTargetAvg'] = a_stats['shots_target']
        df.at[index, 'AwayCornersAvg'] = a_stats['corners']
        df.at[index, 'AwayClean_sheets'] = a_stats['clean_sheets']

        
        df.at[index, 'EffDiff'] = h_stats['efficiency'] - a_stats['efficiency']

        
        # Brak kursów -> NaN (XGBoost obsługuje braki natywnie, tak samo jak przy predykcji)
        odds_h = pd.to_numeric(row.get('B365H'), errors='coerce')
        odds_a = pd.to_numeric(row.get('B365A'), errors='coerce')
        if pd.notna(odds_h) and pd.notna(odds_a) and odds_h > 0 and odds_a > 0:
            df.at[index, 'MarketDiff'] = 1 / odds_h - 1 / odds_a
        else:
            df.at[index, 'MarketDiff'] = np.nan

    df['GD_Diff'] = (df['HomeScoredAvg'] - df['HomeLosedAvg']) - (df['AwayScoredAvg'] - df['AwayLosedAvg'])

    
    df_clean = df[(df['HomeGaz'] > 0) | (df['AwayGaz'] > 0)].copy()

    return df_clean

def main():
    
    df = download_data()
    
    
    if not df.empty:
        df_processed = process_history(df)
        
        
        filename = 'processed_data.csv'
        df_processed.to_csv(filename, index=False)
        print("Udało się pobrać dane")
    else:
        print("BŁĄD: Nie udało się pobrać danych.")
        sys.exit(1)

if __name__ == "__main__":
    main()