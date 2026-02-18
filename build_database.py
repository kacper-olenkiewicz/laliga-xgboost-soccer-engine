import pandas as pd
import requests
import io
import numpy as np
from utils import check_form, calculate_stats


URLS = [
    "https://www.football-data.co.uk/mmz4281/2425/SP1.csv", 
    "https://www.football-data.co.uk/mmz4281/2324/SP1.csv",
    "https://www.football-data.co.uk/mmz4281/2223/SP1.csv",
    "https://www.football-data.co.uk/mmz4281/2122/SP1.csv",
    "https://www.football-data.co.uk/mmz4281/2021/SP1.csv",
    "https://www.football-data.co.uk/mmz4281/1920/SP1.csv"
]

def download_data():

    all_data = []
    
    for url in URLS:
        try:           
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(url, headers=headers)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            all_data.append(df)
        except Exception as e:
            print(f"Błąd przy pobieraniu {url}: {e}")

    
    full_df = pd.concat(all_data, ignore_index=True)
    
    
    full_df['Date'] = pd.to_datetime(full_df['Date'], dayfirst=True, errors='coerce')
    full_df = full_df.sort_values('Date').reset_index(drop=True)
    return full_df

def process_history(df):   
    cols_to_add = [
        'HomeGaz', 'AwayGaz', 
        'HomeScoredAvg', 'HomeLosedAvg', 'HomeShotsAvg', 'HomeSTargetAvg', 'HomeCornersAvg',
        'AwayScoredAvg', 'AwayLosedAvg', 'AwayShotsAvg', 'AwaySTargetAvg', 'AwayCornersAvg',
        'EffDiff', 'HomeClean_sheets', 'AwayClean_sheets', 'MarketDiff'
    ]
    
    for col in cols_to_add:
        df[col] = 0.0

    
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

        
        try:
           
            if pd.notna(row['B365H']) and pd.notna(row['B365A']) and row['B365H'] > 0:
                prob_h = 1 / row['B365H']
                prob_a = 1 / row['B365A']
                df.at[index, 'MarketDiff'] = prob_h - prob_a
            else:
                df.at[index, 'MarketDiff'] = 0
        except:
            df.at[index, 'MarketDiff'] = 0

        

   
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
        print("Nie udało się pobrać danych.")

if __name__ == "__main__":
    main()