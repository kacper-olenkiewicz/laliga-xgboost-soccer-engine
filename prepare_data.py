import pandas as pd
from utils import check_form , calculate_stats

def main():
    url = "https://www.football-data.co.uk/mmz4281/2425/SP1.csv"
    df = pd.read_csv(url)

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['FTHG'] = df['FTHG'].fillna(0).astype(int)
    df['FTAG'] = df['FTAG'].fillna(0).astype(int)
    kolumny_do_analizy = [
    'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 
    'HS', 'AS',   
    'HST', 'AST', 
    'HC', 'AC',
    'B365H', 'B365D', 'B365A'    
    ]
    df_main = df[kolumny_do_analizy].copy()
    
    for i in range(len(df_main)):
        current = df_main.iloc[i]
        
        
        h_gaz = check_form(current['HomeTeam'], df_main, current['Date'])
        a_gaz = check_form(current['AwayTeam'], df_main, current['Date'])
        
        h_stats= calculate_stats(current['HomeTeam'], df_main, current['Date'])
        a_stats = calculate_stats(current['AwayTeam'], df_main, current['Date'])
        

        
        df_main.at[i, 'HomeGaz'] = h_gaz
        df_main.at[i, 'HomeScoredAvg'] = h_stats['scored']
        df_main.at[i, 'HomeLosedAvg'] = h_stats['losed']
        df_main.at[i, 'HomeShotsAvg'] = h_stats['shots']
        df_main.at[i, 'HomeSTargetAvg'] = h_stats['shots_target']
        df_main.at[i, 'HomeCornersAvg'] = h_stats['corners']
        df_main.at[i,'HomeEfficiency'] = h_stats['efficiency']
        df_main.at[i,'HomeClean_sheets'] = h_stats['clean_sheets']
        df_main.at[i,'HomeMarketProb'] = h_stats['market_prob']
    


        df_main.at[i, 'AwayGaz'] = a_gaz
        df_main.at[i, 'AwayScoredAvg'] = a_stats['scored']
        df_main.at[i, 'AwayLosedAvg'] = a_stats['losed']
        df_main.at[i, 'AwayShotsAvg'] = a_stats['shots']
        df_main.at[i, 'AwaySTargetAvg'] = a_stats['shots_target']
        df_main.at[i, 'AwayCornersAvg'] = a_stats['corners']
        df_main.at[i,'AwayEfficiency'] = a_stats['efficiency']
        df_main.at[i,'AwayClean_sheets'] = a_stats['clean_sheets']
        df_main.at[i,'AwayMarketProb'] = a_stats['market_prob']

    df_main.to_csv('processed_data.csv', index=False)

if __name__ == "__main__":
    main()