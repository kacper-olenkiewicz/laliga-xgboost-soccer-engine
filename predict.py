import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier

def main():
    try:
        df = pd.read_csv('processed_data.csv')
    except FileNotFoundError:
        print('Nie widzę pliku processed_data.csv')
        return

    nowy_mecz = pd.DataFrame({
        'Date': ['2026-01-18'], 
        'HomeTeam': ['Real Madrid'], 
        'AwayTeam': ['Barcelona'],
        'FTR': [np.nan],       
        
        
        'HomeGaz': [12],        
        'HomeScoredAvg': [2.4],
        'HomeLosedAvg': [0.8],
        'HomeShotsAvg': [14.5], 
        'HomeSTargetAvg': [6.2],
        'HomeCornersAvg': [5.5],
        'HomeEfficiency': [1.2],
        'HomeClean_sheets': [2],
        'HomeMarketProb': [0.6],

        
        'AwayGaz': [10],        
        'AwayScoredAvg': [1.9],
        'AwayLosedAvg': [1.2],
        'AwayShotsAvg': [13.0], 
        'AwaySTargetAvg': [5.8],
        'AwayCornersAvg': [4.8],
        'AwayEfficiency': [1.1],
        'AwayClean_sheets': [1],
        'AwayMarketProb': [0.3] 
    })

    df = pd.concat([df, nowy_mecz], ignore_index=True)


    df['EffDiff'] = df['HomeEfficiency'] - df['AwayEfficiency']
    df['GD_Diff'] = (df['HomeScoredAvg'] - df['HomeLosedAvg']) - (df['AwayScoredAvg'] - df['AwayLosedAvg'])
    

    if 'HomeMarketProb' in df.columns and 'AwayMarketProb' in df.columns:
        df['MarketDiff'] = df['HomeMarketProb'] - df['AwayMarketProb']
    else:
        df['MarketDiff'] = 0 


    features = [
        'HomeGaz','AwayGaz',
        'HomeScoredAvg', 'HomeLosedAvg', 'HomeShotsAvg', 'HomeSTargetAvg', 'HomeCornersAvg',
        'AwayScoredAvg', 'AwayLosedAvg', 'AwayShotsAvg', 'AwaySTargetAvg', 'AwayCornersAvg', 
        'EffDiff', 'HomeClean_sheets', 'AwayClean_sheets', 'GD_Diff', 'MarketDiff'
    ]


    train_data = df.dropna(subset=['FTR']).copy()
    
    
    future_data = df[df['FTR'].isna()].copy()

    
    target_map = {'H': 2, 'A': 0, 'D': 1}
    train_data['FTR_encoded'] = train_data['FTR'].map(target_map)

    
    X = train_data[features]
    y = train_data['FTR_encoded']


    train_data = train_data.sort_values('Date')
    
    split_point = int(len(train_data) * 0.8)
    
    X_train = X.iloc[:split_point]
    y_train = y.iloc[:split_point]
    
    X_test = X.iloc[split_point:]
    y_test = y.iloc[split_point:]

    print("--- KOLUMNY UŻYWANE DO NAUKI ---")
    print(X.columns.tolist())


    model = XGBClassifier(
        n_estimators=50,        
        learning_rate=0.01,     
        max_depth=3,            
        min_child_weight=5,     
        subsample=0.8,          
        random_state=42,
        eval_metric='mlogloss'
    )

    model.fit(X_train, y_train)
    

    y_pred_test = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred_test)
    
    print("\n" + "="*40)
    print("      MACIERZ BŁĘDÓW (Test Set)")
    print("="*40)
    print(f"Faktyczne A:       {cm[0][0]:^12} | {cm[0][1]:^12} | {cm[0][2]:^12}")
    print(f"Faktyczne D:       {cm[1][0]:^12} | {cm[1][1]:^12} | {cm[1][2]:^12}")
    print(f"Faktyczne H:       {cm[2][0]:^12} | {cm[2][1]:^12} | {cm[2][2]:^12}")
    print("="*40)

    print("\n--- RAPORT KLASYFIKACJI ---")
    print(classification_report(y_test, y_pred_test, target_names=['Away (A)', 'Draw (D)', 'Home (H)'], zero_division=0))


    scores = cross_val_score(model, X_train, y_train, cv=5) 
    print(f"Średnia skuteczność (CV): {scores.mean() * 100:.2f}%")

    importances = model.feature_importances_
    ranking = pd.DataFrame({
        'Statystyka': features,
        'Ważność (%)': importances * 100
    })
    ranking = ranking.sort_values(by='Ważność (%)', ascending=False)
    print("\n--- RANKING WAŻNOŚCI STATYSTYK ---")
    print(ranking.to_string(index=False))

    if future_data.empty:
        print('\nBrak nowych meczów do przewidzenia.')
        return

    print("\n--- TWOJE TYPY NA NAJBLIŻSZE MECZE ---")
    
    X_future = future_data[features]
    
    probabilities = model.predict_proba(X_future)

    for i in range(len(future_data)):
        mecz = future_data.iloc[i]
        
        prob_away = probabilities[i][0] * 100
        prob_draw = probabilities[i][1] * 100
        prob_home = probabilities[i][2] * 100

        print(f"\n(Gospodarz) {mecz['HomeTeam']} vs (Gość) {mecz['AwayTeam']}")
        print(f"  > GOSPODARZ: {prob_home:.1f}%")
        print(f"  > REMIS:     {prob_draw:.1f}%")
        print(f"  > GOŚĆ:      {prob_away:.1f}%")

if __name__ == '__main__':
    main()