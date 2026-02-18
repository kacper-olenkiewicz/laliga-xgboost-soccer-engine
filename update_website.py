import pandas as pd
import json
import os
from xgboost import XGBClassifier
from utils import check_form, calculate_stats

def calculate_fair_odds(home_gaz, away_gaz):
    base_h = 0.45 
    diff = home_gaz - away_gaz
    prob_h = max(0.1, min(0.9, base_h + (diff * 0.02)))
    prob_a = max(0.1, min(0.9, 0.30 - (diff * 0.02)))
    return round(1/prob_h, 2), round(1/prob_a, 2)

def main():
    try:
        df = pd.read_csv('processed_data.csv')
        df['Date'] = pd.to_datetime(df['Date'])
    except FileNotFoundError:
        print("BŁĄD: Brak pliku processed_data.csv!")
        return
    
    try:
        fixtures = pd.read_csv('weekend_fixtures.csv').to_dict('records')
    except FileNotFoundError:
        print("Brak pliku weekend_fixtures.csv! Uruchom najpierw scraper.py")
        return

    results_for_json = []

    train_df = df.dropna(subset=['FTR']).copy()
    target_map = {'H': 2, 'A': 0, 'D': 1}
    y_train = train_df['FTR'].map(target_map)

    features = [
        'HomeGaz', 'AwayGaz', 
        'HomeScoredAvg', 'HomeLosedAvg', 'HomeShotsAvg', 'HomeSTargetAvg', 'HomeCornersAvg',
        'AwayScoredAvg', 'AwayLosedAvg', 'AwayShotsAvg', 'AwaySTargetAvg', 'AwayCornersAvg',
        'EffDiff', 'HomeClean_sheets', 'AwayClean_sheets', 'GD_Diff', 'MarketDiff'
    ]

    available_features = [f for f in features if f in df.columns]
    X_train = train_df[available_features]

    model = XGBClassifier(n_estimators=50, learning_rate=0.01, max_depth=3, subsample=0.8, random_state=42)
    model.fit(X_train, y_train)

    future_date = "2030-01-01"

    for match in fixtures:
        home = match['Home']
        away = match['Away']
        

        h_stats = calculate_stats(home, df, future_date)
        a_stats = calculate_stats(away, df, future_date)
        h_gaz = check_form(home, df, future_date)
        a_gaz = check_form(away, df, future_date)

        odds_h, odds_a = match['OddsH'], match['OddsA']
        is_synthetic = False
        if odds_h == 0:
            odds_h, odds_a = calculate_fair_odds(h_gaz, a_gaz)
            is_synthetic = True


        input_row = pd.DataFrame([{
            'HomeGaz': h_gaz, 'AwayGaz': a_gaz,
            'HomeScoredAvg': h_stats['scored'], 'HomeLosedAvg': h_stats['losed'],
            'HomeShotsAvg': h_stats['shots'], 'HomeSTargetAvg': h_stats['shots_target'],
            'HomeCornersAvg': h_stats['corners'], 'AwayScoredAvg': a_stats['scored'],
            'AwayLosedAvg': a_stats['losed'], 'AwayShotsAvg': a_stats['shots'],
            'AwaySTargetAvg': a_stats['shots_target'], 'AwayCornersAvg': a_stats['corners'],
            'EffDiff': h_stats['efficiency'] - a_stats['efficiency'],
            'HomeClean_sheets': h_stats['clean_sheets'], 'AwayClean_sheets': a_stats['clean_sheets'],
            'GD_Diff': (h_stats['scored'] - h_stats['losed']) - (a_stats['scored'] - a_stats['losed']),
            'MarketDiff': (1/odds_h) - (1/odds_a)
        }])

        input_row = input_row[available_features]
        
        probs = model.predict_proba(input_row)[0]
        
        p_home = float(probs[2] * 100)
        p_draw = float(probs[1] * 100)
        p_away = float(probs[0] * 100)

        rec = "Ryzykowny mecz "
        match_class = "neutral"

        if p_home > 55:
            rec = "Wygrana Gospodarzy "
            match_class = "win"
        elif p_away > 55:
            rec = "Wygrana Gości "
            match_class = "lose"
        elif (p_home + pd) > 75:
            rec = "Gospodarz lub Remis "
            match_class = "safe"
        elif (p_away + pd) > 75:
            rec = "Goście lub Remis "
            match_class = "safe"

        results_for_json.append({
            "home": input_row['HomeTeam'],
            "away": input_row['AwayTeam'],
            "p_home": round(p_home, 1),
            "p_draw": round(p_draw, 1),
            "p_away": round(p_away, 1),
            "rec": rec,            
            "class": match_class
        })

    if not os.path.exists('docs'):
        os.makedirs('docs')
        
    with open('docs/matches.json', 'w', encoding='utf-8') as f:
        json.dump(results_for_json, f, ensure_ascii=False, indent=4)
        print(f" SUKCES! Zapisano {len(results_for_json)} meczy do pliku docs/matches.json")

if __name__ == "__main__":
    main()



    