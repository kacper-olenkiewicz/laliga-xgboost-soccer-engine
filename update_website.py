import pandas as pd
import json
import os
from xgboost import XGBClassifier
from utils import check_form, calculate_stats

def main():
    try:
        df = pd.read_csv('processed_data.csv')
        df['Date'] = pd.to_datetime(df['Date'])
    except FileNotFoundError:
        print("BŁĄD: Brak pliku processed_data.csv!")
        return
    fixtures = [
        {"Home": "Real Madrid", "Away": "Barcelona", "OddsH": 2.10, "OddsA": 3.20},
        {"Home": "Girona", "Away": "Sevilla", "OddsH": 1.85, "OddsA": 4.10},
        {"Home": "Betis", "Away": "Valencia", "OddsH": 2.05, "OddsA": 3.60},
        {"Home": "Las Palmas", "Away": "Mallorca", "OddsH": 2.80, "OddsA": 2.90}
    ]

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
        odds_h = match['OddsH']
        odds_a = match['OddsA']

        h_stats = calculate_stats(home, df, future_date)
        a_stats = calculate_stats(away, df, future_date)
        h_gaz = check_form(home, df, future_date)
        a_gaz = check_form(away, df, future_date)

        prob_h = (1 / odds_h)
        prob_a = (1 / odds_a)
        market_diff = prob_h - prob_a

        input_row = pd.DataFrame([{
            'HomeGaz': h_gaz, 'AwayGaz': a_gaz,
            'HomeScoredAvg': h_stats['scored'], 'HomeLosedAvg': h_stats['losed'],
            'HomeShotsAvg': h_stats['shots'], 'HomeSTargetAvg': h_stats['shots_target'], 'HomeCornersAvg': h_stats['corners'],
            'AwayScoredAvg': a_stats['scored'], 'AwayLosedAvg': a_stats['losed'],
            'AwayShotsAvg': a_stats['shots'], 'AwaySTargetAvg': a_stats['shots_target'], 'AwayCornersAvg': a_stats['corners'],
            'EffDiff': h_stats['efficiency'] - a_stats['efficiency'],
            'HomeClean_sheets': h_stats['clean_sheets'], 'AwayClean_sheets': a_stats['clean_sheets'],
            'GD_Diff': (h_stats['scored'] - h_stats['losed']) - (a_stats['scored'] - a_stats['losed']),
            'MarketDiff': market_diff
        }])

        input_row = input_row[available_features]
        
        probs = model.predict_proba(input_row)[0]

        rec = "No Bet"
        color_class = "neutral"
        
        p_home = float(probs[2] * 100)
        p_draw = float(probs[1] * 100)
        p_away = float(probs[0] * 100)

        if p_home > 50: 
            rec = "1 (Gospodarz)"; color_class = "win"
        elif p_away > 45: 
            rec = "2 (Gość)"; color_class = "lose"
        elif p_home > 40:
            rec = "1X (Bezpiecznie)"; color_class = "safe"

        results_for_json.append({
            "home": home,
            "away": away,
            "p_home": round(p_home, 1),
            "p_draw": round(p_draw, 1),
            "p_away": round(p_away, 1),
            "rec": rec,
            "class": color_class
        })

        if not os.path.exists('docs'):
            os.makedirs('docs')

        with open('docs/matches.json', 'w', encoding='utf-8') as f:
            json.dump(results_for_json, f, ensure_ascii=False, indent=4)
    
        print(f" SUKCES! Zapisano {len(results_for_json)} meczy do pliku docs/matches.json")

if __name__ == "__main__":
    main()



    