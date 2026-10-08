import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timezone
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss
from utils import check_form, calculate_stats

FEATURES = [
    'HomeGaz', 'AwayGaz',
    'HomeScoredAvg', 'HomeLosedAvg', 'HomeShotsAvg', 'HomeSTargetAvg', 'HomeCornersAvg',
    'AwayScoredAvg', 'AwayLosedAvg', 'AwayShotsAvg', 'AwaySTargetAvg', 'AwayCornersAvg',
    'EffDiff', 'HomeClean_sheets', 'AwayClean_sheets', 'GD_Diff', 'MarketDiff'
]
TARGET_MAP = {'H': 2, 'A': 0, 'D': 1}
HOLDOUT_SIZE = 300

def make_model():
    # Parametry dobrane na walidacji czasowej (log-loss): więcej płytkich drzew z małym learning rate
    # daje lepiej skalibrowane prawdopodobieństwa niż poprzednie 50 drzew (zbyt ściśnięte wokół średniej)
    return XGBClassifier(n_estimators=200, learning_rate=0.01, max_depth=2, min_child_weight=10,
                         subsample=0.8, random_state=42)

def market_diff(odds_h, odds_a):
    # Brak kursów (0 / NaN) -> NaN, tak samo jak w danych treningowych
    try:
        odds_h, odds_a = float(odds_h), float(odds_a)
    except (TypeError, ValueError):
        return np.nan
    if np.isnan(odds_h) or np.isnan(odds_a) or odds_h <= 0 or odds_a <= 0:
        return np.nan
    return 1 / odds_h - 1 / odds_a

def evaluate_model(train_df, features):
    """Prosta walidacja czasowa: trening na starszych meczach, test na ostatnich HOLDOUT_SIZE."""
    try:
        data = train_df.sort_values('Date')
        if len(data) <= HOLDOUT_SIZE * 2:
            print("Za mało danych do ewaluacji modelu.")
            return

        train, test = data.iloc[:-HOLDOUT_SIZE], data.iloc[-HOLDOUT_SIZE:]
        y_tr = train['FTR'].map(TARGET_MAP)
        y_te = test['FTR'].map(TARGET_MAP)

        model = make_model()
        model.fit(train[features], y_tr)
        probs = model.predict_proba(test[features])

        print(f"--- Ewaluacja (ostatnie {len(test)} meczów, od {test['Date'].min().date()} do {test['Date'].max().date()}) ---")
        print(f"Model:      accuracy = {accuracy_score(y_te, probs.argmax(axis=1)):.3f}, log-loss = {log_loss(y_te, probs, labels=[0, 1, 2]):.4f}")

        # Baseline: prawdopodobieństwa z kursów B365 (znormalizowane o marżę bukmachera)
        odds = test[['B365A', 'B365D', 'B365H']].apply(pd.to_numeric, errors='coerce')
        mask = (odds > 0).all(axis=1)
        if mask.sum() > 0:
            implied = 1 / odds[mask].values
            implied = implied / implied.sum(axis=1, keepdims=True)
            y_b = y_te[mask]
            print(f"Bukmacher:  accuracy = {accuracy_score(y_b, implied.argmax(axis=1)):.3f}, log-loss = {log_loss(y_b, implied, labels=[0, 1, 2]):.4f} (na {mask.sum()} meczach)")

        freq = y_tr.value_counts(normalize=True).reindex([0, 1, 2]).fillna(0).values
        print(f"Częstości:  log-loss = {log_loss(y_te, np.tile(freq, (len(y_te), 1)), labels=[0, 1, 2]):.4f} (zawsze ten sam rozkład H/D/A)")
    except Exception as e:
        print(f"Błąd podczas ewaluacji modelu: {e}")

def main():
    try:
        df = pd.read_csv('processed_data.csv')
        df['Date'] = pd.to_datetime(df['Date'])
    except FileNotFoundError:
        print("BŁĄD: Brak pliku processed_data.csv! Uruchom najpierw build_database.py")
        sys.exit(1)

    try:
        fixtures = pd.read_csv('weekend_fixtures.csv').to_dict('records')
    except FileNotFoundError:
        print("BŁĄD: Brak pliku weekend_fixtures.csv! Uruchom najpierw scraper.py")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        fixtures = []

    if not fixtures:
        print("Brak meczów w terminarzu (np. przerwa reprezentacyjna) - zapisuję pustą listę.")

    results_for_json = []

    train_df = df[df['FTR'].isin(TARGET_MAP.keys())].copy()
    available_features = [f for f in FEATURES if f in df.columns]

    evaluate_model(train_df, available_features)

    # Model końcowy trenowany na wszystkich danych
    model = make_model()
    model.fit(train_df[available_features], train_df['FTR'].map(TARGET_MAP))

    # Model zapasowy bez MarketDiff - dla meczów bez kursów (w treningu kursy są zawsze,
    # więc NaN w MarketDiff byłby traktowany przez XGBoost jak silny faworyt gospodarzy)
    no_odds_features = [f for f in available_features if f != 'MarketDiff']
    model_no_odds = make_model()
    model_no_odds.fit(train_df[no_odds_features], train_df['FTR'].map(TARGET_MAP))

    today = pd.Timestamp.now().normalize()

    for match in fixtures:
        home = match['Home']
        away = match['Away']

        # Statystyki liczone tylko z meczów przed datą spotkania
        match_date = pd.to_datetime(match.get('Date'), errors='coerce')
        if pd.isna(match_date):
            match_date = today + pd.Timedelta(days=1)

        h_stats = calculate_stats(home, df, match_date)
        a_stats = calculate_stats(away, df, match_date)
        h_gaz = check_form(home, df, match_date)
        a_gaz = check_form(away, df, match_date)

        if h_gaz == 0 and h_stats['scored'] == 0 and h_stats['shots'] == 0:
            print(f"  Uwaga: brak historii dla {home} - statystyki zerowe.")
        if a_gaz == 0 and a_stats['scored'] == 0 and a_stats['shots'] == 0:
            print(f"  Uwaga: brak historii dla {away} - statystyki zerowe.")

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
            'MarketDiff': market_diff(match.get('OddsH'), match.get('OddsA'))
        }])

        input_row = input_row[available_features].astype(float)

        if 'MarketDiff' in input_row and input_row['MarketDiff'].isna().any():
            probs = model_no_odds.predict_proba(input_row[no_odds_features])[0]
        else:
            probs = model.predict_proba(input_row)[0]

        p_home = float(probs[2] * 100)
        p_draw = float(probs[1] * 100)
        p_away = float(probs[0] * 100)

        rec = "Ryzykowny mecz"
        match_class = "neutral"

        if p_home > 55:
            rec = "Wygrana Gospodarzy"
            match_class = "win"
        elif p_away > 55:
            rec = "Wygrana Gości"
            match_class = "lose"
        elif (p_home + p_draw) > 75:
            rec = "Gospodarz lub Remis"
            match_class = "safe"
        elif (p_away + p_draw) > 75:
            rec = "Goście lub Remis"
            match_class = "safe"

        results_for_json.append({
            "home": home,
            "away": away,
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

    # Znacznik czasu - zapisywany przy każdym uruchomieniu (commit utrzymuje aktywność repozytorium)
    with open('docs/last_update.json', 'w', encoding='utf-8') as f:
        json.dump({"updated": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}, f)

if __name__ == "__main__":
    main()
