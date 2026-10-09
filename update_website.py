import pandas as pd
import numpy as np
import json
import os
import sys
import pickle
from datetime import datetime, timezone
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from features import market_features

TARGET_MAP = {'H': 2, 'A': 0, 'D': 1}      # kolumny prawdopodobieństw: A, D, H
TRAIN_FROM = '0506'                         # wcześniejsze sezony tylko "rozgrzewają" Elo / Dixona-Colesa
STATE_FILE = 'team_state.pkl'
EVAL_MATCHES = 760                          # ewaluacja na ok. 2 ostatnich sezonach

ELO_DC = ['EloDiff', 'lDC_HD', 'lDC_AD', 'DC_tot']
EWM_STATS = ['gf', 'ga', 'tf', 'ta', 'pts', 'sf', 'sa', 'cf']
# Poziom 1 (kursy H/D/A): XGBoost poprawia prawdopodobieństwa rynkowe (base_margin = log p_rynek)
FEATURES_T1 = ELO_DC + [f'd8_{s}' for s in EWM_STATS] + ['ppg_diff', 'promoH', 'promoA', 'Overround']
MARKET = ['MktA', 'MktD', 'MktH']
# Poziom 2 (tylko kursy H/A): regresja logistyczna na log(1/kurs)
FEATURES_T2 = ['LogInvH', 'LogInvA']
# Poziom 3 (bez kursów): regresja logistyczna na Elo, Dixonie-Colesie i średnich EWM
FEATURES_T3 = ELO_DC + [f'd20_{s}' for s in EWM_STATS]

def make_model():
    # Płytkie drzewa i mały learning rate: model tylko lekko koryguje rynek (dobrane na walidacji walk-forward)
    return XGBClassifier(n_estimators=100, learning_rate=0.01, max_depth=2, min_child_weight=10,
                         subsample=0.8, random_state=42)

def make_lr(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))

def fit_models(train):
    y = train['FTR'].map(TARGET_MAP)
    t1 = train.dropna(subset=MARKET)
    t2 = train.dropna(subset=FEATURES_T2)
    t3 = train.dropna(subset=FEATURES_T3)
    return {
        'xgb': make_model().fit(t1[FEATURES_T1], y[t1.index], base_margin=np.log(t1[MARKET].values)),
        'lr2': make_lr(1.0).fit(t2[FEATURES_T2].values, y[t2.index]),
        'lr3': make_lr(0.1).fit(t3[FEATURES_T3].values, y[t3.index]),
        'means3': t3[FEATURES_T3].mean(),
    }

def predict_tier(models, X, tier):
    if tier == 1:
        return models['xgb'].predict_proba(X[FEATURES_T1], base_margin=np.log(X[MARKET].values))
    if tier == 2:
        return models['lr2'].predict_proba(X[FEATURES_T2].values)
    return models['lr3'].predict_proba(X[FEATURES_T3].fillna(models['means3']).values)

def predict(models, X):
    """Prawdopodobieństwa (A, D, H) i poziom modelu: 1 = kursy H/D/A, 2 = tylko kursy H/A, 3 = bez kursów."""
    X = X.reset_index(drop=True)
    tiers = np.where(X[MARKET].notna().all(axis=1), 1, np.where(X[FEATURES_T2].notna().all(axis=1), 2, 3))
    probs = np.zeros((len(X), 3))
    for t in (1, 2, 3):
        m = tiers == t
        if m.any():
            probs[m] = predict_tier(models, X[m], t)
    return probs, tiers

def match_metrics(P, y):
    P = np.clip(P, 1e-12, 1)
    ll = -np.log(P[np.arange(len(y)), y])
    cum_p, cum_o = np.cumsum(P, 1), np.cumsum(np.eye(3)[y], 1)
    rps = ((cum_p[:, :2] - cum_o[:, :2]) ** 2).sum(1) / 2
    return ll, rps, (P.argmax(1) == y)

def bootstrap_diff(d, groups, n=1000):
    """95% przedział dla średniej różnicy, bootstrap blokowy po tygodniach (mecze z kolejki są zależne)."""
    _, inv = np.unique(groups, return_inverse=True)
    sums, counts = np.bincount(inv, d), np.bincount(inv)
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(sums), (n, len(sums)))
    boot = sums[draws].sum(1) / counts[draws].sum(1)
    return np.percentile(boot, [2.5, 97.5])

def evaluate_model(train_df):
    """Walk-forward: ostatnie EVAL_MATCHES meczów, modele trenowane co miesiąc na wszystkich wcześniejszych meczach."""
    try:
        data = train_df.sort_values('Date').reset_index(drop=True)
        if len(data) <= EVAL_MATCHES * 3:
            print("Za mało danych do ewaluacji modelu.")
            return

        test = np.arange(len(data) - EVAL_MATCHES, len(data))
        months = data['Date'].dt.to_period('M').values
        P = {k: np.full((len(data), 3), np.nan) for k in ('model', 't2', 't3')}
        for month in pd.unique(months[test]):
            rows = test[months[test] == month]
            models = fit_models(data[data['Date'] < data.loc[rows, 'Date'].min()])
            X = data.loc[rows]
            P['model'][rows] = predict(models, X)[0]
            ok = X[FEATURES_T2].notna().all(axis=1).values     # regresja logistyczna nie przyjmuje NaN (mecz bez kursów)
            if ok.any():
                P['t2'][rows[ok]] = predict_tier(models, X[ok], 2)
            P['t3'][rows] = predict_tier(models, X, 3)

        d = data.loc[test]
        d = d[d[MARKET].notna().all(axis=1)]
        rows, y = d.index.values, d['FTR'].map(TARGET_MAP).values
        freq = data.loc[:test[0] - 1, 'FTR'].map(TARGET_MAP).value_counts(normalize=True).reindex([0, 1, 2]).values
        candidates = {
            'Model (XGBoost + rynek)': P['model'][rows],
            'Tylko kursy H/A (poziom 2)': P['t2'][rows],
            'Bez kursów (poziom 3)': P['t3'][rows],
            'Bukmacher (normalizacja)': d[['BasicA', 'BasicD', 'BasicH']].values,
            'Bukmacher (metoda potęgowa)': d[MARKET].values,
            'Częstości H/D/A': np.tile(freq, (len(d), 1)),
        }
        print(f"--- Ewaluacja walk-forward ({len(d)} meczów od {d['Date'].min().date()} do {d['Date'].max().date()}, "
              f"trening co miesiąc) ---")
        ll = {}
        for name, probs in candidates.items():
            ll[name], rps, acc = match_metrics(probs, y)
            print(f"{name:30s} log-loss = {ll[name].mean():.4f}, RPS = {rps.mean():.4f}, accuracy = {acc.mean():.3f}")

        weeks = d['Date'].dt.to_period('W').astype(str).values
        for base in ('Bukmacher (normalizacja)', 'Bukmacher (metoda potęgowa)'):
            diff = ll['Model (XGBoost + rynek)'] - ll[base]
            lo, hi = bootstrap_diff(diff, weeks)
            print(f"Różnica log-loss model - {base}: {diff.mean():+.4f} (95% CI {lo:+.4f} .. {hi:+.4f}; ujemna = model lepszy)")
    except Exception as e:
        print(f"Błąd podczas ewaluacji modelu: {e}")

def main():
    try:
        df = pd.read_csv('processed_data.csv', dtype={'Season': str})
        df['Date'] = pd.to_datetime(df['Date'])
        with open(STATE_FILE, 'rb') as f:
            state = pickle.load(f)
    except FileNotFoundError as e:
        print(f"BŁĄD: Brak pliku {e.filename}! Uruchom najpierw build_database.py")
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

    train_df = df[df['FTR'].isin(TARGET_MAP.keys()) & (df['Season'] >= TRAIN_FROM)].copy()

    evaluate_model(train_df)

    # Modele końcowe trenowane na wszystkich danych
    models = fit_models(train_df)

    today = pd.Timestamp.now().normalize()
    tier_names = {1: "kursy H/D/A + XGBoost", 2: "tylko kursy H/A", 3: "bez kursów"}

    for match in fixtures:
        home = str(match['Home']).strip()
        away = str(match['Away']).strip()

        match_date = pd.to_datetime(match.get('Date'), errors='coerce')
        if pd.isna(match_date):
            match_date = today + pd.Timedelta(days=1)

        for team in (home, away):
            if not state.knows(team):
                print(f"  Uwaga: brak historii dla {team} w ostatnich sezonach - wartości startowe beniaminka.")

        # Cechy z wyników do dziś + kursy (brak = 0 / NaN; stary format CSV bez OddsD też działa)
        row = state.fixture_features(home, away, match_date)
        row.update(market_features(match.get('OddsH'), match.get('OddsD', 0), match.get('OddsA')).iloc[0].to_dict())
        probs, tiers = predict(models, pd.DataFrame([row]))
        probs = probs[0]
        print(f"  {home} - {away}: model {tier_names[tiers[0]]}")

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
