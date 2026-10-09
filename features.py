"""Cechy meczów liczone sekwencyjnie (wyłącznie z meczów rozegranych wcześniej).

Wspólne dla build_database.py (historia) i update_website.py (nadchodzące mecze):
- Elo (ważone różnicą bramek, regresja do średniej między sezonami),
- wykładnicze średnie kroczące (EWM) statystyk drużyn,
- model Dixona-Colesa (Poisson atak/obrona, przeliczany co tydzień),
- prawdopodobieństwa rynkowe z kursów przed meczem (metoda potęgowa usuwania marży).
Stan drużyn aktualizowany jest dopiero po całym dniu meczowym, więc mecze z tego samego dnia się nie widzą.
"""
import copy
import csv
import io
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

# Kolumny z plików football-data.co.uk (tylko kursy przed meczem - bez kursów zamknięcia)
KEEP = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC',
        'B365H', 'B365D', 'B365A', 'AvgH', 'AvgD', 'AvgA', 'BbAvH', 'BbAvD', 'BbAvA']

# Elo: K=10, przewaga własnego boiska 60 pkt, 10% regresji do średniej między sezonami,
# beniaminek = średnia spadkowiczów + 50 pkt
ELO_K, ELO_HFA, ELO_REGRESS, ELO_PROMO_OFFSET = 10, 60, 0.1, 50
HALF_LIVES = (8, 20)                       # okresy półtrwania EWM (w meczach)
STATS = ['gf', 'ga', 'sf', 'sa', 'tf', 'ta', 'cf', 'ca', 'pts']   # gole, strzały, celne, rożne, punkty (za/przeciw)
# Dixon-Coles: waga meczu exp(-xi * dni), okno 1100 dni, ridge 2, prior beniaminka (atak -0.15 / obrona +0.15)
DC_XI, DC_WINDOW, DC_RIDGE, DC_PROMO_ATT, DC_PROMO_DEF = 0.0019, 1100, 2.0, -0.15, 0.15


def read_season(text, code):
    """Wczytuje CSV sezonu. Pliki football-data mają wiersze o różnej długości (psują zwykłe read_csv)."""
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return pd.DataFrame(columns=KEEP + ['Season'])
    hdr = [h.strip() for h in rows[0]]
    n = len(hdr)
    data = [r[:n] + [''] * (n - len(r)) for r in rows[1:] if r and any(x.strip() for x in r)]
    df = pd.DataFrame(data, columns=hdr)
    df = df.loc[:, ~df.columns.duplicated()]
    out = pd.DataFrame({c: df[c] if c in df else np.nan for c in KEEP})
    for c in KEEP[3:]:
        if c != 'FTR':
            out[c] = pd.to_numeric(out[c], errors='coerce')
    out['HomeTeam'] = out['HomeTeam'].astype(str).str.strip()
    out['AwayTeam'] = out['AwayTeam'].astype(str).str.strip()
    out['Date'] = pd.to_datetime(out['Date'], dayfirst=True, errors='coerce', format='mixed')
    out = out.dropna(subset=['Date', 'FTHG', 'FTAG'])
    out = out[(out['HomeTeam'] != '') & (out['AwayTeam'] != '')]
    # FTR z goli (na wypadek braków w kolumnie)
    out['FTR'] = np.where(out['FTHG'] > out['FTAG'], 'H', np.where(out['FTHG'] < out['FTAG'], 'A', 'D'))
    out['Season'] = code
    return out


def season_code(date):
    """Kod sezonu football-data dla daty, np. 2026-10-09 -> '2627' (sezon zaczyna się w lipcu)."""
    y = date.year if date.month >= 7 else date.year - 1
    return f"{y % 100:02d}{(y + 1) % 100:02d}"


# ---------------------------------------------------------------- kursy
def clean_odds(x):
    x = pd.to_numeric(pd.Series(np.atleast_1d(x)), errors='coerce').values.astype(float)
    return np.where(x > 1, x, np.nan)


def pick_odds(df):
    """Kursy przed meczem jako komplet H/D/A z jednego źródła: B365, potem średnia rynkowa (Avg, dawniej BbAv)."""
    out = np.full((len(df), 3), np.nan)
    for src in ('BbAv', 'Avg', 'B365'):          # od najmniej preferowanego - kolejne nadpisują
        o = np.column_stack([clean_odds(df[src + s]) for s in 'HDA'])
        ok = np.isfinite(o).all(1)
        out[ok] = o[ok]
    df['OddsH'], df['OddsD'], df['OddsA'] = out[:, 0], out[:, 1], out[:, 2]
    return df


def implied_basic(oh, od, oa):
    p = 1 / np.column_stack([oh, od, oa])
    return p / p.sum(1, keepdims=True)


def implied_power(oh, od, oa):
    """Metoda potęgowa: p_i = (1/o_i)^k, k dobrane tak, by suma = 1 (częściowo usuwa efekt faworyt-outsider)."""
    inv = 1 / np.column_stack([oh, od, oa])
    lo, hi = np.full(len(inv), 0.5), np.full(len(inv), 3.0)
    for _ in range(60):
        k = (lo + hi) / 2
        s = (inv ** k[:, None]).sum(1)
        lo, hi = np.where(s > 1, k, lo), np.where(s > 1, hi, k)
    p = inv ** k[:, None]
    return p / p.sum(1, keepdims=True)


def market_features(oh, od, oa):
    """Cechy rynkowe z kursów (NaN, gdy brak kursów). Kolejność prawdopodobieństw: H, D, A."""
    oh, od, oa = clean_odds(oh), clean_odds(od), clean_odds(oa)
    pp, pb = implied_power(oh, od, oa), implied_basic(oh, od, oa)
    return pd.DataFrame({
        'MktH': pp[:, 0], 'MktD': pp[:, 1], 'MktA': pp[:, 2],          # metoda potęgowa
        'BasicH': pb[:, 0], 'BasicD': pb[:, 1], 'BasicA': pb[:, 2],    # zwykła normalizacja 1/kurs
        'Overround': 1 / oh + 1 / od + 1 / oa - 1,
        'LogInvH': np.log(1 / oh), 'LogInvA': np.log(1 / oa),          # tylko kursy H/A
    })


# ---------------------------------------------------------------- Dixon-Coles
def fit_poisson(hi, ai, gh, ga, w, nteam, prior_att, prior_def):
    """Ważony model Poissona: log mu_h = c + home + att_h + def_a, log mu_a = c + att_a + def_h."""
    def f(x):
        c, hm = x[0], x[1]
        att, de = x[2:2 + nteam], x[2 + nteam:]
        lh, la = c + hm + att[hi] + de[ai], c + att[ai] + de[hi]
        mh, ma = np.exp(lh), np.exp(la)
        nll = np.sum(w * (mh - gh * lh)) + np.sum(w * (ma - ga * la))
        nll += DC_RIDGE * (np.sum((att - prior_att) ** 2) + np.sum((de - prior_def) ** 2))
        rh, ra = w * (mh - gh), w * (ma - ga)
        g = np.zeros_like(x)
        g[0], g[1] = rh.sum() + ra.sum(), rh.sum()
        g[2:2 + nteam] = np.bincount(hi, rh, nteam) + np.bincount(ai, ra, nteam) + 2 * DC_RIDGE * (att - prior_att)
        g[2 + nteam:] = np.bincount(ai, rh, nteam) + np.bincount(hi, ra, nteam) + 2 * DC_RIDGE * (de - prior_def)
        return nll, g
    x0 = np.zeros(2 + 2 * nteam)
    x0[0], x0[1] = 0.1, 0.25
    return minimize(f, x0, jac=True, method='L-BFGS-B').x


def dc_tau(x, y, lh, la, rho):
    t = np.ones_like(lh)
    t = np.where((x == 0) & (y == 0), 1 - lh * la * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lh * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + la * rho, t)
    return np.where((x == 1) & (y == 1), 1 - rho, t)


def score_probs(lh, la, rho, maxg=10):
    """P(H), P(D), P(A) z macierzy wyników Poissona z poprawką Dixona-Colesa dla niskich wyników."""
    g = np.arange(maxg + 1)
    M = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    M[0, 0] *= 1 - lh * la * rho
    M[0, 1] *= 1 + lh * rho
    M[1, 0] *= 1 + la * rho
    M[1, 1] *= 1 - rho
    M /= M.sum()
    p_h, p_d = np.tril(M, -1).sum(), np.trace(M)
    return p_h, p_d, 1 - p_h - p_d


# ---------------------------------------------------------------- stan drużyn
class TeamState:
    def __init__(self):
        self.R = {}                 # Elo
        self.ew = {}                # (drużyna, półtrwanie) -> {statystyka: wartość}
        self.acc = {}               # bieżący sezon: drużyna -> [mecze, punkty, gole za, gole przeciw]
        self.season = None
        self.season_teams, self.prev_teams, self.promoted = set(), set(), set()
        self.promo_elo, self.promo_ew = 1450.0, {}
        self.dc = None
        self.last_date = None

    def new_season(self, code):
        prev = self.season_teams
        if prev:
            mean = np.mean([self.R[t] for t in prev])
            # Spadkowicze = 3 ostatnie drużyny poprzedniego sezonu; beniaminkowie startują od ich średniej
            table = sorted(prev, key=lambda t: (self.acc.get(t, [0, 0, 0, 0])[1],
                                                self.acc.get(t, [0, 0, 0, 0])[2] - self.acc.get(t, [0, 0, 0, 0])[3]))
            relegated = table[:3]
            self.promo_elo = np.mean([self.R[t] for t in relegated]) + ELO_PROMO_OFFSET
            self.promo_ew = {}
            for hl in HALF_LIVES:
                states = [self.ew[(t, hl)] for t in relegated if (t, hl) in self.ew]
                if states:
                    vals = {k: [s[k] for s in states if np.isfinite(s[k])] for k in STATS}
                    self.promo_ew[hl] = {k: np.mean(v) if v else np.nan for k, v in vals.items()}
            for t in prev:
                self.R[t] = mean + (self.R[t] - mean) * (1 - ELO_REGRESS)
        self.prev_teams, self.season_teams, self.promoted, self.acc = prev, set(), set(), {}
        self.season = code

    def ensure(self, t):
        """Dołącza drużynę do bieżącego sezonu; drużyna spoza poprzedniego sezonu (beniaminek, nowa nazwa)
        dostaje wartości startowe beniaminka zamiast zer."""
        if t in self.season_teams:
            return
        self.season_teams.add(t)
        if t in self.prev_teams:
            return
        if self.prev_teams:
            self.promoted.add(t)
        elif t in self.R:
            return
        self.R[t] = self.promo_elo
        for hl in HALF_LIVES:
            if hl in self.promo_ew:
                self.ew[(t, hl)] = dict(self.promo_ew[hl])
            else:
                self.ew.pop((t, hl), None)

    def knows(self, t):
        return t in self.season_teams or t in self.prev_teams

    def refit_dc(self, date, hist):
        """Dopasowanie modelu Dixona-Colesa do meczów z okna DC_WINDOW dni przed datą (hist posortowane po Date)."""
        dates = hist['Date'].values
        lo = np.searchsorted(dates, np.datetime64(date - pd.Timedelta(days=DC_WINDOW)))
        hi_ = np.searchsorted(dates, np.datetime64(date))
        past = hist.iloc[lo:hi_]
        if len(past) <= 200:
            return
        tl = sorted(set(past['HomeTeam']) | set(past['AwayTeam']) | self.season_teams)
        ti = {t: i for i, t in enumerate(tl)}
        hi, ai = past['HomeTeam'].map(ti).values, past['AwayTeam'].map(ti).values
        gh, ga = past['FTHG'].values.astype(float), past['FTAG'].values.astype(float)
        w = np.exp(-DC_XI * (date - past['Date']).dt.days.values)
        prior_att = np.array([DC_PROMO_ATT if t in self.promoted else 0.0 for t in tl])
        prior_def = np.array([DC_PROMO_DEF if t in self.promoted else 0.0 for t in tl])
        n = len(tl)
        x = fit_poisson(hi, ai, gh, ga, w, n, prior_att, prior_def)
        c, hm, att, de = x[0], x[1], x[2:2 + n], x[2 + n:]
        lh, la = np.exp(c + hm + att[hi] + de[ai]), np.exp(c + att[ai] + de[hi])
        rho = minimize_scalar(lambda r: -np.sum(w * np.log(np.clip(dc_tau(gh, ga, lh, la, r), 1e-9, None))),
                              bounds=(-0.3, 0.3), method='bounded').x
        self.dc = (ti, c, hm, att, de, rho)

    def match_features(self, h, a):
        self.ensure(h)
        self.ensure(a)
        f = {'EloDiff': self.R[h] - self.R[a]}
        for hl in HALF_LIVES:
            eh, ea = self.ew.get((h, hl)), self.ew.get((a, hl))
            for st in STATS:
                f[f'd{hl}_{st}'] = eh[st] - ea[st] if (eh and ea) else np.nan
        ah, aa = self.acc.get(h), self.acc.get(a)
        f['ppg_diff'] = (ah[1] / ah[0] if ah else np.nan) - (aa[1] / aa[0] if aa else np.nan)
        f['promoH'], f['promoA'] = float(h in self.promoted), float(a in self.promoted)
        if self.dc is not None:
            ti, c, hm, att, de, rho = self.dc
            att_h, def_h = (att[ti[h]], de[ti[h]]) if h in ti else (DC_PROMO_ATT, DC_PROMO_DEF)
            att_a, def_a = (att[ti[a]], de[ti[a]]) if a in ti else (DC_PROMO_ATT, DC_PROMO_DEF)
            lh, la = np.exp(c + hm + att_h + def_a), np.exp(c + att_a + def_h)
            p_h, p_d, p_a = score_probs(lh, la, rho)
            f.update(DC_tot=lh + la, lDC_HD=np.log(p_h / p_d), lDC_AD=np.log(p_a / p_d))
        else:
            f.update(DC_tot=np.nan, lDC_HD=np.nan, lDC_AD=np.nan)
        return f

    def update(self, r):
        h, a, gh, ga = r.HomeTeam, r.AwayTeam, r.FTHG, r.FTAG
        # Elo z mnożnikiem za różnicę bramek
        e = 1 / (1 + 10 ** (-(self.R[h] + ELO_HFA - self.R[a]) / 400))
        s = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
        g = abs(gh - ga)
        mult = 1.0 if g <= 1 else (1.5 if g == 2 else (11 + g) / 8)
        self.R[h] += ELO_K * mult * (s - e)
        self.R[a] -= ELO_K * mult * (s - e)
        p_h = 3.0 if gh > ga else (1.0 if gh == ga else 0.0)
        p_a = 3.0 if ga > gh else (1.0 if gh == ga else 0.0)
        obs = {h: dict(gf=gh, ga=ga, sf=r.HS, sa=r.AS, tf=r.HST, ta=r.AST, cf=r.HC, ca=r.AC, pts=p_h),
               a: dict(gf=ga, ga=gh, sf=r.AS, sa=r.HS, tf=r.AST, ta=r.HST, cf=r.AC, ca=r.HC, pts=p_a)}
        for t, o in obs.items():
            for hl in HALF_LIVES:
                al = 1 - 0.5 ** (1 / hl)
                cur = self.ew.setdefault((t, hl), {k: np.nan for k in STATS})
                for k in STATS:
                    if np.isfinite(o[k]):
                        cur[k] = o[k] if not np.isfinite(cur[k]) else (1 - al) * cur[k] + al * o[k]
            acc = self.acc.setdefault(t, [0, 0.0, 0.0, 0.0])
            acc[0] += 1
            acc[1] += o['pts']
            acc[2] += o['gf']
            acc[3] += o['ga']

    def fixture_features(self, h, a, date):
        """Cechy nadchodzącego meczu na podstawie wszystkich wyników do dziś (stan bazowy się nie zmienia)."""
        st = copy.deepcopy(self)
        code = season_code(date)
        if st.season is None or code > st.season:
            st.new_season(code)     # pierwszy mecz nowego sezonu, którego wyników jeszcze nie ma w bazie
        return st.match_features(h, a)


def build_features(df, today=None):
    """Jedno przejście po historii. Zwraca (df z cechami przed meczem, stan drużyn na dziś)."""
    df = df.sort_values(['Date', 'HomeTeam'], kind='mergesort').reset_index(drop=True)
    st = TeamState()
    feats = [None] * len(df)
    last_week = None
    for date, day in df.groupby('Date', sort=True):
        if day['Season'].iloc[0] != st.season:
            st.new_season(day['Season'].iloc[0])
        for r in day.itertuples():          # drużyny dnia znane przed dopasowaniem DC
            st.ensure(r.HomeTeam)
            st.ensure(r.AwayTeam)
        week = date.to_period('W')
        if week != last_week:
            last_week = week
            st.refit_dc(date, df)
        for r in day.itertuples():
            feats[r.Index] = st.match_features(r.HomeTeam, r.AwayTeam)
        for r in day.itertuples():
            st.update(r)
        st.last_date = date
    # Model DC na dziś - do predykcji nadchodzących meczów
    today = today or pd.Timestamp.now().normalize()
    st.refit_dc(max(today, st.last_date + pd.Timedelta(days=1)), df)
    out = pd.concat([df, pd.DataFrame(feats, index=df.index)], axis=1)
    out = pd.concat([out, market_features(out['OddsH'], out['OddsD'], out['OddsA'])], axis=1)
    return out, st
