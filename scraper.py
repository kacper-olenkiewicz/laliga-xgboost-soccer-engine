import cloudscraper
import requests
import unicodedata
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta
from difflib import get_close_matches

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
ESPN_API_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/esp.1/scoreboard"
URL = "https://www.espn.com/soccer/fixtures/_/league/esp.1"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9,pl;q=0.8',
    'Referer': 'https://www.google.com/'
}

COLUMNS = ["Date", "Home", "Away", "OddsH", "OddsD", "OddsA"]

# Nazwy drużyn dokładnie tak, jak w danych z football-data.co.uk
MY_TEAMS = [
    "Real Madrid", "Barcelona", "Girona", "Ath Madrid", "Ath Bilbao",
    "Betis", "Sociedad", "Valencia", "Las Palmas", "Getafe",
    "Osasuna", "Alaves", "Villarreal", "Vallecano", "Sevilla",
    "Mallorca", "Celta", "Espanol", "Valladolid", "Leganes",
    "Cadiz", "Granada", "Almeria", "Elche", "Levante", "Oviedo",
    "Malaga", "Santander", "La Coruna", "Eibar", "Huesca"
]

# Nazwy z ESPN (bez polskich/hiszpańskich znaków) -> nazwy z football-data.co.uk
NAME_MAPPING = {
    "Atletico Madrid": "Ath Madrid", "Athletic Club": "Ath Bilbao",
    "Real Betis": "Betis", "Rayo Vallecano": "Vallecano",
    "Celta Vigo": "Celta", "Real Sociedad": "Sociedad",
    "Alaves": "Alaves", "Real Valladolid": "Valladolid",
    "Espanyol": "Espanol", "Real Oviedo": "Oviedo",
    "Racing Santander": "Santander", "Deportivo": "La Coruna",
    "Deportivo La Coruna": "La Coruna", "Real Mallorca": "Mallorca",
    "Mallorca": "Mallorca", "Malaga": "Malaga", "Leganes": "Leganes",
    "Cadiz": "Cadiz", "Almeria": "Almeria", "UD Las Palmas": "Las Palmas"
}

def strip_accents(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))

def map_name(name):
    clean = strip_accents(str(name).split(' vs ')[0].strip())
    if clean.startswith("v "): clean = clean[2:].strip()

    if clean in NAME_MAPPING: return NAME_MAPPING[clean]
    if clean in MY_TEAMS: return clean

    matches = get_close_matches(clean, MY_TEAMS, n=1, cutoff=0.5)
    return matches[0] if matches else None

def to_float(value):
    try:
        value = float(value)
        return value if value > 1 else 0
    except Exception:
        return 0

def american_to_decimal(odds):
    # Kursy amerykańskie (np. "+295", "-120") -> kursy dziesiętne
    try:
        odds = float(str(odds).replace("+", ""))
        if odds > 0: return round(1 + odds / 100, 2)
        if odds < 0: return round(1 + 100 / abs(odds), 2)
    except Exception:
        pass
    return 0

def scrape_football_data(div='SP1'):
    """Główne źródło: fixtures.csv z football-data.co.uk (nazwy drużyn zgodne z bazą + kursy)."""
    fixtures = []
    try:
        r = requests.get(FIXTURES_URL, headers=HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.content.decode('utf-8-sig', errors='replace')))
        df.columns = [c.strip() for c in df.columns]

        df = df[df['Div'] == div].copy()
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df[df['Date'] >= pd.Timestamp.now().normalize()]

        for _, row in df.iterrows():
            # Kursy brane jako komplet H/D/A z jednego źródła: B365, a gdy niepełne - średnia rynkowa
            odds = (0, 0, 0)
            for src in ('B365', 'Avg'):
                triple = tuple(to_float(row.get(src + s)) for s in 'HDA')
                if all(triple):
                    odds = triple
                    break
            odds_h, odds_d, odds_a = odds

            print(f"  > Znaleziono: {row['HomeTeam']} vs {row['AwayTeam']} ({odds_h} / {odds_d} / {odds_a})")
            fixtures.append({"Date": row['Date'].strftime('%Y-%m-%d'), "Home": row['HomeTeam'],
                             "Away": row['AwayTeam'], "OddsH": odds_h, "OddsD": odds_d, "OddsA": odds_a})
    except Exception as e:
        print(f"Błąd przy pobieraniu {FIXTURES_URL}: {e}")
    return fixtures

def scrape_espn_api(days=7):
    """Zapasowe źródło nr 1: API ESPN (mecze na najbliższe dni + kursy DraftKings)."""
    fixtures = []
    today = datetime.now()
    for d in range(days):
        day = (today + timedelta(days=d)).strftime('%Y%m%d')
        try:
            r = requests.get(ESPN_API_URL, params={'dates': day}, headers=HEADERS, timeout=15)
            r.raise_for_status()
            events = r.json().get('events', [])
        except Exception as e:
            print(f"Błąd API ESPN ({day}): {e}")
            continue

        for event in events:
            try:
                if event['status']['type']['state'] != 'pre': continue
                comp = event['competitions'][0]
                teams = {c['homeAway']: c['team']['displayName'] for c in comp['competitors']}
                h = map_name(teams['home'])
                a = map_name(teams['away'])
                if not h or not a or h == a: continue

                odds_h, odds_d, odds_a = 0, 0, 0
                if comp.get('odds'):
                    odds = comp['odds'][0]
                    ml = odds.get('moneyline', {})
                    odds_h = american_to_decimal(ml.get('home', {}).get('close', {}).get('odds'))
                    odds_a = american_to_decimal(ml.get('away', {}).get('close', {}).get('odds'))
                    odds_d = american_to_decimal(ml.get('draw', {}).get('close', {}).get('odds')
                                                 or odds.get('drawOdds', {}).get('moneyLine'))

                print(f"  > Znaleziono: {h} vs {a} ({odds_h} / {odds_d} / {odds_a})")
                fixtures.append({"Date": event['date'][:10], "Home": h, "Away": a,
                                 "OddsH": odds_h, "OddsD": odds_d, "OddsA": odds_a})
            except Exception:
                continue
    return fixtures

def scrape_espn_html():
    """Zapasowe źródło nr 2: tabela meczów na stronie ESPN (bez kursów)."""
    scraper = cloudscraper.create_scraper(
        browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True}
    )
    fixtures = []

    try:
        response = scraper.get(URL, headers=HEADERS, timeout=15)

        if response.status_code != 200:
            print(f"Błąd połączenia: {response.status_code}")
        else:
            try:
                dfs = pd.read_html(StringIO(response.content.decode('utf-8', errors='replace')))

                for df in dfs:
                    if df.shape[1] < 2: continue

                    for index, row in df.iterrows():
                        try:
                            raw_home = str(row.iloc[0])
                            raw_away = str(row.iloc[1])

                            if "match" in str(row.iloc[1]).lower() or ":" in str(row.iloc[1]):
                                raw_away = str(row.iloc[2])

                            h = map_name(raw_home)
                            a = map_name(raw_away)

                            if h and a and h != a:
                                print(f"  > Znaleziono: {h} vs {a}")
                                fixtures.append({"Date": "", "Home": h, "Away": a, "OddsH": 0, "OddsD": 0, "OddsA": 0})
                        except Exception:
                            continue
            except ValueError:
                print("Nie znaleziono żadnych tabel na stronie (ESPN mogło ukryć mecze lub włączyć blokadę bota).")

    except Exception as e:
        print(f" Błąd krytyczny: {e}")
    return fixtures

def scrape():
    print("Pobieram mecze z football-data.co.uk (fixtures.csv)...")
    fixtures = scrape_football_data()

    if not fixtures:
        print("Brak meczów La Liga w fixtures.csv. Próbuję API ESPN...")
        fixtures = scrape_espn_api()

    if not fixtures:
        print("API ESPN nie zwróciło meczów. Próbuję strony ESPN...")
        fixtures = scrape_espn_html()

    if fixtures:
        df_fix = pd.DataFrame(fixtures, columns=COLUMNS).drop_duplicates(subset=["Home", "Away"])
        print(f"Sukces! Zapisano {len(df_fix)} meczów.")
    else:
        print(" Brak meczów na liście. Tworzę awaryjny, pusty plik CSV.")
        df_fix = pd.DataFrame(columns=COLUMNS)

    df_fix.to_csv('weekend_fixtures.csv', index=False)


if __name__ == "__main__":
    scrape()
