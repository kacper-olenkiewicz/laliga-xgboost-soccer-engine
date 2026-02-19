import cloudscraper
import pandas as pd
from io import StringIO
from difflib import get_close_matches

URL = "https://www.espn.com/soccer/fixtures/_/league/esp.1"

MY_TEAMS = [
    "Real Madrid", "Barcelona", "Girona", "Ath Madrid", "Ath Bilbao", 
    "Betis", "Sociedad", "Valencia", "Las Palmas", "Getafe", 
    "Osasuna", "Alaves", "Villarreal", "Rayo Vallecano", "Sevilla", 
    "Mallorca", "Celta", "Espanyol", "Valladolid", "Leganes",
    "Cadiz", "Granada", "Almeria", "Elche", "Levante", "Oviedo"
]

NAME_MAPPING = {
    "Atlético Madrid": "Ath Madrid", "Athletic Club": "Ath Bilbao",
    "Real Betis": "Betis", "Rayo Vallecano": "Rayo Vallecano",
    "Celta Vigo": "Celta", "Real Sociedad": "Sociedad",
    "Alavés": "Alaves", "Real Valladolid": "Valladolid",
    "Espanyol": "Espanyol"
}

def map_name(name):
    clean = name.split(' vs ')[0].strip()
    
    if clean in NAME_MAPPING: return NAME_MAPPING[clean]
    if clean in MY_TEAMS: return clean
    
    matches = get_close_matches(clean, MY_TEAMS, n=1, cutoff=0.5)
    return matches[0] if matches else None

def scrape():
    
    
    
    scraper = cloudscraper.create_scraper(
        browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True}
    )
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9,pl;q=0.8',
        'Referer': 'https://www.google.com/'
    }
    
    fixtures = []
    
    try:
        
        response = scraper.get(URL, headers=headers, timeout=15)
        
        if response.status_code != 200:
            print(f"Błąd połączenia: {response.status_code}")
        else:
            

            try:
                
                dfs = pd.read_html(StringIO(response.text))
                
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
                                fixtures.append({"Home": h, "Away": a, "OddsH": 0, "OddsA": 0})
                        except:
                            continue
            except ValueError:
                print("Nie znaleziono żadnych tabel na stronie (ESPN mogło ukryć mecze lub włączyć blokadę bota).")

    except Exception as e:
        print(f" Błąd krytyczny: {e}")


    if fixtures:
        df_fix = pd.DataFrame(fixtures).drop_duplicates()
        print(f"Sukces! Zapisano {len(df_fix)} meczów.")
    else:
        print("⚠️ Brak meczów na liście. Tworzę awaryjny, pusty plik CSV.")
        df_fix = pd.DataFrame(columns=["Home", "Away", "OddsH", "OddsA"])


    df_fix.to_csv('weekend_fixtures.csv', index=False)


if __name__ == "__main__":
    scrape()