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
    
    scraper = cloudscraper.create_scraper()
    
    try:
        response = scraper.get(URL)
        
        if response.status_code != 200:
            print(f"Błąd połączenia: {response.status_code}")
            return

        print("Połączono! Szukam tabel z meczami...")


        dfs = pd.read_html(StringIO(response.text))
        
        fixtures = []
        

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

        if fixtures:
            df_fix = pd.DataFrame(fixtures).drop_duplicates()
            df_fix.to_csv('weekend_fixtures.csv', index=False)

        else:
            print(" Połączono z ESPN, ale parser nie rozpoznał meczów w tabelach.")
            print("   Może to oznaczać brak meczów w najbliższych dniach.")

    except Exception as e:
        print(f"Błąd: {e}")

if __name__ == "__main__":
    scrape()