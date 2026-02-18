# ⚽ LaLiga XGBoost Soccer Prediction Engine

Automatyczny system przewidywania wyników meczów **La Liga** oparty na modelu **XGBoost**. Projekt codziennie pobiera dane, oblicza statystyki drużyn i generuje prognozy na nadchodzące mecze weekendowe, publikując wyniki na stronie GitHub Pages.

---

## 📋 Spis treści

- [Jak to działa](#-jak-to-działa)
- [Architektura projektu](#-architektura-projektu)
- [Pipeline danych](#-pipeline-danych)
- [Model predykcyjny](#-model-predykcyjny)
- [Cechy (features) modelu](#-cechy-features-modelu)
- [Strona internetowa](#-strona-internetowa)
- [Automatyzacja (CI/CD)](#-automatyzacja-cicd)
- [Wykorzystane biblioteki](#-wykorzystane-biblioteki)
- [Instalacja i uruchomienie](#-instalacja-i-uruchomienie)
- [Struktura plików](#-struktura-plików)
- [Licencja](#-licencja)

---

## 🔍 Jak to działa

1. **Scraping** — system pobiera listę najbliższych meczów La Liga ze strony ESPN
2. **Budowa bazy danych** — historyczne dane meczowe (6 sezonów) są pobierane z football-data.co.uk i wzbogacane o obliczone statystyki
3. **Predykcja** — model XGBoost trenowany na danych historycznych generuje prawdopodobieństwa wyniku (wygrana gospodarzy / remis / wygrana gości)
4. **Publikacja** — wyniki zapisywane są do pliku JSON i wyświetlane na statycznej stronie HTML (GitHub Pages)

Cały proces jest **w pełni zautomatyzowany** dzięki GitHub Actions i uruchamia się codziennie o 5:00 UTC.

---

## 🏗 Architektura projektu

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  ESPN.com    │────▶│   scraper.py     │────▶│ weekend_fixtures │     │             │
│  (fixtures)  │     │  (cloudscraper)  │     │     .csv         │     │             │
└─────────────┘     └──────────────────┘     └────────┬─────────┘     │             │
                                                      │               │  update_    │
┌─────────────┐     ┌──────────────────┐     ┌────────▼─────────┐     │  website.py │
│ football-   │────▶│ build_database.py│────▶│ processed_data   │────▶│             │
│ data.co.uk  │     │  (6 sezonów)     │     │     .csv         │     │  (XGBoost)  │
└─────────────┘     └──────────────────┘     └──────────────────┘     └──────┬──────┘
                                                                             │
                                                                    ┌────────▼────────┐
                                                                    │ docs/matches.json│
                                                                    │ docs/index.html  │
                                                                    │  (GitHub Pages)  │
                                                                    └─────────────────┘
```

---

## 📊 Pipeline danych

### 1. Scraping meczów — [`scraper.py`](scraper.py)

- Pobiera nadchodzące mecze La Liga z ESPN (`espn.com/soccer/fixtures`)
- Wykorzystuje `cloudscraper` do omijania zabezpieczeń Cloudflare
- Mapuje nazwy drużyn na format zgodny z bazą danych (np. `"Atlético Madrid"` → `"Ath Madrid"`) za pomocą [`map_name()`](scraper.py)
- Stosuje fuzzy matching (`difflib.get_close_matches`) gdy dokładne mapowanie nie istnieje
- Zapisuje wynik do [`weekend_fixtures.csv`](weekend_fixtures.csv)

### 2. Budowa bazy danych — [`build_database.py`](build_database.py)

- Pobiera historyczne dane meczowe z **6 sezonów** La Liga (2019/20 — 2024/25) za pomocą funkcji [`download_data()`](build_database.py)
- Źródło danych: [football-data.co.uk](https://www.football-data.co.uk/)
- Funkcja [`process_history()`](build_database.py) wzbogaca każdy mecz o obliczone cechy statystyczne (forma, skuteczność, kursy bukmacherskie itd.)
- Wynik zapisywany jest do [`processed_data.csv`](processed_data.csv) (~2000 meczów)

### 3. Funkcje pomocnicze — [`utils.py`](utils.py)

#### [`check_form()`](utils.py)
Oblicza formę drużyny na podstawie **ostatnich 5 meczów**:
- Wygrana u siebie: **3 pkt**
- Remis u siebie: **1 pkt**
- Wygrana na wyjeździe: **3.5 pkt** (bonus za trudniejsze warunki)
- Remis na wyjeździe: **1.5 pkt**

#### [`calculate_stats()`](utils.py)
Oblicza średnie statystyki z ostatnich 5 meczów:
- Strzelone / stracone bramki
- Strzały / strzały celne
- Rzuty rożne
- Skuteczność (efficiency)
- Czyste konta (clean sheets)

---

## 🤖 Model predykcyjny

| Parametr | Wartość |
|---|---|
| Algorytm | **XGBClassifier** |
| Liczba drzew (`n_estimators`) | 50 |
| Learning rate | 0.01 |
| Maksymalna głębokość (`max_depth`) | 3 |
| Subsampling | 0.8 |
| Klasy | `H` (wygrana gospodarzy), `D` (remis), `A` (wygrana gości) |
| Skuteczność testowa | ~53% |

Model trenowany jest na pełnym zbiorze historycznym przy każdym uruchomieniu pipeline'u.

### Logika rekomendacji

Funkcja w [`update_website.py`](update_website.py) generuje typy na podstawie prawdopodobieństw:

| Warunek | Rekomendacja | Kolor |
|---|---|---|
| `p_home > 50%` | **1 (Gospodarz)** | 🟢 Zielony |
| `p_away > 45%` | **2 (Gość)** | 🔴 Czerwony |
| `p_home > 40%` | **1X (Bezpiecznie)** | 🟡 Żółty |
| Inne | **No Bet** | ⚪ Neutralny |

Jeśli kursy bukmacherskie nie są dostępne (syntetyczne), rekomendacja oznaczana jest gwiazdką `*`.

---

## 📐 Cechy (features) modelu

Model wykorzystuje **16 cech** obliczanych dla każdego meczu:

| Cecha | Opis |
|---|---|
| `HomeGaz` / `AwayGaz` | Forma drużyny (ostatnie 5 meczów) |
| `HomeScoredAvg` / `AwayScoredAvg` | Średnia strzelonych bramek |
| `HomeLosedAvg` / `AwayLosedAvg` | Średnia straconych bramek |
| `HomeShotsAvg` / `AwayShotsAvg` | Średnia strzałów |
| `HomeSTargetAvg` / `AwaySTargetAvg` | Średnia strzałów celnych |
| `HomeCornersAvg` / `AwayCornersAvg` | Średnia rzutów rożnych |
| `HomeClean_sheets` / `AwayClean_sheets` | Czyste konta w ostatnich 5 meczach |
| `EffDiff` | Różnica skuteczności drużyn |
| `GD_Diff` | Różnica bilansu bramkowego |
| `MarketDiff` | Różnica implikowanych prawdopodobieństw z kursów bukmacherskich (B365) |

---

## 🌐 Strona internetowa

Statyczna strona HTML hostowana na **GitHub Pages**:

- Plik: [`docs/index.html`](docs/index.html)
- Dane: [`docs/matches.json`](docs/matches.json)
- Design: ciemny motyw, responsywny layout z kartami meczów
- Kolorowe oznaczenia typów (zielony/żółty/czerwony)
- Automatyczne ładowanie danych z JSON przez `fetch()`

---

## ⚙ Automatyzacja (CI/CD)

Plik [`update_predictions.yml`](.github/workflows/update_predictions.yml) definiuje workflow GitHub Actions:

```
Codziennie o 5:00 UTC (cron: '0 5 * * *')
       │
       ├── 1. python scraper.py        → Pobranie listy meczów
       ├── 2. python build_database.py → Aktualizacja bazy danych
       ├── 3. python update_website.py → Generowanie prognoz
       └── 4. git commit & push        → Publikacja wyników
```

Workflow można też uruchomić ręcznie (`workflow_dispatch`).

---

## 📦 Wykorzystane biblioteki

| Biblioteka | Zastosowanie |
|---|---|
| **pandas** | Operacje na danych tabelarycznych, CSV |
| **xgboost** | Model klasyfikacyjny XGBClassifier |
| **scikit-learn** | Preprocessing danych, metryki |
| **cloudscraper** | Web scraping z omijaniem Cloudflare |
| **beautifulsoup4** | Parsowanie HTML |
| **requests** | Pobieranie danych z football-data.co.uk |
| **lxml** / **html5lib** | Parsery HTML dla `pd.read_html()` |
| **numpy** | Operacje numeryczne |

---

## 🚀 Instalacja i uruchomienie

### Wymagania
- Python 3.9+

### Instalacja

```bash
git clone https://github.com/<twoj-username>/laliga-xgboost-soccer-engine.git
cd laliga-xgboost-soccer-engine
pip install -r requirements.txt
```

### Uruchomienie krok po kroku

```bash
# 1. Pobranie nadchodzących meczów z ESPN
python scraper.py

# 2. Budowa / aktualizacja bazy danych historycznych
python build_database.py

# 3. Generowanie prognoz i aktualizacja strony
python update_website.py
```

### Podgląd strony lokalnie

Otwórz [`docs/index.html`](docs/index.html) w przeglądarce lub uruchom lokalny serwer:

```bash
cd docs
python -m http.server 8000
```

Następnie otwórz `http://localhost:8000` w przeglądarce.

---

## 📁 Struktura plików

```
├── scraper.py              # Scraping meczów z ESPN
├── build_database.py       # Pobieranie i przetwarzanie danych historycznych
├── update_website.py       # Trening modelu i generowanie prognoz
├── utils.py                # Funkcje pomocnicze (forma, statystyki)
├── processed_data.csv      # Przetworzona baza meczów (~2000 rekordów)
├── weekend_fixtures.csv    # Nadchodzące mecze
├── requirements.txt        # Zależności Pythona
├── LICENSE                 # Licencja MIT
├── docs/
│   ├── index.html          # Strona z prognozami (GitHub Pages)
│   └── matches.json        # Dane prognoz w formacie JSON
└── .github/
    └── workflows/
        └── update_predictions.yml  # Automatyzacja GitHub Actions
```

---

## 📄 Licencja

Projekt udostępniony na licencji [MIT](LICENSE).

**Autor:** Kacper Olenkiewicz © 2026

---

> ⚠️ **Disclaimer:** Prognozy generowane przez model mają charakter wyłącznie informacyjny i edukacyjny. Nie stanowią porady bukmacherskiej. Skuteczność modelu wynosi ~53%, co oznacza, że błędne predykcje są częste.