# ⚽ LaLiga XGBoost Soccer Prediction Engine

Automatyczny system przewidywania wyników meczów **La Liga** oparty na modelu **XGBoost**. Projekt codziennie pobiera dane, oblicza cechy drużyn (Elo, model Dixona-Colesa, średnie kroczące) i generuje prognozy na nadchodzące mecze weekendowe, publikując wyniki na stronie GitHub Pages.
**[Strona z wynikami](https://kacper-olenkiewicz.github.io/laliga-xgboost-soccer-engine/)**
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

1. **Pobranie terminarza** — system pobiera listę najbliższych meczów La Liga wraz z kursami bukmacherskimi z football-data.co.uk (`fixtures.csv`), a gdy tam ich brak — z ESPN
2. **Budowa bazy danych** — historyczne dane meczowe (wszystkie sezony od 2000/01 do bieżącego) są pobierane z football-data.co.uk i wzbogacane o cechy liczone w jednym przejściu po historii
3. **Predykcja** — model XGBoost koryguje prawdopodobieństwa wynikające z kursów bukmacherskich (najpierw oceniany metodą walk-forward na ok. 2 ostatnich sezonach, potem trenowany na całej historii) i generuje prawdopodobieństwa wyniku (wygrana gospodarzy / remis / wygrana gości)
4. **Publikacja** — wyniki zapisywane są do pliku JSON i wyświetlane na statycznej stronie HTML (GitHub Pages)

Cały proces jest **w pełni zautomatyzowany** dzięki GitHub Actions i uruchamia się codziennie o 5:00 UTC.

---

## 🏗 Architektura projektu

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌─────────────┐
│ fixtures.csv │────▶│   scraper.py     │────▶│ weekend_fixtures │     │             │
│ (ESPN zapas) │     │                  │     │     .csv         │     │             │
└─────────────┘     └──────────────────┘     └────────┬─────────┘     │             │
                                                      │               │  update_    │
┌─────────────┐     ┌──────────────────┐     ┌────────▼─────────┐     │  website.py │
│ football-   │────▶│ build_database.py│────▶│ processed_data   │────▶│             │
│ data.co.uk  │     │  + features.py   │     │ + team_state.pkl │     │  (XGBoost)  │
└─────────────┘     └──────────────────┘     └──────────────────┘     └──────┬──────┘
                                                                             │
                                                                    ┌────────▼────────┐
                                                                    │ docs/matches.json│
                                                                    │ docs/last_update │
                                                                    │ docs/index.html  │
                                                                    │  (GitHub Pages)  │
                                                                    └─────────────────┘
```

---

## 📊 Pipeline danych

### 1. Pobranie meczów — [`scraper.py`](scraper.py)

Źródła sprawdzane po kolei (następne używane tylko, gdy poprzednie nic nie zwróci):

1. **football-data.co.uk** — [`fixtures.csv`](https://www.football-data.co.uk/fixtures.csv), wiersze z `Div == 'SP1'` i datą od dziś. Nazwy drużyn są identyczne jak w danych historycznych, a kursy H/D/A pochodzą z kolumn `B365H`/`B365D`/`B365A` (gdy niepełne — ze średnich `AvgH`/`AvgD`/`AvgA`)
2. **API ESPN** (`site.api.espn.com`, mecze z najbliższych 7 dni) — kursy moneyline H/D/A (DraftKings) przeliczane na kursy dziesiętne
3. **Strona ESPN** (`espn.com/soccer/fixtures`, przez `cloudscraper`) — bez kursów

- Dla ESPN nazwy drużyn są mapowane na format bazy danych (np. `"Atlético Madrid"` → `"Ath Madrid"`, `"Espanyol"` → `"Espanol"`) za pomocą [`map_name()`](scraper.py), z fuzzy matchingiem (`difflib.get_close_matches`) jako ostatnią deską ratunku
- Zapisuje wynik do `weekend_fixtures.csv` (kolumny `Date, Home, Away, OddsH, OddsD, OddsA`; brak kursów = `0`)

### 2. Budowa bazy danych — [`build_database.py`](build_database.py)

- Pobiera historyczne dane meczowe La Liga **od sezonu 2000/01 do bieżącego** za pomocą funkcji [`download_data()`](build_database.py). Lista sezonów liczona jest dynamicznie z bieżącej daty (sezon zaczyna się w lipcu). Brakujące (404) lub puste pliki są pomijane, a wiersze o różnej długości w plikach CSV są poprawnie wczytywane
- Źródło danych: [football-data.co.uk](https://www.football-data.co.uk/)
- Pierwsze sezony służą tylko do „rozgrzania” Elo i modelu Dixona-Colesa — model trenowany jest od sezonu 2005/06
- Cechy liczone są w **jednym przejściu po historii** (funkcja `build_features()` w [`features.py`](features.py)); stan drużyn aktualizowany jest dopiero po całym dniu meczowym, więc cechy pochodzą wyłącznie z meczów rozegranych **przed** danym spotkaniem (brak wycieku danych), a mecze z tego samego dnia się nie widzą
- Wynik zapisywany jest do `processed_data.csv` (~10 000 meczów, z kolumną `Season`), a stan drużyn na dziś (Elo, średnie, parametry Dixona-Colesa) do `team_state.pkl` — z niego liczone są cechy nadchodzących meczów

### 3. Cechy — [`features.py`](features.py)

Moduł wspólny dla `build_database.py` i `update_website.py`:
- **Elo** ważone różnicą bramek (K=10, przewaga własnego boiska 60 pkt, 10% regresji do średniej między sezonami)
- **Dixon-Coles** — ważony model Poissona (atak/obrona drużyn), przeliczany co tydzień na meczach z ostatnich 1100 dni (starsze mecze mają mniejszą wagę)
- **Średnie wykładnicze (EWM)** statystyk drużyn (gole, strzały, strzały celne, rożne, punkty — za i przeciw)
- **Beniaminkowie** i drużyny bez historii dostają wartości startowe (średnia spadkowiczów), a nie zera
- **Kursy przed meczem** (B365, zapasowo średnia rynkowa; nigdy kursy zamknięcia) zamieniane na prawdopodobieństwa **metodą potęgową**: `p_i = (1/kurs_i)^k`, gdzie `k` dobrane tak, by suma wynosiła 1 — usuwa marżę bukmachera lepiej niż zwykła normalizacja

---

## 🤖 Model predykcyjny

Prognoza dla każdego meczu wybierana jest według tego, jakie kursy są dostępne:

| Poziom | Dane wejściowe | Model |
|---|---|---|
| 1 | Kursy H/D/A | **XGBClassifier** startujący od prawdopodobieństw rynkowych (`base_margin = log p_rynek`) i korygujący je cechami Elo / Dixon-Coles / EWM |
| 2 | Tylko kursy H/A | Regresja logistyczna na `log(1/kurs)` |
| 3 | Brak kursów | Regresja logistyczna na Elo, Dixonie-Colesie i średnich EWM |

| Parametr XGBoost | Wartość |
|---|---|
| Liczba drzew (`n_estimators`) | 100 |
| Learning rate | 0.01 |
| Maksymalna głębokość (`max_depth`) | 2 |
| Min. waga liścia (`min_child_weight`) | 10 |
| Subsampling | 0.8 |
| Klasy | `H` (wygrana gospodarzy), `D` (remis), `A` (wygrana gości) |

Przy każdym uruchomieniu [`update_website.py`](update_website.py) najpierw wykonuje **ewaluację walk-forward** na ostatnich 760 meczach (ok. 2 sezony): co miesiąc modele trenowane są na wszystkich wcześniejszych meczach i oceniane na meczach z kolejnego miesiąca. Wypisywane są log-loss, RPS i accuracy modelu w porównaniu z kursami B365 (zwykła normalizacja i metoda potęgowa) oraz 95% przedział ufności (bootstrap po tygodniach) dla różnicy log-loss. Następnie modele końcowe trenowane są na pełnym zbiorze historycznym.

Przykładowy wynik (październik 2026):

```
Model (XGBoost + rynek)        log-loss = 0.9567, RPS = 0.1937, accuracy = 0.547
Tylko kursy H/A (poziom 2)     log-loss = 0.9614, RPS = 0.1946, accuracy = 0.546
Bez kursów (poziom 3)          log-loss = 0.9667, RPS = 0.1971, accuracy = 0.529
Bukmacher (normalizacja)       log-loss = 0.9607, RPS = 0.1946, accuracy = 0.543
Bukmacher (metoda potęgowa)    log-loss = 0.9583, RPS = 0.1941, accuracy = 0.543
Częstości H/D/A                log-loss = 1.0595, RPS = 0.2270, accuracy = 0.466
```

### Logika rekomendacji

Funkcja w [`update_website.py`](update_website.py) generuje typy na podstawie prawdopodobieństw:

| Warunek | Rekomendacja | Kolor |
|---|---|---|
| `p_home > 55%` | **Wygrana Gospodarzy** | 🟢 Zielony |
| `p_away > 55%` | **Wygrana Gości** | 🔴 Czerwony |
| `p_home + p_draw > 75%` | **Gospodarz lub Remis** | 🟡 Żółty |
| `p_away + p_draw > 75%` | **Goście lub Remis** | 🟡 Żółty |
| Inne | **Ryzykowny mecz** | ⚪ Neutralny |

---

## 📐 Cechy (features) modelu

| Cecha | Opis | Poziom |
|---|---|---|
| `MktH` / `MktD` / `MktA` | Prawdopodobieństwa z kursów H/D/A (metoda potęgowa) — punkt startowy XGBoost | 1 |
| `Overround` | Marża bukmachera | 1 |
| `EloDiff` | Różnica rankingów Elo | 1, 3 |
| `lDC_HD` / `lDC_AD` | Log-ilorazy szans z modelu Dixona-Colesa (gospodarz/remis, goście/remis) | 1, 3 |
| `DC_tot` | Oczekiwana liczba goli w meczu (Dixon-Coles) | 1, 3 |
| `d8_*` / `d20_*` | Różnice średnich EWM (półtrwanie 8 / 20 meczów): gole, strzały, celne, rożne, punkty | 1 / 3 |
| `ppg_diff` | Różnica punktów na mecz w bieżącym sezonie | 1 |
| `promoH` / `promoA` | Czy drużyna jest beniaminkiem | 1 |
| `LogInvH` / `LogInvA` | `log(1/kurs)` gospodarzy i gości | 2 |

---

## 🌐 Strona internetowa

Statyczna strona HTML hostowana na **GitHub Pages**:

- Plik: [`docs/index.html`](docs/index.html)
- Dane: [`docs/matches.json`](docs/matches.json)
- Czas ostatniej aktualizacji: `docs/last_update.json` (`{"updated": "<czas UTC>"}`), wyświetlany w nagłówku strony
- Design: jasny motyw (Bootstrap 5), responsywny layout z kartami meczów
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

Workflow używa Pythona 3.12 z cache'em pip. Można go też uruchomić ręcznie (`workflow_dispatch`).

GitHub wyłącza zaplanowane workflowy po 60 dniach braku aktywności w repozytorium. Dlatego przy każdym uruchomieniu zapisywany i commitowany jest też `docs/last_update.json` — commit powstaje codziennie, nawet gdy prognozy się nie zmieniły.

---

## 📦 Wykorzystane biblioteki

| Biblioteka | Zastosowanie |
|---|---|
| **pandas** | Operacje na danych tabelarycznych, CSV |
| **xgboost** | Model klasyfikacyjny XGBClassifier |
| **scikit-learn** | Regresja logistyczna (poziomy 2 i 3), standaryzacja cech |
| **scipy** | Dopasowanie modelu Dixona-Colesa, rozkład Poissona |
| **cloudscraper** | Zapasowy scraping strony ESPN z omijaniem Cloudflare |
| **requests** | Pobieranie danych z football-data.co.uk i API ESPN |
| **lxml** | Parser HTML dla `pd.read_html()` |
| **numpy** | Operacje numeryczne |

---

## 🚀 Instalacja i uruchomienie

### Wymagania
- Python 3.12 (zalecany)

### Instalacja

```bash
git clone https://github.com/kacper-olenkiewicz/laliga-xgboost-soccer-engine.git
cd laliga-xgboost-soccer-engine
pip install -r requirements.txt
```

### Uruchomienie krok po kroku

```bash
# 1. Pobranie nadchodzących meczów (football-data.co.uk, zapasowo ESPN)
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
├── scraper.py              # Pobieranie nadchodzących meczów i kursów
├── build_database.py       # Pobieranie i przetwarzanie danych historycznych
├── update_website.py       # Trening modelu i generowanie prognoz
├── features.py             # Cechy: Elo, Dixon-Coles, EWM, kursy (wspólne dla bazy i predykcji)
├── processed_data.csv      # Przetworzona baza meczów (generowana, ~10 000 rekordów, poza repo - .gitignore)
├── team_state.pkl          # Stan drużyn na dziś (generowany, poza repo - .gitignore)
├── weekend_fixtures.csv    # Nadchodzące mecze (generowane, poza repo - .gitignore)
├── requirements.txt        # Zależności Pythona
├── LICENSE                 # Licencja MIT
├── docs/
│   ├── index.html          # Strona z prognozami (GitHub Pages)
│   ├── matches.json        # Dane prognoz w formacie JSON
│   └── last_update.json    # Czas ostatniej aktualizacji
└── .github/
    └── workflows/
        └── update_predictions.yml  # Automatyzacja GitHub Actions
```

---

## 📄 Licencja

Projekt udostępniony na licencji [MIT](LICENSE).

**Autor:** Kacper Olenkiewicz © 2026

---

> ⚠️ **Disclaimer:** Prognozy generowane przez model mają charakter wyłącznie informacyjny i edukacyjny. Nie stanowią porady bukmacherskiej. Skuteczność modelu wynosi ~54%, co oznacza, że błędne predykcje są częste.