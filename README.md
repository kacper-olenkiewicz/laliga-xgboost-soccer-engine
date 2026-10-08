# ⚽ LaLiga XGBoost Soccer Prediction Engine

Automatyczny system przewidywania wyników meczów **La Liga** oparty na modelu **XGBoost**. Projekt codziennie pobiera dane, oblicza statystyki drużyn i generuje prognozy na nadchodzące mecze weekendowe, publikując wyniki na stronie GitHub Pages.
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
2. **Budowa bazy danych** — historyczne dane meczowe (7 ostatnich sezonów, aż do bieżącego) są pobierane z football-data.co.uk i wzbogacane o obliczone statystyki
3. **Predykcja** — model XGBoost (najpierw oceniany na ostatnich 300 meczach, potem trenowany na całej historii) generuje prawdopodobieństwa wyniku (wygrana gospodarzy / remis / wygrana gości)
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
│ data.co.uk  │     │  (7 sezonów)     │     │     .csv         │     │  (XGBoost)  │
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

1. **football-data.co.uk** — [`fixtures.csv`](https://www.football-data.co.uk/fixtures.csv), wiersze z `Div == 'SP1'` i datą od dziś. Nazwy drużyn są identyczne jak w danych historycznych, a kursy pochodzą z kolumn `B365H`/`B365A` (lub średnich `AvgH`/`AvgA`)
2. **API ESPN** (`site.api.espn.com`, mecze z najbliższych 7 dni) — kursy moneyline (DraftKings) przeliczane na kursy dziesiętne
3. **Strona ESPN** (`espn.com/soccer/fixtures`, przez `cloudscraper`) — bez kursów

- Dla ESPN nazwy drużyn są mapowane na format bazy danych (np. `"Atlético Madrid"` → `"Ath Madrid"`, `"Espanyol"` → `"Espanol"`) za pomocą [`map_name()`](scraper.py), z fuzzy matchingiem (`difflib.get_close_matches`) jako ostatnią deską ratunku
- Zapisuje wynik do `weekend_fixtures.csv` (kolumny `Date, Home, Away, OddsH, OddsA`; brak kursów = `0`)

### 2. Budowa bazy danych — [`build_database.py`](build_database.py)

- Pobiera historyczne dane meczowe z **7 ostatnich sezonów** La Liga za pomocą funkcji [`download_data()`](build_database.py). Lista sezonów liczona jest dynamicznie z bieżącej daty (sezon zaczyna się w lipcu, np. w październiku 2026 bieżący to 2026/27 → sezony 2020/21 — 2026/27). Brakujące (404) lub puste pliki są pomijane
- Źródło danych: [football-data.co.uk](https://www.football-data.co.uk/)
- Funkcja [`process_history()`](build_database.py) wzbogaca każdy mecz o obliczone cechy statystyczne (forma, skuteczność, kursy bukmacherskie itd.)
- Cechy liczone są wyłącznie z meczów rozegranych **przed** datą danego spotkania (brak wycieku danych)
- Gdy brak kursów B365, `MarketDiff` = `NaN` (XGBoost obsługuje braki natywnie)
- Wynik zapisywany jest do `processed_data.csv` (~2300 meczów, z kolumną `Season`)

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
| Liczba drzew (`n_estimators`) | 200 |
| Learning rate | 0.01 |
| Maksymalna głębokość (`max_depth`) | 2 |
| Min. waga liścia (`min_child_weight`) | 10 |
| Subsampling | 0.8 |
| Klasy | `H` (wygrana gospodarzy), `D` (remis), `A` (wygrana gości) |
| Skuteczność testowa | ~53% (bukmacher: ~53%) |

Przy każdym uruchomieniu [`update_website.py`](update_website.py) najpierw wykonuje **ewaluację czasową**: model trenowany jest na wszystkich meczach poza ostatnimi 300, a na tych 300 wypisywane są accuracy i log-loss modelu w porównaniu z prawdopodobieństwami z kursów B365 (znormalizowanymi o marżę) oraz z rozkładem częstości H/D/A. Następnie model końcowy trenowany jest na pełnym zbiorze historycznym. Parametry modelu zostały dobrane na tej walidacji (najniższy log-loss przy możliwie prostym modelu: płytkie drzewa, mały learning rate) i są wpisane na stałe w `make_model()`.

Przykładowy wynik (październik 2026):

```
Model:      accuracy = 0.533, log-loss = 0.9740
Bukmacher:  accuracy = 0.530, log-loss = 0.9724 (na 300 meczach)
Częstości:  log-loss = 1.0497 (zawsze ten sam rozkład H/D/A)
```

`MarketDiff` dla nadchodzących meczów liczony jest z prawdziwych kursów (`OddsH`/`OddsA`); jeśli ich brak, predykcję wykonuje osobny model trenowany bez cechy `MarketDiff` (dane treningowe zawsze mają kursy, więc `NaN` byłby przez XGBoost traktowany jak silny faworyt gospodarzy).

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

Model wykorzystuje **17 cech** obliczanych dla każdego meczu:

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
| **scikit-learn** | Metryki ewaluacji (accuracy, log-loss) |
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
├── utils.py                # Funkcje pomocnicze (forma, statystyki)
├── processed_data.csv      # Przetworzona baza meczów (generowana, ~2300 rekordów, poza repo - .gitignore)
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

> ⚠️ **Disclaimer:** Prognozy generowane przez model mają charakter wyłącznie informacyjny i edukacyjny. Nie stanowią porady bukmacherskiej. Skuteczność modelu wynosi ~53%, co oznacza, że błędne predykcje są częste.