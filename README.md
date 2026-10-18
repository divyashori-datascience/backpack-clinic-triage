# Project_CIP02
The Backpack Clinic — Field-based dysglycemia triage
HSLU MSc Applied Data Science & AI · CIP02 (HS 2026)

Can measurements a field health worker takes from a backpack (age, sex, BMI,
waist, blood pressure) flag adults who should get a lab A1C test — and how much
should the doctor trust that flag in their county and for this patient?

## Repository structure

| Path | Content |
|---|---|
| `CIP_01_acquisition_cleaning_eda.ipynb` | Acquisition (pynhanes, codebook scraping, Selenium search, CDC PLACES API), cleaning, EDA, frozen analytic sample |
| `CIP_02_modelling.ipynb` | RQ1 model performance, RQ2 county-level PPV, RQ3 subgroup performance |
| `visualizer.py` | Streamlit triage dashboard *(in progress)* |
| `data/processed/` | Analytic sample (`nhanes_analytic.csv`, 4,613 × 15), data dictionary, county prevalence, acquisition metadata, result tables (`rq1_*`, `rq2_*`, `rq3_*`) |
| `models/` | Fitted triage model (`triage_model.joblib`) for the dashboard |
| `data/raw/` | Raw PLACES JSON and rendered Selenium HTML (NHANES `.xpt` files are re-downloaded by notebook 01) |
| `figures/` | Figures produced by the notebooks |

## How to run

```bash
pip install -r requirements.txt
jupyter notebook CIP_01_acquisition_cleaning_eda.ipynb   # optional: data/processed is already committed
jupyter notebook CIP_02_modelling.ipynb
streamlit run visualizer.py
```

Selenium needs Google Chrome installed; the driver is downloaded automatically.

## Data sources

- **NHANES 2021–2023** (CDC) — DEMO, BMX, BPXO, GHB, DIQ files via `pynhanes`
- **NHANES codebook pages** — scraped with requests/BeautifulSoup; variable search driven with Selenium
- **CDC PLACES** county estimates — Socrata REST API (`swc5-untb`)

## Key results

| RQ | Answer |
|---|---|
| RQ1 | Logistic regression on age, sex, waist and blood pressure: cross-validated AUC 0.725, well calibrated; random forest and gradient boosting are no better. At a 26% risk threshold it catches 80% of cases. |
| RQ2 | With the same model, the chance that a positive result is correct ranges from 18% to 73% across 2,957 US counties (median 42%). |
| RQ3 | Sensitivity is 0.91 for White adults but 0.57–0.71 for Asian, Mexican American, Black and Other Hispanic adults; the gap persists after adjusting for age. |

## Course methods used

| Course block | Methods | Where |
|---|---|---|
| SW01 Scraping | `robots.txt` check, requests + BeautifulSoup (`find`, `find_all`), Selenium (`webdriver.Chrome`, `By.ID`, `send_keys`, `WebDriverWait`), error handling | Notebook 01, §3.3 |
| SW02 API & pandas | REST API with `requests.get().json()` → `pd.DataFrame`, pagination; `read_csv`, `loc`/`iloc`, `merge`, `groupby().agg()`, `apply`/`map` | Notebook 01 §3.6–3.7; notebook 02 |
| SW02 Data cleaning | missing values, datatypes, expected ranges, outliers, duplicates, formatting, enrichment | Notebook 01, §4 |
| SW03 Storage & frontends | CSV/JSON storage of raw and processed data; Streamlit dashboard | `data/`, `visualizer.py` |
| SW04 Git & visualisation | git repository with `.gitignore`; matplotlib, seaborn, pandas `.plot`, plotly | whole repo; `figures/` |

## Contributions

| Member | Contribution |
|---|---|
| Michele Mogos | Notebook 01: data acquisition, cleaning, EDA, analytic sample |
| Divya Shori | Notebook 02: modelling and analysis for RQ1–RQ3; reproducibility fixes to notebook 01 |
| Esteban Peña Zhiña | `visualizer.py`: Streamlit dashboard |
