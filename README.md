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
| `CIP_02_modelling.ipynb` | RQ1 model performance, RQ2 county-level PPV, RQ3 subgroup performance *(in progress)* |
| `visualizer.py` | Streamlit triage dashboard *(in progress)* |
| `data/processed/` | Analytic sample (`nhanes_analytic.csv`, 4,613 × 15), data dictionary, county prevalence, acquisition metadata |
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

## Contributions

| Member | Contribution |
|---|---|
| Michele Mogos | Notebook 01: data acquisition, cleaning, EDA, analytic sample |
| Divya Shori | Notebook 02: modelling and analysis for RQ1–RQ3; reproducibility fixes to notebook 01 |
| Esteban Peña Zhiña | `visualizer.py`: Streamlit dashboard |
