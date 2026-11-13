# NYC Green Taxi Demand Forecasting

**Machine Learning Portfolio | Time-Series Model Evaluation**

This portfolio investigates whether historical NYC Green Taxi trip records can forecast the **next hour's observed pickups per zone**, and how predicted pickup shares can inform a hypothetical 20-vehicle allocation. The emphasis is on **chronological validation, baseline comparison, error analysis, and responsible interpretation**, not a deployed dispatch system.

## Evaluation results

| Validation track | Metric | LightGBM Poisson | Baseline |
| --- | --- | ---: | ---: |
| **2025 independent expanding-window 4-fold CV** | MAE, mean ± fold SD | **1.141 ± 0.064** | 3-week mean: 1.278 ± 0.023 |
| **2026 H1 team-reported final model** | MAE | **1.034** | 3-week mean: 1.129 |
| 2026 H1 team-reported final model | RMSE | **1.918** | 3-week mean: 2.136 |
| 2026 H1 team-reported final model | TVD | **0.321** | Weekday-hour mean: 0.327 |
| 2026 H1 team-reported final model | Top-5 observed pickup share | **68.29%** | Weekday-hour mean: 67.34% |

**Important:** The 2025 4-fold study is an **independent reconstruction**, not a verified rerun of the identical final team model. Its MAE must not be compared with the 2026 value as if both results came from one unchanged implementation.

## Model evaluation

- [Evaluation summary and error analysis](reports/readme.md)
- [Validation design, metrics, and caveats](docs/methodology.md)
- [2025 independent cross-validation: fold-level metrics](reports/metrics/2025_4fold_model_scores.csv)
- [2025 independent cross-validation: mean and standard deviation](reports/metrics/2025_4fold_mean_sd.csv)

The 2026 team holdout improved count MAE by about 8.5% relative to the 3-week average, but the gains in TVD and Top-5 pickup share were modest. A morning peak in East Harlem North on 2026-06-01 07:00 (actual 53; predicted about 28.5) illustrates how aggregate MAE can mask high-demand underprediction.

## Data and task

- **Data:** [NYC TLC Green Taxi Trip Records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), January 2024 through June 2026 (30 monthly Parquet files).
- **Team-reported QA:** 1,506,768 original records; 1,501,696 pickup records after pickup-month and location validation.
- **Prediction target:** nonnegative *observed pickup count per zone per hour*, at a next-hour forecasting horizon.
- **Study region:** 20 high-volume pickup zones selected from the available past.
- **Model family:** LightGBM Poisson regression with calendar, historical pickup lags, rolling means, and past completed-trip flow signals.

Download the monthly `green_tripdata_YYYY-MM.parquet` files from the [NYC TLC Trip Record Data page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) and store them locally under `data/raw/` when running the analysis. Raw trip files are not committed to GitHub.

## Business illustration

Predicted zone shares are converted to hypothetical integer allocations for 20 vehicles using the largest-remainder method. This is not a validated dispatch optimization. NYC TLC trip logs do not identify unserved requests or actual taxi availability. The reported 2,470-count scenario difference concerns a *coverage proxy*, not observed additional trips or realized profit.

## Repository status

This repository presently publishes the **evaluation narrative, validation definitions, and reported fold-level results**. The full independent reproduction scripts, notebook, charts, and complete PDF report are being prepared separately; **the results here have not been newly recomputed by a GitHub Actions workflow**. The 2026 team final-model training source is not included.

## Team provenance

Developed from the **4Sight team project**. The 2026 evaluation numbers are team-reported; the 2025 cross-validation is separately reconstructed. Individual work should be described precisely rather than treating every team-developed model component as individually authored.

**Data credit:** [NYC Taxi & Limousine Commission](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
