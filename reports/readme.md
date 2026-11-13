# NYC Green Taxi — Model Evaluation Summary

**Machine-learning portfolio | 4Sight team project extension**

## Key findings

| Experiment | Metric | Model | Comparator |
| --- | --- | ---: | ---: |
| **2025 independently reconstructed 4-fold CV** | MAE (mean ± fold SD) | **1.141 ± 0.064** | 3-week mean: 1.278 ± 0.023 |
| **2025 independently reconstructed 4-fold CV** | TVD mean | **0.293** | Weekday-hour mean: 0.303 |
| **2026 team-reported holdout** | MAE | **1.034** | 3-week mean: 1.129 |
| **2026 team-reported holdout** | RMSE | **1.918** | 3-week mean: 2.136 |
| **2026 team-reported holdout** | TVD | **0.321** | Weekday-hour mean: 0.327 |
| **2026 team-reported holdout** | Top-5 observed pickup share | **68.29%** | Weekday-hour mean: 67.34% |

**Do not compare 1.141 and 1.034 as one model's progression.** The 2025 cross-validation was independently reconstructed and is not proven identical to the team's final LightGBM pipeline.

## 1. Problem and data

Predict the next hour's *observed pickup count* for each of 20 selected NYC Green Taxi zones. Raw records: 1,506,768 trips (2024 Jan–2026 Jun); 1,501,696 records after month/zone checks, as reported in the team project. The dataset cannot reveal unfulfilled requests or actual available taxis.

## 2. Temporal validation

The independently reconstructed 2025 experiment uses expanding-window 4-fold validation, with training strictly preceding validation. Training begins in 2024; each 2025 quarter is tested once. Scores and fold-level results: [2025 fold scores](metrics/2025_4fold_model_scores.csv) and [mean/SD table](metrics/2025_4fold_mean_sd.csv).

The reported 2026 retrospective evaluation uses a final model selected in 2025 and then evaluated by hour for January through June 2026. Its original training source is not yet reproduced in this repository.

## 3. Error analysis

The final model's 2026 MAE was 1.034 zone-hour pickups, approximately 8.5% below the historical three-week baseline. That aggregate does not guarantee good performance at every busy hour. For example, on **2026-06-01 at 07:00**, East Harlem North recorded **53 pickups**, whereas the published model prediction was approximately **28.5**. Peak-demand underprediction merits targeted follow-up.

## 4. Evaluation beyond MAE

Regional pickup shares determine ranking and hypothetical allocation. In the 2026 team results, TVD moves from about 0.327 for weekday-hour mean to 0.321 for the Poisson model, while Top-5 observed share moves from 67.34% to 68.29%. These are **modest improvements** and are not proof of more completed rides.

## 5. Illustrative decision use

For 2026-06-01 17:00, predicted shares of selected zones imply a hypothetical 20-car allocation: East Harlem North 28.7% → 6 cars; East Harlem South 26.3% → 5 cars (largest-remainder rounding). Actual cars, relocation times, legal and availability constraints are not modeled. A 2,470-count six-month *coverage proxy* difference is not 2,470 additional rides. Conditional profitability calculations are scenario assumptions, not realized financial impact.

## 6. Limitations and next steps

- Integrate the 2025 and 2026 evaluations under one identical, leakage-tested training pipeline.
- Independently rerun the entire model from the original public TLC data, with fixed environment versions.
- Audit peak-hour residuals and error by zone and time period.
- Clearly credit the original 4Sight team and distinguish individual portfolio work.

**Dataset:** [NYC Taxi & Limousine Commission Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).