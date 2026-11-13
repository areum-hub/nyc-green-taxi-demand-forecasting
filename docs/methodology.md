# Model validation methodology

## Dataset and target

NYC TLC Green Taxi Trip Record Data, January 2024–June 2026 (30 monthly Parquet files). The team-reported quality review starts with **1,506,768** rows and retains **1,501,696** valid pickup records after month and pickup-zone validation. The target is the number of **observed pickups per zone per next hour**, not unserved passenger requests.

## Two distinct evaluations

| Validation | Training/selection | Evaluation | Status |
| --- | --- | --- | --- |
| 2025 4-fold independent reconstruction | Expanding window from 2024 onward | 2025 Q1, Q2, Q3, Q4 | Separate experiment; fold results included in `reports/metrics/` |
| 2026 team final-model evaluation | 2025 Q4 internal model selection; 2024–25 refit | 2026 Jan–Jun retrospective hourly evaluation | Team-reported; original team training pipeline **not included** |

The two tracks differ in implementation and **must not be compared as if they are one model experiment**.

### Expanding-window folds

| Fold | Training | Validation |
| --- | --- | --- |
| 1 | 2024 Jan–Dec | 2025 Jan–Mar |
| 2 | 2024 Jan–2025 Mar | 2025 Apr–Jun |
| 3 | 2024 Jan–2025 Jun | 2025 Jul–Sep |
| 4 | 2024 Jan–2025 Sep | 2025 Oct–Dec |

In the independently reconstructed experiment, target zones are selected from **2024 only**; 2026 observations do not enter training or validation. Lag 504 initialization and NYC DST transition days are excluded from evaluation. The independently reconstructed model uses 19 features: six zone/calendar variables, seven historical pickup variables, and six past completed-trip/arrival variables. Time-dependent completed-trip signals must be lagged to avoid future information.

### Fixed model / baselines

The reconstructed model is LightGBM with Poisson objective, 600 estimators, learning_rate=0.05, num_leaves=31, min_child_samples=25 and random_state=42. Baselines are 3-week historical same-hour average and weekday-hour mean. Original team code was not available for verification of exact implementation equivalence.

### Metrics

- **MAE, RMSE:** count prediction errors per zone-hour; lower is better.
- **TVD:** mean hourly total variation distance between actual and predicted pickup shares over the selected 20 zones; lower is better.
- **Top-5:** mean observed pickup share of the five highest-ranked predicted zones; higher is better.
- **Fold error bars:** four-fold **sample standard deviation**, not a confidence interval.

## Caveats

TLC records describe completed pickups, not unmet demand or fleet availability. The allocation of 20 vehicles is a hypothetical, largest-remainder conversion of prediction shares and not a validated dispatch policy. Six-month coverage counts and economic sensitivity are scenario metrics, not observed incremental rides or realized profit.

**Source:** [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).