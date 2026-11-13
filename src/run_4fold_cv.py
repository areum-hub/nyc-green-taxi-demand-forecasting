"""Independent 4-fold expanding-window reconstruction of NYC Green Taxi demand.

Run from repository root after:
    python src/prepare_data.py
    python src/run_4fold_cv.py

This script uses 2024-2025 history only. It is NOT the original 4Sight final
model implementation or its 2026 evaluation pipeline. The 19 features are
independently reconstructed; do not describe the resulting scores as a rerun
of the team's final model.
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "metrics"
OUT.mkdir(parents=True, exist_ok=True)
trips = pd.read_pickle(ROOT / "data" / "processed" / "valid_trips.pkl")
trips = trips.loc[trips.pickup.lt(pd.Timestamp("2026-01-01"))].copy()
trips["pickup_zone"] = trips.pickup_zone.astype("int16")
trips["dropoff_zone"] = pd.to_numeric(trips.dropoff_zone, errors="coerce")
trips["distance"] = pd.to_numeric(trips.distance, errors="coerce")
trips["fare"] = pd.to_numeric(trips.fare, errors="coerce")

# Select zones using the first fold's training period ONLY (2024).
ZONES = (trips.loc[trips.pickup.lt("2025-01-01"), "pickup_zone"]
         .value_counts().head(20).index.astype(int).tolist())
all_hours = pd.date_range("2024-01-01 00:00", "2025-12-31 23:00", freq="h")
full_index = pd.MultiIndex.from_product(
    [ZONES, all_hours], names=["zone_id", "target_hour"]
)

# Observed pickups per zone-hour, including zero-pickup hours.
base = (trips.loc[trips.pickup_zone.isin(ZONES)]
        .assign(target_hour=lambda df: df.pickup.dt.floor("h"))
        .groupby(["pickup_zone", "target_hour"]).size().rename("y"))
base.index = base.index.set_names(["zone_id", "target_hour"])
panel = base.reindex(full_index, fill_value=0).rename("y").to_frame()

# Completed-trip variables must be grouped by dropoff completion hour.
sub = trips.copy()
sub["dur_min"] = (sub.dropoff - sub.pickup).dt.total_seconds() / 60
sub["mph"] = sub.distance / (sub.dur_min / 60)
sub = sub.loc[
    sub.dropoff_zone.between(1, 263)
    & sub.dur_min.between(1, 180)
    & sub.distance.between(0, 100)
    & sub.fare.between(0, 500)
    & sub.mph.between(0, 100)
    & sub.dropoff.notna()
].copy()
sub["completed_hour"] = sub.dropoff.dt.floor("h")
sub["same_zone"] = sub.pickup_zone.eq(sub.dropoff_zone).astype("int8")
completed_by_pickup = (
    sub.loc[sub.pickup_zone.isin(ZONES)]
    .groupby(["pickup_zone", "completed_hour"])
    .agg(
        complete_n=("fare", "size"),
        complete_dur=("dur_min", "sum"),
        complete_fare=("fare", "sum"),
        complete_same=("same_zone", "sum"),
    )
)
completed_by_pickup.index = completed_by_pickup.index.set_names(
    ["zone_id", "target_hour"]
)
panel = panel.join(completed_by_pickup, how="left")
for name in ["complete_n", "complete_dur", "complete_fare", "complete_same"]:
    panel[name] = panel[name].fillna(0)

arrivals = (
    sub.loc[sub.dropoff_zone.isin(ZONES)]
    .groupby(["dropoff_zone", "completed_hour"])
    .size().rename("arrivals")
)
arrivals.index = arrivals.index.set_names(["zone_id", "target_hour"])
panel = panel.join(arrivals, how="left")
panel["arrivals"] = panel.arrivals.fillna(0)
panel = panel.reset_index().sort_values(["zone_id", "target_hour"]).reset_index(drop=True)

z = panel.groupby("zone_id", sort=False)
for lag in [1, 24, 168, 336, 504]:
    panel[f"lag_{lag}"] = z["y"].shift(lag)
for w in [24, 168]:
    panel[f"rolling_mean_{w}"] = z["y"].transform(
        lambda s: s.shift(1).rolling(w, min_periods=w).mean()
    )

# All trip-based historical signals lag at least 1 hour.
for name in ["complete_n", "complete_dur", "complete_fare", "complete_same"]:
    panel[f"past_{name}_168h"] = z[name].transform(
        lambda s: s.shift(1).rolling(168, min_periods=168).sum()
    )
n = panel.past_complete_n_168h.replace(0, np.nan)
dur = panel.past_complete_dur_168h.replace(0, np.nan)
panel["mean_duration_168h"] = panel.past_complete_dur_168h / n
panel["fare_per_hour_168h"] = panel.past_complete_fare_168h / dur * 60
panel["same_zone_share_168h"] = panel.past_complete_same_168h / n
panel["arrivals_lag1"] = z["arrivals"].shift(1)
panel["arrivals_mean24"] = z["arrivals"].transform(
    lambda s: s.shift(1).rolling(24, min_periods=24).mean()
)
panel["net_flow_mean24"] = panel.arrivals_mean24 - panel.rolling_mean_24

panel["hour"] = panel.target_hour.dt.hour
panel["dow"] = panel.target_hour.dt.dayofweek
panel["month"] = panel.target_hour.dt.month
panel["weekend"] = (panel.dow >= 5).astype("int8")
panel["day_index"] = (panel.target_hour - pd.Timestamp("2024-01-01")).dt.days
panel["zone_id"] = panel.zone_id.astype("category")

FEATURES = [
    "zone_id", "hour", "dow", "month", "weekend", "day_index",
    "lag_1", "lag_24", "lag_168", "rolling_mean_24", "rolling_mean_168",
    "mean_duration_168h", "fare_per_hour_168h", "same_zone_share_168h",
    "arrivals_lag1", "arrivals_mean24", "net_flow_mean24", "lag_336", "lag_504",
]
assert len(FEATURES) == 19
panel = panel.dropna(subset=["lag_504"]).copy()
DST_DAYS = ["2024-03-10", "2024-11-03", "2025-03-09", "2025-11-02"]
panel = panel.loc[
    ~panel.target_hour.dt.normalize().isin(pd.to_datetime(DST_DAYS))
].copy()
panel["pred_3week"] = (panel.lag_168 + panel.lag_336 + panel.lag_504) / 3

print("Selected zone IDs:", ZONES, flush=True)
print("Panel rows:", len(panel), "period:", panel.target_hour.min(),
      panel.target_hour.max(), flush=True)
print("Historical completed trips:", len(sub), "at", time.asctime(), flush=True)

# Boundary = first excluded training date = first included validation date.
FOLDS = [
    ("Fold 1", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-04-01")),
    ("Fold 2", pd.Timestamp("2025-04-01"), pd.Timestamp("2025-07-01")),
    ("Fold 3", pd.Timestamp("2025-07-01"), pd.Timestamp("2025-10-01")),
    ("Fold 4", pd.Timestamp("2025-10-01"), pd.Timestamp("2026-01-01")),
]
rows = []
for fold, va_start, va_end in FOLDS:
    tra = panel.loc[panel.target_hour.lt(va_start)].copy()
    va = panel.loc[
        panel.target_hour.ge(va_start) & panel.target_hour.lt(va_end)
    ].copy()
    assert tra.target_hour.max() < va.target_hour.min()
    assert tra.target_hour.min().year == 2024

    est = lgb.LGBMRegressor(
        objective="poisson", n_estimators=600, learning_rate=0.05,
        num_leaves=31, max_depth=-1, min_child_samples=25,
        reg_lambda=0.0, random_state=42, n_jobs=4, verbosity=-1,
    )
    X_train, X_val = tra[FEATURES].copy(), va[FEATURES].copy()
    X_train["zone_id"] = X_train["zone_id"].cat.as_ordered()
    X_val["zone_id"] = X_val["zone_id"].cat.as_ordered()
    est.fit(X_train, tra["y"], categorical_feature=["zone_id"])
    va["pred_model"] = np.clip(est.predict(X_val), 0, None)
    va["pred_3weeks"] = va.pred_3week

    static = tra.groupby(["zone_id", "dow", "hour"], observed=True)["y"].mean()
    va = va.join(static.rename("pred_static"), on=["zone_id", "dow", "hour"])
    va["pred_static"] = va.pred_static.fillna(0)
    va["obs_total"] = va.groupby("target_hour")["y"].transform("sum")
    for col in ["pred_model", "pred_3weeks", "pred_static"]:
        va[col + "_sum"] = va.groupby("target_hour")[col].transform("sum")
        va[col + "_share"] = np.where(
            va[col + "_sum"].gt(0), va[col] / va[col + "_sum"], 1 / 20
        )
    va["actual_share"] = np.where(
        va.obs_total.gt(0), va.y / va.obs_total, np.nan
    )
    evaluated = va.loc[va.obs_total.gt(0)]
    for name, col in [
        ("Poisson · 19-feature reproduction", "pred_model"),
        ("3-week average", "pred_3weeks"),
        ("Weekday-hour mean", "pred_static"),
    ]:
        mae = float(mean_absolute_error(va.y, va[col]))
        rmse = float(np.sqrt(mean_squared_error(va.y, va[col])))
        tvd = float((
            evaluated.assign(
                err=(evaluated[col + "_share"] - evaluated.actual_share).abs()
            ).groupby("target_hour").err.sum() / 2
        ).mean())
        ranked = evaluated.sort_values(
            ["target_hour", col + "_share", "zone_id"],
            ascending=[True, False, True],
        )
        top5 = float(
            ranked.groupby("target_hour", sort=False).head(5)
            .groupby("target_hour").actual_share.sum().mean()
        )
        rows.append({
            "fold": fold,
            "train_until_exclusive": str(va_start.date()),
            "validation_from": str(va_start.date()),
            "validation_through": str((va_end - pd.Timedelta("1d")).date()),
            "method": name,
            "mae": mae, "rmse": rmse, "tvd": tvd, "top5": top5,
            "train_rows": len(tra), "validation_rows": len(va),
            "n_valid_hours": va.target_hour.nunique(),
            "n_share_hours": evaluated.target_hour.nunique(),
        })

    results = pd.DataFrame(rows)
    results.to_csv(
        OUT / "2025_4fold_model_scores.csv", index=False, encoding="utf-8"
    )
    print(f"{fold} - train={len(tra):,}, valid={len(va):,}", flush=True)
    print(
        results.loc[results.fold.eq(fold), ["method", "mae", "tvd", "top5"]]
        .to_string(index=False), flush=True,
    )

scores = pd.DataFrame(rows)
summary = scores.groupby("method", as_index=False).agg(
    mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
    tvd_mean=("tvd", "mean"), tvd_sd=("tvd", "std"),
    top5_mean=("top5", "mean"), top5_sd=("top5", "std"),
    rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
)
summary.to_csv(OUT / "2025_4fold_mean_sd.csv", index=False, encoding="utf-8")
print("SUMMARY:", summary.to_string(index=False), flush=True)

with (OUT / "methodology.json").open("w", encoding="utf-8") as fp:
    json.dump({
        "folds": [
            {
                "train_end_exclusive": str(a.date()),
                "valid_start": str(a.date()),
                "valid_end_exclusive": str(b.date()),
            }
            for _, a, b in FOLDS
        ],
        "zones": ZONES,
        "features": FEATURES,
        "excluded_dst_days": DST_DAYS,
        "scope": "Independent 2025 expanding-window 4-Fold reproduction",
        "notice": "NOT verified identical to original team 2026 pipeline",
        "model": {
            "name": "LightGBM Poisson",
            "n_estimators": 600, "learning_rate": 0.05,
            "num_leaves": 31, "min_child_samples": 25, "seed": 42,
        },
    }, fp, ensure_ascii=False, indent=2)
