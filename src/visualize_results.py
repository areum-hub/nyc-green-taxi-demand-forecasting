"""GitHub portfolio plots from previously evaluated summary CSVs.

Run from the repository root:
    pip install pandas matplotlib
    python src/visualize_results.py

This script only visualizes recorded model metrics. It does not train a model.
2025: independent expanding-window CV reconstruction.
2026: separately reported 4Sight final-model holdout.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "metrics"
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
METHODS = {
    "Poisson · 19-feature reproduction": "LightGBM Poisson",
    "3-week average": "3-week average",
    "Weekday-hour mean": "Weekday-hour mean",
}
PALETTE = {
    "LightGBM Poisson": "#13795b",
    "3-week average": "#d99b29",
    "Weekday-hour mean": "#7a899d",
    "Weekday-hour average": "#7a899d",
}


def finish(fig, name):
    fig.set_facecolor("#f8fafc")
    fig.tight_layout()
    fig.savefig(FIGURES / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_2025_folds():
    df = pd.read_csv(METRICS / "2025_4fold_model_scores.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for original, label in METHODS.items():
        sub = df.loc[df["method"].eq(original)].copy()
        sub["fold_no"] = sub["fold"].str.extract(r"(\d+)").astype(int)
        sub = sub.sort_values("fold_no")
        ax.plot(sub["fold_no"], sub["mae"], "-o", linewidth=2,
                label=label, color=PALETTE[label])
    ax.set_xticks([1, 2, 3, 4], ["Fold 1", "Fold 2", "Fold 3", "Fold 4"])
    ax.set_title("2025 independent 4-fold CV — MAE by fold")
    ax.set_ylabel("MAE (pickups per zone-hour; lower is better)")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    finish(fig, "2025_cv_mae_by_fold")


def plot_2025_summary():
    df = pd.read_csv(METRICS / "2025_4fold_mean_sd.csv")
    rows = [df.loc[df["method"].eq(k)].iloc[0] for k in METHODS]
    labels = list(METHODS.values())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (row, label) in enumerate(zip(rows, labels)):
        ax.errorbar(row["mae_mean"], i, xerr=row["mae_sd"], fmt="o",
                    capsize=6, markersize=9, color=PALETTE[label])
        ax.text(row["mae_mean"] + row["mae_sd"] + .012, i,
                f'{row["mae_mean"]:.3f} ± {row["mae_sd"]:.3f}', va="center")
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(1.0, 1.5)
    ax.set_xlabel("MAE (pickups per zone-hour)")
    ax.set_title("2025 independent CV — Mean ± fold standard deviation")
    ax.grid(axis="x", alpha=.2)
    ax.text(.02, -.21, "Error bars show sample SD, NOT confidence intervals",
            transform=ax.transAxes, fontsize=9)
    finish(fig, "2025_cv_mae_mean_sd")


def plot_2026_mae():
    df = pd.read_csv(METRICS / "2026_team_holdout_summary.csv")
    sub = df.loc[df["metric"].eq("mae")]
    labels = ["LightGBM Poisson", "3-week average"]
    values = [sub.loc[sub["method"].eq(k), "value"].iloc[0] for k in labels]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    bars = ax.barh(labels, values, color=[PALETTE[k] for k in labels])
    ax.bar_label(bars, fmt="%.3f", padding=5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.4)
    ax.set_xlabel("MAE (pickups per zone-hour)")
    ax.set_title("2026 Jan–Jun — Team-reported holdout MAE")
    finish(fig, "2026_holdout_mae")


def plot_2026_share():
    df = pd.read_csv(METRICS / "2026_team_holdout_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, metric, title, factor in [
        (axes[0], "tvd", "Demand share TVD (lower is better)", 1.0),
        (axes[1], "top5", "Top-5 observed share (higher is better)", 100.0),
    ]:
        sub = df.loc[df["metric"].eq(metric)]
        labels = ["Weekday-hour average", "LightGBM Poisson"]
        vals = [sub.loc[sub["method"].eq(k), "value"].iloc[0] * factor
                for k in labels]
        bars = ax.bar(labels, vals, color=[PALETTE[k] for k in labels])
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, value*1.025,
                    f"{value:.3f}" if metric == "tvd" else f"{value:.2f}%",
                    ha="center")
        ax.set_ylim(0, max(vals)*1.22)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", rotation=12)
    fig.suptitle("2026 holdout — Team-reported share metrics")
    finish(fig, "2026_share_metrics")


if __name__ == "__main__":
    plot_2025_folds()
    plot_2025_summary()
    plot_2026_mae()
    plot_2026_share()
    print("Created four SVG charts under reports/figures/")
