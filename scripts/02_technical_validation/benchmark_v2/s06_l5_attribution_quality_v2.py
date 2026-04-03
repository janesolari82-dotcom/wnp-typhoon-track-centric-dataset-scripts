from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
BENCHMARK_DIR = PARENT_DIR / "benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark.s06_l5_emdat import run_l5 as legacy_run_l5  # noqa: E402
from benchmark.s00_core_common import RunConfig, StageResult  # noqa: E402
from s00_common_v2 import create_results_layout_v2, save_figure, stage_dirs, write_plot_data  # noqa: E402


def run_l5_v2(cfg: RunConfig) -> StageResult:
    create_results_layout_v2(cfg)
    result = legacy_run_l5(cfg)
    out = stage_dirs(cfg, "L5")

    alloc_df = pd.read_csv(out["root"] / "allocation_status_by_year.csv")
    cons_df = pd.read_csv(out["root"] / "conservation_check.csv")
    zero_df = pd.read_csv(out["root"] / "zero_weight_diagnosis.csv")

    plot_alloc = alloc_df[alloc_df["scope"] == "all_records"].copy()
    write_plot_data(plot_alloc, out["csv"] / "allocation_status_stacked_by_year_plot_data.csv", cfg)

    pivot = plot_alloc.pivot(index="year", columns="status", values="count").fillna(0).sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = None
    colors = {"allocated": "tab:green", "missing_loss": "tab:orange", "zero_weight": "tab:red", "unmatched": "tab:purple"}
    for status in [c for c in ["allocated", "missing_loss", "zero_weight", "unmatched"] if c in pivot.columns]:
        vals = pivot[status].to_numpy(dtype=float)
        ax.bar(pivot.index.astype(int), vals, bottom=bottom, label=status, color=colors.get(status))
        bottom = vals if bottom is None else bottom + vals
    ax.set_title("Allocation Status by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Record count")
    ax.grid(alpha=0.2, axis="y")
    ax.legend()
    save_figure(fig, out["png"] / "allocation_status_stacked_by_year.png", cfg)

    cons_long = cons_df[["year", "max_conservation_error", "mean_conservation_error", "p95_conservation_error", "p99_conservation_error"]].copy()
    write_plot_data(cons_long, out["csv"] / "conservation_error_boxplot_by_year_plot_data.csv", cfg)

    fig, ax = plt.subplots(figsize=(10, 5))
    data = []
    labels = []
    for _, row in cons_df.sort_values("year").iterrows():
        data.append([row["mean_conservation_error"], row["p95_conservation_error"], row["p99_conservation_error"], row["max_conservation_error"]])
        labels.append(str(int(row["year"])))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title("Conservation Error Diagnostics by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Error")
    ax.grid(alpha=0.2)
    save_figure(fig, out["png"] / "conservation_error_boxplot_by_year.png", cfg)

    diag_breakdown = zero_df.groupby("diagnosis_label", as_index=False).size().rename(columns={"size": "count"}).sort_values(
        "count", ascending=False
    )
    write_plot_data(diag_breakdown, out["csv"] / "zero_weight_diagnosis_breakdown_plot_data.csv", cfg)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(diag_breakdown["diagnosis_label"], diag_breakdown["count"], color="tab:red")
    ax.set_title("Zero-weight Diagnosis Breakdown")
    ax.set_xlabel("Diagnosis label")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2, axis="y")
    ax.tick_params(axis="x", rotation=45)
    save_figure(fig, out["png"] / "zero_weight_diagnosis_breakdown.png", cfg)

    for artifact in [
        out["csv"] / "allocation_status_stacked_by_year_plot_data.csv",
        out["csv"] / "conservation_error_boxplot_by_year_plot_data.csv",
        out["csv"] / "zero_weight_diagnosis_breakdown_plot_data.csv",
        out["png"] / "allocation_status_stacked_by_year.png",
        out["png"] / "conservation_error_boxplot_by_year.png",
        out["png"] / "zero_weight_diagnosis_breakdown.png",
    ]:
        result.artifacts.append(str(artifact))
    return result
