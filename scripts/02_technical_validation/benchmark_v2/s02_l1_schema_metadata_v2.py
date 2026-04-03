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

from benchmark.s02_l1_schema import run_l1 as legacy_run_l1  # noqa: E402
from benchmark.s00_core_common import RunConfig, StageResult  # noqa: E402
from s00_common_v2 import create_results_layout_v2, save_figure, stage_dirs, write_plot_data  # noqa: E402


def run_l1_v2(cfg: RunConfig) -> StageResult:
    create_results_layout_v2(cfg)
    result = legacy_run_l1(cfg)
    out = stage_dirs(cfg, "L1")

    frames = []
    for year in cfg.years:
        path = out["root"] / f"schema_diff_{year}.csv"
        if path.exists():
            df = pd.read_csv(path)
            if not df.empty:
                frames.append(df)
    issues = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["year", "check_type", "severity", "stage_src", "stage_dst", "group", "var", "expected", "actual", "message"]
    )

    by_year = pd.DataFrame({"year": cfg.years})
    if not issues.empty:
        counts = (
            issues.groupby(["year", "severity"], as_index=False)
            .size()
            .pivot(index="year", columns="severity", values="size")
            .fillna(0)
            .reset_index()
        )
        by_year = by_year.merge(counts, how="left", on="year").fillna(0)
    for col in ["FAIL", "WARN"]:
        if col not in by_year.columns:
            by_year[col] = 0
    by_year = by_year[["year", "FAIL", "WARN"]]
    write_plot_data(by_year, out["csv"] / "schema_issue_count_by_year_plot_data.csv", cfg)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(by_year["year"], by_year["FAIL"], marker="o", label="FAIL")
    ax.plot(by_year["year"], by_year["WARN"], marker="o", label="WARN")
    ax.set_title("Schema Issues by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Issue count")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, out["png"] / "schema_issue_count_by_year.png", cfg)

    if not issues.empty:
        breakdown = issues.groupby("check_type", as_index=False).size().rename(columns={"size": "issue_count"}).sort_values(
            "issue_count", ascending=False
        )
    else:
        breakdown = pd.DataFrame({"check_type": ["none"], "issue_count": [0]})
    write_plot_data(breakdown, out["csv"] / "schema_check_type_breakdown_plot_data.csv", cfg)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(breakdown["check_type"], breakdown["issue_count"], color="tab:orange")
    ax.set_title("Schema Issue Breakdown by Check Type")
    ax.set_xlabel("Check type")
    ax.set_ylabel("Issue count")
    ax.grid(alpha=0.2, axis="y")
    ax.tick_params(axis="x", rotation=45)
    save_figure(fig, out["png"] / "schema_check_type_breakdown.png", cfg)

    for artifact in [
        out["csv"] / "schema_issue_count_by_year_plot_data.csv",
        out["csv"] / "schema_check_type_breakdown_plot_data.csv",
        out["png"] / "schema_issue_count_by_year.png",
        out["png"] / "schema_check_type_breakdown.png",
    ]:
        result.artifacts.append(str(artifact))
    return result
