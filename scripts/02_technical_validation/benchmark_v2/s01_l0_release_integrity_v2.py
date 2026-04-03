from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
BENCHMARK_DIR = PARENT_DIR / "benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark.s01_l0_integrity import run_l0 as legacy_run_l0  # noqa: E402
from benchmark.s00_core_common import RunConfig, StageResult  # noqa: E402
from s00_common_v2 import create_results_layout_v2, save_figure, stage_dirs, write_plot_data  # noqa: E402


def run_l0_v2(cfg: RunConfig) -> StageResult:
    create_results_layout_v2(cfg)
    result = legacy_run_l0(cfg)
    out = stage_dirs(cfg, "L0")
    inventory_path = out["root"] / "file_inventory.csv"
    df = pd.read_csv(inventory_path)

    bar_df = (
        df.assign(pass_flag=(df["status"] == "PASS").astype(int))
        .groupby("stage", as_index=False)["pass_flag"]
        .sum()
        .rename(columns={"pass_flag": "pass_count"})
        .sort_values("stage")
        .reset_index(drop=True)
    )
    write_plot_data(bar_df, out["csv"] / "release_integrity_status_bar_plot_data.csv", cfg)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(bar_df["stage"], bar_df["pass_count"], color="tab:blue")
    ax.set_title("Release Integrity Status by Stage")
    ax.set_xlabel("Stage")
    ax.set_ylabel("PASS count")
    ax.grid(alpha=0.2, axis="y")
    save_figure(fig, out["png"] / "release_integrity_status_bar.png", cfg)

    heat_df = (
        df.assign(status_code=df["status"].map({"PASS": 0, "WARN": 1, "FAIL": 2}).fillna(2).astype(int))
        .pivot_table(index="year", columns="stage", values="status_code", aggfunc="max")
        .sort_index()
    )
    heat_long = heat_df.reset_index().melt(id_vars="year", var_name="stage", value_name="status_code")
    write_plot_data(heat_long, out["csv"] / "release_integrity_by_year_heatmap_plot_data.csv", cfg)

    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    im = ax.imshow(heat_df.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=2)
    ax.set_title("Release Integrity by Year and Stage")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Year")
    ax.set_xticks(np.arange(len(heat_df.columns)))
    ax.set_xticklabels(heat_df.columns.tolist(), rotation=45, ha="right")
    ax.set_yticks(np.arange(len(heat_df.index)))
    ax.set_yticklabels([str(int(y)) for y in heat_df.index.tolist()])
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(["PASS", "WARN", "FAIL"])
    save_figure(fig, out["png"] / "release_integrity_by_year_heatmap.png", cfg)

    for artifact in [
        out["csv"] / "release_integrity_status_bar_plot_data.csv",
        out["csv"] / "release_integrity_by_year_heatmap_plot_data.csv",
        out["png"] / "release_integrity_status_bar.png",
        out["png"] / "release_integrity_by_year_heatmap.png",
    ]:
        result.artifacts.append(str(artifact))
    return result
