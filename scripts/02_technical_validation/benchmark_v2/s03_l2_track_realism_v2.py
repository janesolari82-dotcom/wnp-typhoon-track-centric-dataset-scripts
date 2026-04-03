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

from benchmark.s03_l2_track_physics import run_l2 as legacy_run_l2  # noqa: E402
from benchmark.s00_core_common import RunConfig, StageResult, get_stage_paths, open_netcdf_readonly  # noqa: E402
from s00_common_v2 import TRACK_VARS, create_results_layout_v2, save_figure, stage_dirs, valid_mask_default, write_plot_data  # noqa: E402


def _collect_distribution_rows(cfg: RunConfig) -> pd.DataFrame:
    rows = []
    for year in cfg.years:
        paths = get_stage_paths(cfg, year)
        with open_netcdf_readonly(paths["base"], cfg) as ds_base, open_netcdf_readonly(paths["final"], cfg) as ds_final:
            groups = sorted(set(ds_base.groups.keys()) & set(ds_final.groups.keys()))
            for var in TRACK_VARS:
                base_vals = []
                final_vals = []
                for group_name in groups:
                    gb = ds_base.groups[group_name]
                    gf = ds_final.groups[group_name]
                    if var not in gb.variables or var not in gf.variables:
                        continue
                    b = np.asarray(gb.variables[var][:], dtype=float).reshape(-1)
                    f = np.asarray(gf.variables[var][:], dtype=float).reshape(-1)
                    n = min(b.size, f.size)
                    if n == 0:
                        continue
                    b = b[:n]
                    f = f[:n]
                    base_vals.append(b)
                    final_vals.append(f)
                if not base_vals:
                    continue
                b_all = np.concatenate(base_vals)
                f_all = np.concatenate(final_vals)
                for source, arr in [("base", b_all), ("final", f_all)]:
                    vals = arr[valid_mask_default(arr)]
                    if vals.size == 0:
                        continue
                    sample = vals if vals.size <= 5000 else vals[np.linspace(0, vals.size - 1, 5000, dtype=int)]
                    for value in sample:
                        rows.append({"year": year, "variable": var, "source": source, "value": float(value)})
    return pd.DataFrame(rows)


def _yearly_counts(cfg: RunConfig) -> pd.DataFrame:
    rows = []
    for year in cfg.years:
        paths = get_stage_paths(cfg, year)
        with open_netcdf_readonly(paths["base"], cfg) as ds:
            storm_count = len(ds.groups)
            track_points = 0
            for grp in ds.groups.values():
                if "time" in grp.dimensions:
                    track_points += len(grp.dimensions["time"])
        rows.append({"year": year, "storm_count": storm_count, "track_point_count": track_points})
    return pd.DataFrame(rows)


def run_l2_v2(cfg: RunConfig) -> StageResult:
    create_results_layout_v2(cfg)
    result = legacy_run_l2(cfg)
    out = stage_dirs(cfg, "L2")

    dist_df = _collect_distribution_rows(cfg)
    write_plot_data(dist_df, out["csv"] / "track_distribution_base_vs_final_plot_data.csv", cfg)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()
    for ax, var in zip(axes, TRACK_VARS):
        sub = dist_df[dist_df["variable"] == var]
        base = sub[sub["source"] == "base"]["value"].to_numpy(dtype=float)
        final = sub[sub["source"] == "final"]["value"].to_numpy(dtype=float)
        bins = 40
        if base.size:
            ax.hist(base, bins=bins, alpha=0.5, density=True, label="base")
        if final.size:
            ax.hist(final, bins=bins, alpha=0.5, density=True, label="final")
        ax.set_title(var)
        ax.grid(alpha=0.2)
    for ax in axes[len(TRACK_VARS) :]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if labels:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    save_figure(fig, out["png"] / "track_distribution_base_vs_final.png", cfg)

    frames = []
    for year in cfg.years:
        path = out["root"] / f"track_physics_metrics_{year}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    metrics_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    write_plot_data(metrics_df, out["csv"] / "track_variable_trend_by_year_plot_data.csv", cfg)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()
    for ax, var in zip(axes, TRACK_VARS):
        sub = metrics_df[metrics_df["variable"] == var].sort_values("year")
        if not sub.empty:
            ax.plot(sub["year"], sub["base_mean"], marker="o", label="base_mean")
            ax.plot(sub["year"], sub["final_mean"], marker="o", label="final_mean")
        ax.set_title(var)
        ax.grid(alpha=0.2)
    for ax in axes[len(TRACK_VARS) :]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if labels:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    save_figure(fig, out["png"] / "track_variable_trend_by_year.png", cfg)

    counts_df = _yearly_counts(cfg)
    write_plot_data(counts_df, out["csv"] / "storm_count_and_track_points_by_year_plot_data.csv", cfg)

    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.bar(counts_df["year"], counts_df["storm_count"], alpha=0.6, color="tab:blue", label="storm_count")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Storm count", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(counts_df["year"], counts_df["track_point_count"], color="tab:red", marker="o", label="track_point_count")
    ax2.set_ylabel("Track point count", color="tab:red")
    ax1.set_title("Storm Count and Track Points by Year")
    ax1.grid(alpha=0.2)
    save_figure(fig, out["png"] / "storm_count_and_track_points_by_year.png", cfg)

    for artifact in [
        out["csv"] / "track_distribution_base_vs_final_plot_data.csv",
        out["csv"] / "track_variable_trend_by_year_plot_data.csv",
        out["csv"] / "storm_count_and_track_points_by_year_plot_data.csv",
        out["png"] / "track_distribution_base_vs_final.png",
        out["png"] / "track_variable_trend_by_year.png",
        out["png"] / "storm_count_and_track_points_by_year.png",
    ]:
        result.artifacts.append(str(artifact))
    return result
