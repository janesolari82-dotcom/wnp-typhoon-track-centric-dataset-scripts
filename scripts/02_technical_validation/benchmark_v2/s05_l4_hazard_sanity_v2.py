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

from benchmark.s05_l4_hazard_formula import run_l4 as legacy_run_l4  # noqa: E402
from benchmark.s00_core_common import RunConfig, StageResult, get_stage_paths, open_netcdf_readonly  # noqa: E402
from s00_common_v2 import create_results_layout_v2, quantile_safe, save_figure, stage_dirs, write_plot_data  # noqa: E402


def _sample_hazard_values(cfg: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dist_rows = []
    quant_rows = []
    panel_rows = []
    rng = np.random.default_rng(int(cfg.seed))
    representative = None

    for year in cfg.years:
        path = get_stage_paths(cfg, year)["final"]
        with open_netcdf_readonly(path, cfg) as ds:
            pairs = []
            for group_name, grp in ds.groups.items():
                if "time" not in grp.dimensions:
                    continue
                for idx in range(len(grp.dimensions["time"])):
                    pairs.append((group_name, idx))
            if not pairs:
                continue
            if len(pairs) > 30:
                sel = rng.choice(len(pairs), size=30, replace=False)
                pairs = [pairs[int(i)] for i in sel]
            for var in ["hazard_wind_power_daily", "hazard_rain_daily", "hazard_compound_daily"]:
                collected = []
                for group_name, idx in pairs:
                    grp = ds.groups[group_name]
                    if var not in grp.variables:
                        continue
                    arr = np.asarray(grp.variables[var][idx, :, :], dtype=float)
                    vals = arr[np.isfinite(arr)]
                    if vals.size == 0:
                        continue
                    if vals.size > 200:
                        vals = vals[rng.choice(vals.size, size=200, replace=False)]
                    collected.append(vals)
                    if representative is None and var == "hazard_compound_daily":
                        representative = (year, group_name, idx, arr.copy())
                if not collected:
                    continue
                all_vals = np.concatenate(collected)
                sample_vals = all_vals if all_vals.size <= 5000 else all_vals[rng.choice(all_vals.size, size=5000, replace=False)]
                for value in sample_vals:
                    dist_rows.append({"year": year, "variable": var, "value": float(value)})
                quant_rows.append(
                    {
                        "year": year,
                        "variable": var,
                        "p01": quantile_safe(all_vals, 0.01),
                        "p05": quantile_safe(all_vals, 0.05),
                        "p50": quantile_safe(all_vals, 0.50),
                        "p95": quantile_safe(all_vals, 0.95),
                        "p99": quantile_safe(all_vals, 0.99),
                        "max": float(np.nanmax(all_vals)),
                    }
                )
    if representative is not None:
        year, group_name, idx, arr = representative
        panel_rows = pd.DataFrame(
            {
                "row": np.repeat(np.arange(arr.shape[0]), arr.shape[1]),
                "col": np.tile(np.arange(arr.shape[1]), arr.shape[0]),
                "value": arr.reshape(-1),
                "year": year,
                "group": group_name,
                "time_idx": idx,
            }
        )
    return pd.DataFrame(dist_rows), pd.DataFrame(quant_rows), panel_rows


def run_l4_v2(cfg: RunConfig) -> StageResult:
    create_results_layout_v2(cfg)
    result = legacy_run_l4(cfg)
    out = stage_dirs(cfg, "L4")

    dist_df, quant_df, panel_df = _sample_hazard_values(cfg)
    write_plot_data(dist_df, out["csv"] / "hazard_distribution_by_year_plot_data.csv", cfg)
    write_plot_data(quant_df, out["csv"] / "hazard_quantile_diagnostics_plot_data.csv", cfg)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for ax, var in zip(axes, ["hazard_wind_power_daily", "hazard_rain_daily", "hazard_compound_daily"]):
        sub = dist_df[dist_df["variable"] == var]
        if not sub.empty:
            grouped = sub.groupby("year")["value"].median().reset_index()
            ax.plot(grouped["year"], grouped["value"], marker="o")
        ax.set_title(var)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Year")
    save_figure(fig, out["png"] / "hazard_distribution_by_year.png", cfg)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for var, sub in quant_df.groupby("variable"):
        ax.plot(sub["year"], sub["p95"], marker="o", label=f"{var} p95")
        ax.plot(sub["year"], sub["p99"], marker=".", linestyle="--", label=f"{var} p99")
    ax.set_title("Hazard Quantile Diagnostics")
    ax.set_xlabel("Year")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    save_figure(fig, out["png"] / "hazard_quantile_diagnostics.png", cfg)

    if not panel_df.empty:
        arr = panel_df["value"].to_numpy(dtype=float).reshape(41, 41)
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        im = ax.imshow(arr, origin="lower", cmap="viridis")
        ax.set_title("Representative Hazard Slice")
        ax.set_xlabel("window_lon index")
        ax.set_ylabel("window_lat index")
        fig.colorbar(im, ax=ax, shrink=0.8)
        save_figure(fig, out["png"] / "hazard_slice_sanity_panel.png", cfg)

    for artifact in [
        out["csv"] / "hazard_distribution_by_year_plot_data.csv",
        out["csv"] / "hazard_quantile_diagnostics_plot_data.csv",
        out["png"] / "hazard_distribution_by_year.png",
        out["png"] / "hazard_quantile_diagnostics.png",
        out["png"] / "hazard_slice_sanity_panel.png",
    ]:
        if artifact.exists():
            result.artifacts.append(str(artifact))
    return result
