from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
BENCHMARK_DIR = PARENT_DIR / "benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from benchmark.s04_l3_alignment import (  # noqa: E402
    _aggregate_patch,
    _missing_mask,
    _year_raster_specs,
    run_l3 as legacy_run_l3,
)
from benchmark.s00_core_common import RunConfig, StageResult, get_stage_paths, open_netcdf_readonly  # noqa: E402
from s00_common_v2 import create_results_layout_v2, save_figure, stage_dirs, write_plot_data  # noqa: E402


def _make_example_panel(cfg: RunConfig, reproj_df: pd.DataFrame, out: dict[str, Path]) -> list[str]:
    valid = reproj_df[(reproj_df["status"] == "PASS") & reproj_df["year"].notna()]
    if valid.empty:
        return []
    row = valid.iloc[0]
    year = int(row["year"])
    group_name = str(row["group"])
    time_idx = int(row["time_idx"])
    variable = str(row["variable"])

    specs = {spec.var_name: spec for spec in _year_raster_specs(cfg, year)}
    if variable not in specs:
        return []
    raster_path = Path(specs[variable].path)
    method = specs[variable].method
    final_path = get_stage_paths(cfg, year)["poplight"]

    with open_netcdf_readonly(final_path, cfg) as ds, rasterio.open(raster_path) as rs:
        grp = ds.groups[group_name]
        lat = float(grp.variables["center_lat"][time_idx])
        lon = float(grp.variables["center_lon"][time_idx])
        row_idx, col_idx = rs.index(lon, lat)
        patch = rs.read(1, window=Window(col_idx - 205, row_idx - 205, 410, 410), boundless=True, fill_value=rs.nodata).astype(
            np.float32
        )
        recalc = _aggregate_patch(patch, rs.nodata, method, flip=True)
        nc_slice = np.asarray(grp.variables[variable][time_idx, :, :], dtype=np.float32)
        diff = np.abs(recalc - nc_slice)
        diff[_missing_mask(recalc) | _missing_mask(nc_slice)] = np.nan

    panel_df = pd.DataFrame(
        {
            "row": np.repeat(np.arange(recalc.shape[0]), recalc.shape[1]),
            "col": np.tile(np.arange(recalc.shape[1]), recalc.shape[0]),
            "recalc": recalc.reshape(-1),
            "nc_slice": nc_slice.reshape(-1),
            "abs_diff": diff.reshape(-1),
        }
    )
    write_plot_data(panel_df, out["csv"] / "window_reprojection_example_panel_data.csv", cfg)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    ims = [
        axes[0].imshow(recalc, origin="lower", cmap="viridis"),
        axes[1].imshow(nc_slice, origin="lower", cmap="viridis"),
        axes[2].imshow(diff, origin="lower", cmap="magma"),
    ]
    titles = ["Reconstructed from raster", "Stored window slice", "Absolute difference"]
    for ax, im, title in zip(axes, ims, titles):
        ax.set_title(title)
        ax.set_xlabel("window_lon index")
        ax.set_ylabel("window_lat index")
        fig.colorbar(im, ax=ax, shrink=0.75)
    save_figure(fig, out["png"] / "window_reprojection_example_panel.png", cfg)
    return [
        str(out["csv"] / "window_reprojection_example_panel_data.csv"),
        str(out["png"] / "window_reprojection_example_panel.png"),
    ]


def run_l3_v2(cfg: RunConfig) -> StageResult:
    create_results_layout_v2(cfg)
    result = legacy_run_l3(cfg)
    out = stage_dirs(cfg, "L3")

    fill_df = pd.read_csv(out["root"] / "fill_ratio_by_year.csv")
    write_plot_data(fill_df, out["csv"] / "fill_ratio_trend_by_year_plot_data.csv", cfg)

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    for variable, sub in fill_df.groupby("variable"):
        ax.plot(sub["year"], sub["fill_ratio"], marker="o", label=variable)
    ax.set_title("Fill Ratio by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Fill ratio")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, out["png"] / "fill_ratio_trend_by_year.png", cfg)

    reproj_df = pd.read_csv(out["root"] / "window_reprojection_check.csv")
    write_plot_data(reproj_df, out["csv"] / "reprojection_mae_distribution_plot_data.csv", cfg)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    data = []
    labels = []
    for variable, sub in reproj_df.groupby("variable"):
        vals = pd.to_numeric(sub["mae"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size:
            data.append(vals)
            labels.append(variable)
    if data:
        ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title("Reprojection MAE Distribution")
    ax.set_xlabel("Variable")
    ax.set_ylabel("MAE")
    ax.grid(alpha=0.2)
    save_figure(fig, out["png"] / "reprojection_mae_distribution.png", cfg)

    extra = _make_example_panel(cfg, reproj_df, out)

    for artifact in [
        out["csv"] / "fill_ratio_trend_by_year_plot_data.csv",
        out["csv"] / "reprojection_mae_distribution_plot_data.csv",
        out["png"] / "fill_ratio_trend_by_year.png",
        out["png"] / "reprojection_mae_distribution.png",
    ] + [Path(x) for x in extra]:
        result.artifacts.append(str(artifact))
    return result
