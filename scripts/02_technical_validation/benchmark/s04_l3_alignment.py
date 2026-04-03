from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from s00_core_common import (
    RunConfig,
    StageResult,
    build_run_config,
    create_results_layout,
    get_stage_paths,
    make_shared_parser,
    open_netcdf_readonly,
    status_from_thresholds,
    write_csv,
    write_text,
)

OUT_N = 41
AGG_FACTOR = 10
PATCH_N = OUT_N * AGG_FACTOR
HALF_PATCH = PATCH_N // 2
FILL_VALUE = -9999.0


@dataclass
class RasterSpec:
    var_name: str
    path: str
    method: str  # sum or mean


def _missing_mask(arr: np.ndarray) -> np.ndarray:
    return (~np.isfinite(arr)) | (arr == FILL_VALUE)


def _aggregate_patch(patch: np.ndarray, nodata: float | None, method: str, flip: bool = True) -> np.ndarray:
    if patch.shape != (PATCH_N, PATCH_N):
        return np.full((OUT_N, OUT_N), FILL_VALUE, dtype=np.float32)

    valid = np.isfinite(patch)
    if nodata is not None:
        if np.isnan(nodata):
            valid &= ~np.isnan(patch)
        else:
            valid &= patch != nodata

    cleaned = np.where(valid, patch, 0.0)
    reshaped = cleaned.reshape(OUT_N, AGG_FACTOR, OUT_N, AGG_FACTOR)
    valid_reshaped = valid.reshape(OUT_N, AGG_FACTOR, OUT_N, AGG_FACTOR)

    sums = reshaped.sum(axis=(1, 3), dtype=np.float64)
    valid_counts = valid_reshaped.sum(axis=(1, 3))

    if method == "sum":
        out = sums.astype(np.float32)
    elif method == "mean":
        out = np.full((OUT_N, OUT_N), FILL_VALUE, dtype=np.float32)
        np.divide(sums, valid_counts, out=out, where=valid_counts > 0)
    else:
        raise ValueError(f"unsupported aggregation method: {method}")

    out[valid_counts == 0] = FILL_VALUE
    if flip:
        out = out[::-1, :]
    return out


def _year_raster_specs(cfg: RunConfig, year: int) -> list[RasterSpec]:
    root = cfg.processed_root
    return [
        RasterSpec(
            var_name="litpop",
            path=str(root / "litpop_2000_2024_v1" / "01_litpop_annual" / f"litpop_{year}_30as.tif"),
            method="sum",
        ),
        RasterSpec(
            var_name="nightlight_intensity",
            path=str(root / "nightlight_harmonization_v1" / "09_harmonized_ntl" / f"ntl_harmonized_{year}_dn_30as.tif"),
            method="mean",
        ),
        RasterSpec(
            var_name="population_count",
            path=str(root / "population_log_pchip_v1" / "01_population_annual" / f"population_{year}_30as.tif"),
            method="sum",
        ),
    ]


def _sample_group_time_indices(ds, n_samples: int, rng: np.random.Generator) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    for gname, grp in ds.groups.items():
        if "time" not in grp.dimensions:
            continue
        n = len(grp.dimensions["time"])
        for idx in range(n):
            pairs.append((gname, idx))
    if not pairs:
        return []
    if len(pairs) <= n_samples:
        return pairs
    selected_idx = rng.choice(len(pairs), size=n_samples, replace=False)
    return [pairs[int(i)] for i in selected_idx]


def _fill_ratio_for_year(cfg: RunConfig, year: int) -> list[dict[str, object]]:
    p = get_stage_paths(cfg, year)
    rows: list[dict[str, object]] = []
    with open_netcdf_readonly(p["poplight"], cfg) as ds:
        for var in ["litpop", "nightlight_intensity", "population_count"]:
            fill_cells = 0
            total_cells = 0
            for grp in ds.groups.values():
                if var not in grp.variables:
                    continue
                arr = np.asarray(grp.variables[var][:], dtype=np.float32)
                miss = _missing_mask(arr)
                fill_cells += int(np.count_nonzero(miss))
                total_cells += int(arr.size)
            ratio = (fill_cells / total_cells) if total_cells else np.nan
            rows.append(
                {
                    "year": year,
                    "variable": var,
                    "fill_cells": int(fill_cells),
                    "total_cells": int(total_cells),
                    "fill_ratio": float(ratio),
                    "warn_3sigma": 0,
                    "baseline_mean": np.nan,
                    "baseline_std": np.nan,
                    "status": "PASS",
                }
            )
    return rows


def _reprojection_for_year(cfg: RunConfig, year: int) -> list[dict[str, object]]:
    p = get_stage_paths(cfg, year)
    rng = np.random.default_rng(cfg.seed + int(year))
    rows: list[dict[str, object]] = []

    with open_netcdf_readonly(p["poplight"], cfg) as ds:
        samples = _sample_group_time_indices(ds, cfg.l3_samples_per_year, rng)
        specs = _year_raster_specs(cfg, year)

        rasters = {}
        for spec in specs:
            rp = Path(spec.path)
            if not rp.exists():
                # record failure rows for this variable
                for sid, (gname, tidx) in enumerate(samples):
                    rows.append(
                        {
                            "year": year,
                            "sample_id": sid,
                            "group": gname,
                            "time_idx": int(tidx),
                            "variable": spec.var_name,
                            "mae": np.nan,
                            "rmse": np.nan,
                            "max_abs": np.nan,
                            "valid_cells": 0,
                            "orientation_mae_flip": np.nan,
                            "orientation_mae_no_flip": np.nan,
                            "orientation_pass": 0,
                            "status": "FAIL",
                            "note": f"missing_raster:{rp}",
                        }
                    )
                continue
            rasters[spec.var_name] = rasterio.open(rp)

        try:
            for sid, (gname, tidx) in enumerate(samples):
                grp = ds.groups[gname]
                lat = float(grp.variables["center_lat"][tidx])
                lon = float(grp.variables["center_lon"][tidx])
                coord_invalid = (not np.isfinite(lat)) or (not np.isfinite(lon)) or (lat < -90) or (lat > 90) or (lon < -180) or (lon > 180)

                for spec in specs:
                    if spec.var_name not in rasters:
                        continue
                    rs = rasters[spec.var_name]

                    if coord_invalid:
                        rows.append(
                            {
                                "year": year,
                                "sample_id": sid,
                                "group": gname,
                                "time_idx": int(tidx),
                                "variable": spec.var_name,
                                "mae": np.nan,
                                "rmse": np.nan,
                                "max_abs": np.nan,
                                "valid_cells": 0,
                                "orientation_mae_flip": np.nan,
                                "orientation_mae_no_flip": np.nan,
                                "orientation_pass": 0,
                                "status": "FAIL",
                                "note": "invalid_center_coord",
                            }
                        )
                        continue

                    row, col = rs.index(lon, lat)
                    patch = rs.read(
                        1,
                        window=Window(col - HALF_PATCH, row - HALF_PATCH, PATCH_N, PATCH_N),
                        boundless=True,
                        fill_value=rs.nodata,
                    ).astype(np.float32)

                    recalc_flip = _aggregate_patch(patch, rs.nodata, spec.method, flip=True)
                    recalc_no_flip = _aggregate_patch(patch, rs.nodata, spec.method, flip=False)
                    nc_slice = np.asarray(grp.variables[spec.var_name][tidx, :, :], dtype=np.float32)

                    valid_flip = (~_missing_mask(recalc_flip)) & (~_missing_mask(nc_slice))
                    valid_no_flip = (~_missing_mask(recalc_no_flip)) & (~_missing_mask(nc_slice))

                    if np.any(valid_flip):
                        d = recalc_flip[valid_flip] - nc_slice[valid_flip]
                        mae = float(np.mean(np.abs(d)))
                        rmse = float(np.sqrt(np.mean(d * d)))
                        max_abs = float(np.max(np.abs(d)))
                        valid_cells = int(np.count_nonzero(valid_flip))
                    else:
                        mae = np.nan
                        rmse = np.nan
                        max_abs = np.nan
                        valid_cells = 0

                    if np.any(valid_flip):
                        mae_flip = float(np.mean(np.abs(recalc_flip[valid_flip] - nc_slice[valid_flip])))
                    else:
                        mae_flip = np.nan
                    if np.any(valid_no_flip):
                        mae_no_flip = float(np.mean(np.abs(recalc_no_flip[valid_no_flip] - nc_slice[valid_no_flip])))
                    else:
                        mae_no_flip = np.nan

                    orientation_pass = int(np.isfinite(mae_flip) and np.isfinite(mae_no_flip) and mae_flip <= mae_no_flip)
                    status = "PASS"
                    note = ""
                    if np.isfinite(mae) and mae > 1e-5:
                        status = "FAIL"
                        note = "mae_above_threshold"

                    rows.append(
                        {
                            "year": year,
                            "sample_id": sid,
                            "group": gname,
                            "time_idx": int(tidx),
                            "variable": spec.var_name,
                            "mae": mae,
                            "rmse": rmse,
                            "max_abs": max_abs,
                            "valid_cells": valid_cells,
                            "orientation_mae_flip": mae_flip,
                            "orientation_mae_no_flip": mae_no_flip,
                            "orientation_pass": orientation_pass,
                            "status": status,
                            "note": note,
                        }
                    )
        finally:
            for rs in rasters.values():
                rs.close()

    return rows


def run_l3(cfg: RunConfig) -> StageResult:
    layout = create_results_layout(cfg)
    out_dir = layout["L3"]

    fill_rows: list[dict[str, object]] = []
    reproj_rows: list[dict[str, object]] = []

    for year in cfg.years:
        fill_rows.extend(_fill_ratio_for_year(cfg, year))
        reproj_rows.extend(_reprojection_for_year(cfg, year))

    fill_df = pd.DataFrame(fill_rows)
    reproj_df = pd.DataFrame(reproj_rows)

    # Baseline warnings: fill_ratio > mean + 3sigma (per variable across years)
    for var in sorted(fill_df["variable"].dropna().unique()):
        m = fill_df["variable"] == var
        vals = pd.to_numeric(fill_df.loc[m, "fill_ratio"], errors="coerce")
        mu = float(vals.mean(skipna=True))
        sigma = float(vals.std(skipna=True, ddof=0))
        fill_df.loc[m, "baseline_mean"] = mu
        fill_df.loc[m, "baseline_std"] = sigma
        if np.isfinite(mu) and np.isfinite(sigma) and sigma > 0:
            warn_mask = vals > (mu + 3.0 * sigma)
            fill_df.loc[m, "warn_3sigma"] = warn_mask.astype(int)

    fill_df.loc[fill_df["warn_3sigma"] > 0, "status"] = "WARN"

    # L3 fail threshold on reprojection mae
    fail_mask = pd.to_numeric(reproj_df.get("mae", pd.Series(dtype=float)), errors="coerce") > 1e-5
    reproj_df.loc[fail_mask.fillna(False), "status"] = "FAIL"

    fill_ratio_path = out_dir / "fill_ratio_by_year.csv"
    reproj_path = out_dir / "window_reprojection_check.csv"
    summary_path = out_dir / "alignment_summary.md"

    write_csv(fill_df, fill_ratio_path, cfg)
    write_csv(reproj_df, reproj_path, cfg)

    fail_count = int((reproj_df["status"] == "FAIL").sum())
    warn_count = int((fill_df["status"] == "WARN").sum())

    lines: list[str] = []
    lines.append("# L3 Alignment Summary")
    lines.append("")
    lines.append(f"- years: {cfg.years[0]}-{cfg.years[-1]}")
    lines.append(f"- samples_per_year: {cfg.l3_samples_per_year}")
    lines.append(f"- reprojection_fail_count (MAE>1e-5): {fail_count}")
    lines.append(f"- fill_ratio_warn_count (>baseline+3sigma): {warn_count}")
    lines.append("")

    lines.append("## Fill Ratio")
    lines.append("")
    lines.append("| year | variable | fill_ratio | baseline_mean | baseline_std | status |")
    lines.append("|---:|---|---:|---:|---:|---|")
    for r in fill_df.sort_values(["year", "variable"]).itertuples(index=False):
        lines.append(
            f"| {int(r.year)} | {r.variable} | {float(r.fill_ratio):.6f} | {float(r.baseline_mean):.6f} | {float(r.baseline_std):.6f} | {r.status} |"
        )

    lines.append("")
    lines.append("## Reprojection Metrics")
    lines.append("")
    by_var = (
        reproj_df.groupby("variable", dropna=False)
        .agg(samples=("variable", "size"), mae_mean=("mae", "mean"), mae_max=("mae", "max"), fail_count=("status", lambda s: int((s == "FAIL").sum())))
        .reset_index()
    )
    lines.append("| variable | samples | mae_mean | mae_max | fail_count |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in by_var.itertuples(index=False):
        lines.append(f"| {r.variable} | {int(r.samples)} | {float(r.mae_mean):.8f} | {float(r.mae_max):.8f} | {int(r.fail_count)} |")

    write_text("\n".join(lines) + "\n", summary_path, cfg)

    status = status_from_thresholds(fail_count=fail_count, warn_count=warn_count)
    return StageResult(
        level="L3",
        status=status,
        fail_count=fail_count,
        warn_count=warn_count,
        metrics={
            "fill_rows": int(len(fill_df)),
            "reprojection_rows": int(len(reproj_df)),
            "fail_count": int(fail_count),
            "warn_count": int(warn_count),
        },
        artifacts=[str(fill_ratio_path), str(reproj_path), str(summary_path)],
        notes=[],
    )


def main() -> int:
    parser = make_shared_parser("L3 alignment validation")
    args = parser.parse_args()
    cfg = build_run_config(args)
    result = run_l3(cfg)
    print(f"L3 status={result.status} fail={result.fail_count} warn={result.warn_count}")
    if cfg.strict and result.status == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
