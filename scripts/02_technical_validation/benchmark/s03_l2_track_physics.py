from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from s00_core_common import (
    RunConfig,
    StageResult,
    build_run_config,
    corr_safe,
    create_results_layout,
    get_stage_paths,
    make_shared_parser,
    open_netcdf_readonly,
    quantile_safe,
    status_from_thresholds,
    valid_mask_default,
    write_csv,
    write_text,
)

TRACK_VARS = ["center_lat", "center_lon", "wind_speed", "central_pressure", "radius_max_wind"]


def _physical_invalid_count(var_name: str, arr: np.ndarray, valid_mask: np.ndarray) -> int:
    if arr.size == 0:
        return 0
    v = arr[valid_mask]
    if v.size == 0:
        return 0

    if var_name == "center_lat":
        bad = (v < -90.0) | (v > 90.0)
    elif var_name == "center_lon":
        bad = (v < -180.0) | (v > 180.0)
    elif var_name == "wind_speed":
        bad = v < 0.0
    elif var_name == "central_pressure":
        bad = (v < 850.0) | (v > 1100.0)
    elif var_name == "radius_max_wind":
        bad = v < 0.0
    else:
        bad = np.zeros(v.shape, dtype=bool)
    return int(np.count_nonzero(bad))


def _compute_year_rows(cfg: RunConfig, year: int) -> list[dict[str, object]]:
    paths = get_stage_paths(cfg, year)
    rows: list[dict[str, object]] = []

    with open_netcdf_readonly(paths["base"], cfg) as ds_base, open_netcdf_readonly(paths["final"], cfg) as ds_final:
        groups = sorted(set(ds_base.groups.keys()) & set(ds_final.groups.keys()))

        for var in TRACK_VARS:
            base_chunks: list[np.ndarray] = []
            final_chunks: list[np.ndarray] = []
            total_count = 0

            for g in groups:
                gb = ds_base.groups[g]
                gf = ds_final.groups[g]
                if var not in gb.variables or var not in gf.variables:
                    continue

                b = np.asarray(gb.variables[var][:], dtype=np.float64).reshape(-1)
                f = np.asarray(gf.variables[var][:], dtype=np.float64).reshape(-1)
                n = min(b.size, f.size)
                if n == 0:
                    continue
                b = b[:n]
                f = f[:n]
                base_chunks.append(b)
                final_chunks.append(f)
                total_count += int(n)

            if total_count == 0:
                rows.append(
                    {
                        "year": year,
                        "variable": var,
                        "total_count": 0,
                        "valid_pairs": 0,
                        "base_mean": np.nan,
                        "base_std": np.nan,
                        "base_p5": np.nan,
                        "base_p50": np.nan,
                        "base_p95": np.nan,
                        "final_mean": np.nan,
                        "final_std": np.nan,
                        "final_p5": np.nan,
                        "final_p50": np.nan,
                        "final_p95": np.nan,
                        "bias": np.nan,
                        "rmse": np.nan,
                        "r": np.nan,
                        "missing_ratio": np.nan,
                        "illegal_count": 0,
                        "illegal_rate": np.nan,
                        "drift_z": np.nan,
                        "drift_warn": 0,
                        "status": "FAIL",
                        "note": "no data",
                    }
                )
                continue

            base_vals = np.concatenate(base_chunks)
            final_vals = np.concatenate(final_chunks)

            base_valid = valid_mask_default(base_vals)
            final_valid = valid_mask_default(final_vals)
            pair_valid = base_valid & final_valid

            base_v = base_vals[base_valid]
            final_v = final_vals[final_valid]

            illegal_count = _physical_invalid_count(var, final_vals, final_valid)
            illegal_rate = float(illegal_count / max(int(np.count_nonzero(final_valid)), 1))
            missing_ratio = float(1.0 - (np.count_nonzero(final_valid) / max(total_count, 1)))

            if np.count_nonzero(pair_valid) > 0:
                diff = final_vals[pair_valid] - base_vals[pair_valid]
                bias = float(np.mean(diff))
                rmse = float(np.sqrt(np.mean(diff * diff)))
                r = corr_safe(base_vals[pair_valid], final_vals[pair_valid])
            else:
                bias = np.nan
                rmse = np.nan
                r = np.nan

            row = {
                "year": year,
                "variable": var,
                "total_count": int(total_count),
                "valid_pairs": int(np.count_nonzero(pair_valid)),
                "base_mean": float(np.nanmean(base_v)) if base_v.size else np.nan,
                "base_std": float(np.nanstd(base_v)) if base_v.size else np.nan,
                "base_p5": quantile_safe(base_v, 0.05),
                "base_p50": quantile_safe(base_v, 0.50),
                "base_p95": quantile_safe(base_v, 0.95),
                "final_mean": float(np.nanmean(final_v)) if final_v.size else np.nan,
                "final_std": float(np.nanstd(final_v)) if final_v.size else np.nan,
                "final_p5": quantile_safe(final_v, 0.05),
                "final_p50": quantile_safe(final_v, 0.50),
                "final_p95": quantile_safe(final_v, 0.95),
                "bias": bias,
                "rmse": rmse,
                "r": r,
                "missing_ratio": missing_ratio,
                "illegal_count": int(illegal_count),
                "illegal_rate": illegal_rate,
                "drift_z": np.nan,
                "drift_warn": 0,
                "status": "PASS",
                "note": "",
            }
            rows.append(row)

    return rows


def run_l2(cfg: RunConfig) -> StageResult:
    layout = create_results_layout(cfg)
    out_dir = layout["L2"]
    summary_path = out_dir / "track_physics_summary.md"

    all_rows: list[dict[str, object]] = []
    for year in cfg.years:
        all_rows.extend(_compute_year_rows(cfg, year))

    df = pd.DataFrame(all_rows)

    # 3-sigma drift on final_mean (per variable, across years)
    for var in TRACK_VARS:
        m = df["variable"] == var
        series = pd.to_numeric(df.loc[m, "final_mean"], errors="coerce")
        mu = float(series.mean(skipna=True))
        sigma = float(series.std(skipna=True, ddof=0))
        if np.isfinite(mu) and np.isfinite(sigma) and sigma > 0:
            z = (series - mu) / sigma
            df.loc[m, "drift_z"] = z
            df.loc[m, "drift_warn"] = (z.abs() > 3.0).astype(int)

    # Status per row
    df.loc[df["illegal_count"] > 0, "status"] = "FAIL"
    df.loc[(df["status"] == "PASS") & (df["drift_warn"] > 0), "status"] = "WARN"

    fail_count = int((df["status"] == "FAIL").sum())
    warn_count = int((df["status"] == "WARN").sum())

    # Per-year files
    for year in cfg.years:
        ydf = df[df["year"] == year].copy().reset_index(drop=True)
        write_csv(ydf, out_dir / f"track_physics_metrics_{year}.csv", cfg)

    # Summary markdown
    lines: list[str] = []
    lines.append("# L2 Track Physics Summary")
    lines.append("")
    lines.append(f"- years: {cfg.years[0]}-{cfg.years[-1]}")
    lines.append(f"- fail_rows: {fail_count}")
    lines.append(f"- warn_rows: {warn_count}")
    lines.append("")

    by_year_status = (
        df.groupby(["year", "status"], dropna=False).size().rename("count").reset_index().sort_values(["year", "status"])
    )
    lines.append("## Row Status By Year")
    lines.append("")
    lines.append("| year | status | count |")
    lines.append("|---:|---|---:|")
    for r in by_year_status.itertuples(index=False):
        lines.append(f"| {int(r.year)} | {r.status} | {int(r.count)} |")

    lines.append("")
    lines.append("## Physical Constraint Rules")
    lines.append("")
    lines.append("- center_lat in [-90, 90]")
    lines.append("- center_lon in [-180, 180]")
    lines.append("- wind_speed >= 0")
    lines.append("- central_pressure in [850, 1100]")
    lines.append("- radius_max_wind >= 0")

    write_text("\n".join(lines) + "\n", summary_path, cfg)

    status = status_from_thresholds(fail_count=fail_count, warn_count=warn_count)
    artifacts = [str(summary_path)] + [str(out_dir / f"track_physics_metrics_{y}.csv") for y in cfg.years]

    return StageResult(
        level="L2",
        status=status,
        fail_count=fail_count,
        warn_count=warn_count,
        metrics={
            "rows": int(len(df)),
            "fail_rows": int(fail_count),
            "warn_rows": int(warn_count),
        },
        artifacts=artifacts,
        notes=["L2 is internal consistency validation against IBTrACS-derived chain."],
    )


def main() -> int:
    parser = make_shared_parser("L2 track and physics validation")
    args = parser.parse_args()
    cfg = build_run_config(args)
    result = run_l2(cfg)
    print(f"L2 status={result.status} fail={result.fail_count} warn={result.warn_count}")
    if cfg.strict and result.status == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
