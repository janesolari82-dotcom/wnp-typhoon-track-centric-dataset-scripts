from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

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


def _minmax_normalize_2d(arr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.float32)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return out
    v = arr[valid]
    vmin = float(v.min())
    vmax = float(v.max())
    if vmax <= vmin:
        return out
    out[valid] = ((arr[valid] - vmin) / (vmax - vmin)).astype(np.float32)
    return out


def _year_formula_check(cfg: RunConfig, year: int) -> dict[str, object]:
    p = get_stage_paths(cfg, year)

    max_err_wind = 0.0
    max_err_hw = 0.0
    max_err_hc = 0.0

    sum_err_wind = 0.0
    sum_err_hw = 0.0
    sum_err_hc = 0.0

    count_err_wind = 0
    count_err_hw = 0
    count_err_hc = 0

    tp_negative_count = 0
    hazard_negative_count = 0
    hc_out_of_range_count = 0
    missing_var_fail = 0

    with open_netcdf_readonly(p["final"], cfg) as ds:
        for gname, grp in ds.groups.items():
            required = [
                "era5_u10_daily_max",
                "era5_v10_daily_max",
                "era5_wind_speed_daily_proxy",
                "era5_tp_daily_sum_mm",
                "hazard_wind_power_daily",
                "hazard_rain_daily",
                "hazard_compound_daily",
            ]
            if any(v not in grp.variables for v in required):
                missing_var_fail += 1
                continue

            u = np.asarray(grp.variables["era5_u10_daily_max"][:], dtype=np.float32)
            v = np.asarray(grp.variables["era5_v10_daily_max"][:], dtype=np.float32)
            w_ref = np.asarray(grp.variables["era5_wind_speed_daily_proxy"][:], dtype=np.float32)
            tp = np.asarray(grp.variables["era5_tp_daily_sum_mm"][:], dtype=np.float32)
            hw_ref = np.asarray(grp.variables["hazard_wind_power_daily"][:], dtype=np.float32)
            hr_ref = np.asarray(grp.variables["hazard_rain_daily"][:], dtype=np.float32)
            hc_ref = np.asarray(grp.variables["hazard_compound_daily"][:], dtype=np.float32)

            w = np.sqrt(np.maximum(u * u + v * v, 0.0), dtype=np.float32)
            hw = np.power(np.maximum(w - np.float32(12.0), np.float32(0.0)), np.float32(3.0), dtype=np.float32)

            hc = np.zeros_like(hc_ref, dtype=np.float32)
            for t in range(hc.shape[0]):
                hc[t, :, :] = 0.5 * _minmax_normalize_2d(hw[t, :, :]) + 0.5 * _minmax_normalize_2d(hr_ref[t, :, :])

            ew = np.abs(w - w_ref)
            ehw = np.abs(hw - hw_ref)
            ehc = np.abs(hc - hc_ref)

            valid_w = np.isfinite(ew)
            valid_hw = np.isfinite(ehw)
            valid_hc = np.isfinite(ehc)

            if np.any(valid_w):
                max_err_wind = max(max_err_wind, float(np.max(ew[valid_w])))
                sum_err_wind += float(np.sum(ew[valid_w], dtype=np.float64))
                count_err_wind += int(np.count_nonzero(valid_w))
            if np.any(valid_hw):
                max_err_hw = max(max_err_hw, float(np.max(ehw[valid_hw])))
                sum_err_hw += float(np.sum(ehw[valid_hw], dtype=np.float64))
                count_err_hw += int(np.count_nonzero(valid_hw))
            if np.any(valid_hc):
                max_err_hc = max(max_err_hc, float(np.max(ehc[valid_hc])))
                sum_err_hc += float(np.sum(ehc[valid_hc], dtype=np.float64))
                count_err_hc += int(np.count_nonzero(valid_hc))

            tp_negative_count += int(np.count_nonzero(np.isfinite(tp) & (tp < 0)))
            hazard_negative_count += int(np.count_nonzero(np.isfinite(hw_ref) & (hw_ref < 0)))
            hazard_negative_count += int(np.count_nonzero(np.isfinite(hr_ref) & (hr_ref < 0)))
            hazard_negative_count += int(np.count_nonzero(np.isfinite(hc_ref) & (hc_ref < 0)))
            hc_out_of_range_count += int(np.count_nonzero(np.isfinite(hc_ref) & ((hc_ref < 0) | (hc_ref > 1))))

    mean_err_wind = (sum_err_wind / count_err_wind) if count_err_wind else np.nan
    mean_err_hw = (sum_err_hw / count_err_hw) if count_err_hw else np.nan
    mean_err_hc = (sum_err_hc / count_err_hc) if count_err_hc else np.nan

    fail_flag = (
        (max_err_wind > 1e-6)
        or (max_err_hw > 1e-6)
        or (max_err_hc > 1e-6)
        or (tp_negative_count > 0)
        or (hazard_negative_count > 0)
        or (hc_out_of_range_count > 0)
        or (missing_var_fail > 0)
    )

    return {
        "year": year,
        "max_abs_err_wind": float(max_err_wind),
        "max_abs_err_hazard_wind_power": float(max_err_hw),
        "max_abs_err_hazard_compound": float(max_err_hc),
        "mean_abs_err_wind": float(mean_err_wind),
        "mean_abs_err_hazard_wind_power": float(mean_err_hw),
        "mean_abs_err_hazard_compound": float(mean_err_hc),
        "tp_negative_count": int(tp_negative_count),
        "hazard_negative_count": int(hazard_negative_count),
        "hazard_compound_out_of_range_count": int(hc_out_of_range_count),
        "missing_var_fail_count": int(missing_var_fail),
        "status": "FAIL" if fail_flag else "PASS",
    }


def run_l4(cfg: RunConfig) -> StageResult:
    layout = create_results_layout(cfg)
    out_dir = layout["L4"]

    rows = [_year_formula_check(cfg, y) for y in cfg.years]
    df = pd.DataFrame(rows)

    formula_path = out_dir / "formula_recompute_error.csv"
    range_path = out_dir / "hazard_value_range_report.md"

    write_csv(df, formula_path, cfg)

    fail_count = int((df["status"] == "FAIL").sum())
    warn_count = 0

    lines: list[str] = []
    lines.append("# L4 Hazard Formula Validation Report")
    lines.append("")
    lines.append("Thresholds:")
    lines.append("- max absolute recompute error <= 1e-6")
    lines.append("- tp >= 0")
    lines.append("- hazard_* >= 0")
    lines.append("- hazard_compound in [0, 1]")
    lines.append("")
    lines.append("| year | max_err_wind | max_err_hw | max_err_hc | tp_neg | hazard_neg | hc_out_range | missing_var | status |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in df.sort_values("year").itertuples(index=False):
        lines.append(
            "| {year} | {e1:.8e} | {e2:.8e} | {e3:.8e} | {tp} | {hn} | {hc} | {mv} | {st} |".format(
                year=int(r.year),
                e1=float(r.max_abs_err_wind),
                e2=float(r.max_abs_err_hazard_wind_power),
                e3=float(r.max_abs_err_hazard_compound),
                tp=int(r.tp_negative_count),
                hn=int(r.hazard_negative_count),
                hc=int(r.hazard_compound_out_of_range_count),
                mv=int(r.missing_var_fail_count),
                st=r.status,
            )
        )

    write_text("\n".join(lines) + "\n", range_path, cfg)

    status = status_from_thresholds(fail_count=fail_count, warn_count=warn_count)
    return StageResult(
        level="L4",
        status=status,
        fail_count=fail_count,
        warn_count=warn_count,
        metrics={
            "years_checked": int(len(df)),
            "failed_years": int(fail_count),
        },
        artifacts=[str(formula_path), str(range_path)],
        notes=[],
    )


def main() -> int:
    parser = make_shared_parser("L4 hazard formula validation")
    args = parser.parse_args()
    cfg = build_run_config(args)
    result = run_l4(cfg)
    print(f"L4 status={result.status} fail={result.fail_count} warn={result.warn_count}")
    if cfg.strict and result.status == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
