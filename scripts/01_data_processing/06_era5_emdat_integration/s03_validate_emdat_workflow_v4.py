#!/usr/bin/env python3
"""
Phase 3: validation for EM-DAT integration workflow.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from s00_common import INPUT_NC_DIR, LOG_DIR, OUTPUT_AUDIT_DIR, OUTPUT_NC_DIR, parse_years_expr, setup_logger


NEW_FLOAT_VARS = [
    "era5_u10_daily_max",
    "era5_v10_daily_max",
    "era5_wind_speed_daily_proxy",
    "era5_tp_daily_sum_mm",
    "hazard_wind_power_daily",
    "hazard_rain_daily",
    "hazard_compound_daily",
    "emdat_loss_allocated_usd",
    "emdat_loss_weight_norm",
]
NEW_INT_VARS = ["emdat_match_count"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate EM-DAT integration workflow outputs.")
    p.add_argument("--years", type=str, default="2013")
    p.add_argument("--output-nc-dir", type=Path, default=OUTPUT_NC_DIR)
    p.add_argument("--audit-dir", type=Path, default=OUTPUT_AUDIT_DIR)
    p.add_argument("--input-nc-dir", type=Path, default=INPUT_NC_DIR)
    p.add_argument("--log-level", type=str, default="INFO")
    p.add_argument("--max-error", type=float, default=1e-6)
    return p.parse_args()


def validate_file_structure(year: int, output_path: Path, input_path: Path, logger) -> bool:
    ok = True
    if not output_path.exists():
        logger.error("[%s] missing output file: %s", year, output_path)
        return False
    if not input_path.exists():
        logger.error("[%s] missing input reference file: %s", year, input_path)
        return False

    with Dataset(input_path, "r") as src, Dataset(output_path, "r") as out:
        if len(src.groups) != len(out.groups):
            logger.error("[%s] group count mismatch: src=%s out=%s", year, len(src.groups), len(out.groups))
            ok = False

        for gname, src_grp in src.groups.items():
            if gname not in out.groups:
                logger.error("[%s] missing group in output: %s", year, gname)
                ok = False
                continue
            out_grp = out.groups[gname]

            # Existing var shape/dtype unchanged
            for vname, src_var in src_grp.variables.items():
                if vname not in out_grp.variables:
                    logger.error("[%s] group=%s missing original var: %s", year, gname, vname)
                    ok = False
                    continue
                out_var = out_grp.variables[vname]
                if src_var.dimensions != out_var.dimensions:
                    logger.error("[%s] group=%s var=%s dims changed", year, gname, vname)
                    ok = False
                if str(src_var.dtype) != str(out_var.dtype):
                    logger.error("[%s] group=%s var=%s dtype changed", year, gname, vname)
                    ok = False

            # New vars exist
            for vname in NEW_FLOAT_VARS:
                if vname not in out_grp.variables:
                    logger.error("[%s] group=%s missing new float var: %s", year, gname, vname)
                    ok = False
                    continue
                v = out_grp.variables[vname]
                if v.dimensions != ("time", "window_lat", "window_lon"):
                    logger.error("[%s] group=%s var=%s unexpected dims=%s", year, gname, vname, v.dimensions)
                    ok = False
                if str(v.dtype) != "float32":
                    logger.error("[%s] group=%s var=%s unexpected dtype=%s", year, gname, vname, v.dtype)
                    ok = False

            for vname in NEW_INT_VARS:
                if vname not in out_grp.variables:
                    logger.error("[%s] group=%s missing new int var: %s", year, gname, vname)
                    ok = False
                    continue
                v = out_grp.variables[vname]
                if v.dimensions != ("time", "window_lat", "window_lon"):
                    logger.error("[%s] group=%s var=%s unexpected dims=%s", year, gname, vname, v.dimensions)
                    ok = False
                if str(v.dtype) not in {"int16", "i2"}:
                    logger.error("[%s] group=%s var=%s unexpected dtype=%s", year, gname, vname, v.dtype)
                    ok = False

    return ok


def validate_audit(year: int, audit_dir: Path, max_error: float, logger) -> bool:
    ok = True
    match_csv = audit_dir / f"emdat_record_match_{year}.csv"
    summary_csv = audit_dir / f"emdat_record_allocation_summary_{year}.csv"
    if not match_csv.exists():
        logger.error("[%s] missing match CSV: %s", year, match_csv)
        ok = False
    if not summary_csv.exists():
        logger.error("[%s] missing summary CSV: %s", year, summary_csv)
        ok = False
    if not ok:
        return False

    match_df = pd.read_csv(match_csv)
    summary_df = pd.read_csv(summary_csv)

    required_match_cols = {
        "DisNo",
        "ISO",
        "EventName",
        "candidate_groups",
        "selected_groups",
        "match_grade",
        "spatial_source",
        "status_prealloc",
    }
    required_sum_cols = {
        "DisNo",
        "ISO",
        "EventName",
        "L_usd",
        "allocated_usd",
        "conservation_error",
        "status",
        "match_grade",
        "spatial_source",
    }
    miss_m = sorted(required_match_cols - set(match_df.columns))
    miss_s = sorted(required_sum_cols - set(summary_df.columns))
    if miss_m:
        logger.error("[%s] match CSV missing columns: %s", year, miss_m)
        ok = False
    if miss_s:
        logger.error("[%s] summary CSV missing columns: %s", year, miss_s)
        ok = False

    if "status" in summary_df.columns:
        status_counts = Counter(summary_df["status"].astype(str))
        logger.info("[%s] status counts: %s", year, dict(status_counts))
        for key in ["allocated", "missing_loss", "unmatched", "zero_weight"]:
            status_counts.get(key, 0)

    if "conservation_error" in summary_df.columns and "status" in summary_df.columns:
        alloc = summary_df[summary_df["status"].astype(str) == "allocated"].copy()
        if not alloc.empty:
            alloc["conservation_error"] = pd.to_numeric(alloc["conservation_error"], errors="coerce")
            bad = alloc[alloc["conservation_error"] > max_error]
            if not bad.empty:
                logger.error("[%s] conservation errors above threshold (%s): %s records", year, max_error, len(bad))
                ok = False
            else:
                logger.info("[%s] conservation check passed for allocated records (%s)", year, len(alloc))
        else:
            logger.warning("[%s] no allocated records in summary CSV", year)

    return ok


def main() -> int:
    args = parse_args()
    years = parse_years_expr(args.years)
    logger = setup_logger("emdat_phase10_validate", LOG_DIR / "10_validate_emdat_workflow_v4.log", args.log_level)
    logger.info("Phase 3 validation started | years=%s", years)

    all_ok = True
    for year in years:
        out_path = args.output_nc_dir / f"typhoon_{year}_ocean_litpop_poplight_emdat.nc"
        src_path = args.input_nc_dir / f"typhoon_{year}_ocean_litpop_poplight.nc"
        ok_structure = validate_file_structure(year, out_path, src_path, logger)
        ok_audit = validate_audit(year, args.audit_dir, args.max_error, logger)
        year_ok = ok_structure and ok_audit
        logger.info("[%s] validation result: %s", year, "PASS" if year_ok else "FAIL")
        all_ok = all_ok and year_ok

    logger.info("Phase 3 validation finished | success=%s", all_ok)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
