#!/usr/bin/env python3
"""
Integrate annual LitPop raster slices into typhoon NetCDF files (2000-2024).

Workflow:
1. Copy each yearly source NetCDF from ocean_integration to lightpop_integration.
2. Add `litpop(time, window_lat, window_lon)` for every typhoon group.
3. For each track point, extract a 410x410 (30as) patch centered on typhoon center.
4. Aggregate to 41x41 by 10x10 sum blocks.
5. Flip latitude axis so output matches window_lat order (- to +).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from netCDF4 import Dataset
from rasterio.windows import Window

# Paths
BASE_DIR = Path(__file__).resolve().parents[3]
INPUT_NC_DIR = BASE_DIR / "03_intermediate_nc" / "ocean_integration"
OUTPUT_NC_DIR = BASE_DIR / "03_intermediate_nc" / "lightpop_integration"
LITPOP_DIR = BASE_DIR / "02_processed_data" / "litpop_2000_2024_v1" / "01_litpop_annual"
LOG_FILE = BASE_DIR / "06_logs" / "03_litpop_integration.log"

# Window config: side length 200 nautical miles ~= 3.333... degrees
# Existing NC uses 41x41 at 0.083333... degrees, i.e., +/-1.666... degrees.
OUT_N = 41
AGG_FACTOR = 10
PATCH_N = OUT_N * AGG_FACTOR  # 410
HALF_PATCH = PATCH_N // 2  # 205

FILL_VALUE = -9999.0
INVALID_INDEX = np.int32(-2_147_483_648)


@dataclass
class YearStats:
    year: int
    typhoons: int = 0
    time_steps: int = 0
    invalid_points: int = 0
    fill_cells: int = 0
    total_cells: int = 0
    elapsed_sec: float = 0.0

    @property
    def fill_ratio(self) -> float:
        return (self.fill_cells / self.total_cells) if self.total_cells else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrate LitPop 41x41 windows into yearly typhoon NetCDF files."
    )
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files (default: skip existing outputs).",
    )
    return parser.parse_args()


def setup_logger() -> logging.Logger:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("litpop_integration")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def is_invalid_coord(lat: float, lon: float) -> bool:
    if not np.isfinite(lat) or not np.isfinite(lon):
        return True
    if lat < -90.0 or lat > 90.0:
        return True
    if lon < -180.0 or lon > 180.0:
        return True
    return False


def aggregate_patch_to_41x41(patch: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    """Aggregate 410x410 patch to 41x41 by 10x10 sum. Return float32 array."""
    if patch.shape != (PATCH_N, PATCH_N):
        return np.full((OUT_N, OUT_N), FILL_VALUE, dtype=np.float32)

    valid = np.isfinite(patch)
    if nodata is not None:
        if np.isnan(nodata):
            valid &= ~np.isnan(patch)
        else:
            valid &= patch != nodata

    cleaned = np.where(valid, patch, 0.0)

    summed = cleaned.reshape(OUT_N, AGG_FACTOR, OUT_N, AGG_FACTOR).sum(
        axis=(1, 3), dtype=np.float64
    )
    valid_count = valid.reshape(OUT_N, AGG_FACTOR, OUT_N, AGG_FACTOR).sum(axis=(1, 3))

    out = summed.astype(np.float32, copy=False)
    out[valid_count == 0] = FILL_VALUE

    # Raster rows are north->south; window_lat in NC is south->north.
    out = out[::-1, :]
    return out


def build_track_index_map(
    ds: Dataset, litpop_ds: rasterio.io.DatasetReader
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, int]:
    """
    Build per-group row/col indices in LitPop raster space.

    Returns:
    - mapping: group -> (rows, cols), each shape (time,)
    - all_valid_rows
    - all_valid_cols
    - invalid_points_count
    """
    mapping: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    valid_rows: List[np.ndarray] = []
    valid_cols: List[np.ndarray] = []
    invalid_points = 0

    for gname, grp in ds.groups.items():
        lats = np.asarray(grp.variables["center_lat"][:], dtype=np.float64)
        lons = np.asarray(grp.variables["center_lon"][:], dtype=np.float64)

        rows = np.full(lats.shape, INVALID_INDEX, dtype=np.int32)
        cols = np.full(lats.shape, INVALID_INDEX, dtype=np.int32)

        for i in range(lats.size):
            lat = float(lats[i])
            lon = float(lons[i])
            if is_invalid_coord(lat, lon):
                invalid_points += 1
                continue
            row, col = litpop_ds.index(lon, lat)
            rows[i] = np.int32(row)
            cols[i] = np.int32(col)

        mapping[gname] = (rows, cols)

        valid_mask = rows != INVALID_INDEX
        if np.any(valid_mask):
            valid_rows.append(rows[valid_mask])
            valid_cols.append(cols[valid_mask])

    if valid_rows:
        all_rows = np.concatenate(valid_rows)
        all_cols = np.concatenate(valid_cols)
    else:
        all_rows = np.array([], dtype=np.int32)
        all_cols = np.array([], dtype=np.int32)

    return mapping, all_rows, all_cols, invalid_points


def ensure_group_litpop_var(grp: Dataset, source_tif_name: str) -> None:
    if "litpop" in grp.variables:
        raise RuntimeError(f"Group {grp.path} already contains variable 'litpop'.")

    v = grp.createVariable(
        "litpop",
        "f4",
        ("time", "window_lat", "window_lon"),
        zlib=True,
        shuffle=True,
        complevel=5,
        fill_value=FILL_VALUE,
        chunksizes=(1, OUT_N, OUT_N),
    )
    v.units = "USD"
    v.long_name = "LitPop exposure aggregated to 0.083333 degree typhoon-relative window"
    v.aggregation = "sum_10x10_30as_to_0.083deg"
    v.source = source_tif_name
    v.missing_value = FILL_VALUE


def process_year(year: int, overwrite: bool, logger: logging.Logger) -> Tuple[str, Optional[YearStats], str]:
    t0 = time.time()
    src_nc = INPUT_NC_DIR / f"typhoon_{year}_ocean.nc"
    src_tif = LITPOP_DIR / f"litpop_{year}_30as.tif"
    dst_nc = OUTPUT_NC_DIR / f"typhoon_{year}_ocean_litpop.nc"

    if not src_nc.exists():
        return "fail", None, f"[{year}] Missing input NC: {src_nc}"
    if not src_tif.exists():
        return "fail", None, f"[{year}] Missing input TIF: {src_tif}"

    if dst_nc.exists():
        if not overwrite:
            return "skip", None, f"[{year}] Output exists, skipped: {dst_nc.name}"
        dst_nc.unlink()

    OUTPUT_NC_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_nc, dst_nc)

    stats = YearStats(year=year)
    nodata: Optional[float] = None
    big_data: Optional[np.ndarray] = None
    row_min = col_min = 0

    try:
        with rasterio.open(src_tif) as litpop_ds:
            nodata = litpop_ds.nodata
            if nodata is None:
                nodata = FILL_VALUE

            with Dataset(dst_nc, "r+") as ds:
                ds.litpop_integration_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ds.litpop_source = src_tif.name

                group_indices, all_rows, all_cols, invalid_points = build_track_index_map(ds, litpop_ds)
                stats.invalid_points = invalid_points
                stats.typhoons = len(ds.groups)
                stats.time_steps = int(
                    sum(len(grp.dimensions["time"]) for grp in ds.groups.values())
                )

                if all_rows.size > 0:
                    row_min = int(all_rows.min()) - HALF_PATCH
                    row_max_exclusive = int(all_rows.max()) + HALF_PATCH
                    col_min = int(all_cols.min()) - HALF_PATCH
                    col_max_exclusive = int(all_cols.max()) + HALF_PATCH

                    height = row_max_exclusive - row_min
                    width = col_max_exclusive - col_min

                    big_data = litpop_ds.read(
                        1,
                        window=Window(col_min, row_min, width, height),
                        boundless=True,
                        fill_value=nodata,
                    ).astype(np.float32, copy=False)

                    logger.info(
                        "[%s] Loaded yearly raster block: shape=%s, window=(row=%s,col=%s,h=%s,w=%s)",
                        year,
                        big_data.shape,
                        row_min,
                        col_min,
                        height,
                        width,
                    )

                fill_slice = np.full((OUT_N, OUT_N), FILL_VALUE, dtype=np.float32)

                for gname, grp in ds.groups.items():
                    ensure_group_litpop_var(grp, src_tif.name)
                    litpop_var = grp.variables["litpop"]
                    rows, cols = group_indices[gname]
                    n_time = rows.shape[0]

                    for t_idx in range(n_time):
                        row = rows[t_idx]
                        col = cols[t_idx]

                        if row == INVALID_INDEX or col == INVALID_INDEX or big_data is None:
                            out = fill_slice
                        else:
                            local_row = int(row) - row_min
                            local_col = int(col) - col_min
                            patch = big_data[
                                local_row - HALF_PATCH : local_row + HALF_PATCH,
                                local_col - HALF_PATCH : local_col + HALF_PATCH,
                            ]
                            out = aggregate_patch_to_41x41(patch, nodata)

                        litpop_var[t_idx, :, :] = out
                        stats.fill_cells += int(np.count_nonzero(out == FILL_VALUE))
                        stats.total_cells += OUT_N * OUT_N

        stats.elapsed_sec = time.time() - t0
        return (
            "ok",
            stats,
            (
                f"[{year}] Done: typhoons={stats.typhoons}, steps={stats.time_steps}, "
                f"invalid_points={stats.invalid_points}, fill_ratio={stats.fill_ratio:.4f}, "
                f"elapsed={stats.elapsed_sec:.1f}s, output={dst_nc.name}"
            ),
        )
    except Exception as exc:
        return "fail", stats, f"[{year}] Failed: {exc}"


def run(start_year: int, end_year: int, overwrite: bool, logger: logging.Logger) -> int:
    logger.info("=" * 72)
    logger.info("STEP3 LitPop Integration Started")
    logger.info("Years: %s-%s | overwrite=%s", start_year, end_year, overwrite)
    logger.info("Input NC dir: %s", INPUT_NC_DIR)
    logger.info("Input TIF dir: %s", LITPOP_DIR)
    logger.info("Output NC dir: %s", OUTPUT_NC_DIR)
    logger.info("=" * 72)

    ok_years: List[int] = []
    skipped_years: List[int] = []
    failed_years: List[int] = []
    all_stats: List[YearStats] = []

    for year in range(start_year, end_year + 1):
        status, stats, msg = process_year(year, overwrite=overwrite, logger=logger)
        logger.info(msg)
        if status == "ok":
            ok_years.append(year)
            if stats is not None:
                all_stats.append(stats)
        elif status == "skip":
            skipped_years.append(year)
        else:
            failed_years.append(year)

    total_typhoons = int(sum(s.typhoons for s in all_stats))
    total_steps = int(sum(s.time_steps for s in all_stats))
    total_invalid = int(sum(s.invalid_points for s in all_stats))
    total_fill_cells = int(sum(s.fill_cells for s in all_stats))
    total_cells = int(sum(s.total_cells for s in all_stats))
    total_elapsed = float(sum(s.elapsed_sec for s in all_stats))
    total_fill_ratio = (total_fill_cells / total_cells) if total_cells else 0.0

    logger.info("=" * 72)
    logger.info("STEP3 LitPop Integration Finished")
    logger.info("Success years (%s): %s", len(ok_years), ok_years)
    logger.info("Skipped years (%s): %s", len(skipped_years), skipped_years)
    logger.info("Failed years (%s): %s", len(failed_years), failed_years)
    logger.info(
        "Summary | typhoons=%s, steps=%s, invalid_points=%s, fill_ratio=%.4f, elapsed=%.1fs",
        total_typhoons,
        total_steps,
        total_invalid,
        total_fill_ratio,
        total_elapsed,
    )
    logger.info("Log file: %s", LOG_FILE)
    logger.info("=" * 72)

    return 0 if not failed_years else 1


def main() -> int:
    args = parse_args()
    if args.start_year > args.end_year:
        print("start-year must be <= end-year", file=sys.stderr)
        return 2

    logger = setup_logger()
    return run(
        start_year=args.start_year,
        end_year=args.end_year,
        overwrite=bool(args.overwrite),
        logger=logger,
    )


if __name__ == "__main__":
    raise SystemExit(main())
