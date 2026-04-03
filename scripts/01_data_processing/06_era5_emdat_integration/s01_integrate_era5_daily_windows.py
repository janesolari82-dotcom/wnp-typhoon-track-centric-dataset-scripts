#!/usr/bin/env python3
"""
Phase 1: integrate ERA5 daily fields and hazard variables into yearly typhoon NetCDF.

Output path is fixed to:
  reproduction_v6/04_final_output/emdat_attribution_integration
"""

from __future__ import annotations

import argparse
import shutil
from collections import OrderedDict
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from netCDF4 import Dataset, num2date

from s00_common import (
    ERA5_ROOT,
    INPUT_NC_DIR,
    LOG_DIR,
    OUTPUT_NC_DIR,
    cfdate_to_date,
    minmax_normalize_2d,
    parse_years_expr,
    setup_logger,
)

FILL_VALUE = -9999.0
WINDOW_N = 41
ERA5_LAT_N = 721
ERA5_LON_N = 1440


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Integrate ERA5 daily fields into pop_light NetCDF files.")
    p.add_argument("--years", type=str, default="2013", help="Year expression, e.g. 2013 or 2000-2024")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    p.add_argument("--era5-root", type=Path, default=ERA5_ROOT)
    p.add_argument("--input-nc-dir", type=Path, default=INPUT_NC_DIR)
    p.add_argument("--output-nc-dir", type=Path, default=OUTPUT_NC_DIR)
    return p.parse_args()


def find_era5_file(root: Path, year: int, var: str) -> Path:
    patterns = {
        "u10": [f"u10max_{year}.nc", f"*u10max*{year}*.nc", f"*u10*{year}*.nc"],
        "v10": [f"v10max_{year}.nc", f"*v10max*{year}*.nc", f"*v10*{year}*.nc"],
        "tp": [f"tp_{year}.nc", f"*tp*{year}*.nc"],
    }[var]
    for pattern in patterns:
        found = sorted(root.rglob(pattern))
        if found:
            return found[0]
    raise FileNotFoundError(f"ERA5 file for {var} {year} not found under: {root}")


class DaySliceCache:
    def __init__(self, u_ds, v_ds, tp_ds, max_items: int = 12):
        self.u_ds = u_ds
        self.v_ds = v_ds
        self.tp_ds = tp_ds
        self.max_items = max_items
        self.cache: "OrderedDict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]" = OrderedDict()

    def get(self, day_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if day_idx in self.cache:
            self.cache.move_to_end(day_idx)
            return self.cache[day_idx]
        u = np.asarray(self.u_ds[day_idx, :, :], dtype=np.float32)
        v = np.asarray(self.v_ds[day_idx, :, :], dtype=np.float32)
        tp = np.asarray(self.tp_ds[day_idx, :, :], dtype=np.float32)
        self.cache[day_idx] = (u, v, tp)
        if len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return u, v, tp


def to_day_indices(time_values: np.ndarray, units: str, year: int, max_day_idx: int) -> List[int]:
    dts = num2date(time_values, units=units, calendar="standard")
    y0 = date(year, 1, 1)
    out: List[int] = []
    for dt in dts:
        d = cfdate_to_date(dt)
        idx = (d - y0).days
        if idx < 0:
            idx = 0
        if idx > max_day_idx:
            idx = max_day_idx
        out.append(int(idx))
    return out


def ensure_var(grp: Dataset, name: str, dtype: str = "f4", fill_value=FILL_VALUE):
    if name in grp.variables:
        return grp.variables[name]
    if dtype == "i2":
        var = grp.createVariable(
            name,
            "i2",
            ("time", "window_lat", "window_lon"),
            zlib=True,
            shuffle=True,
            complevel=4,
            fill_value=np.int16(0),
            chunksizes=(1, WINDOW_N, WINDOW_N),
        )
    else:
        var = grp.createVariable(
            name,
            "f4",
            ("time", "window_lat", "window_lon"),
            zlib=True,
            shuffle=True,
            complevel=4,
            fill_value=np.float32(fill_value),
            chunksizes=(1, WINDOW_N, WINDOW_N),
        )
        var.missing_value = np.float32(fill_value)
    return var


def prepare_output_file(src_nc: Path, dst_nc: Path, overwrite: bool) -> str:
    if not src_nc.exists():
        return f"missing_source:{src_nc}"
    dst_nc.parent.mkdir(parents=True, exist_ok=True)
    if dst_nc.exists():
        if not overwrite:
            return "exists_skip"
        dst_nc.unlink()
    shutil.copy2(src_nc, dst_nc)
    return "ok"


def integrate_year(
    year: int,
    era5_root: Path,
    input_nc_dir: Path,
    output_nc_dir: Path,
    overwrite: bool,
    logger,
) -> bool:
    src_nc = input_nc_dir / f"typhoon_{year}_ocean_litpop_poplight.nc"
    dst_nc = output_nc_dir / f"typhoon_{year}_ocean_litpop_poplight_emdat.nc"
    prep = prepare_output_file(src_nc, dst_nc, overwrite)
    if prep != "ok":
        if prep == "exists_skip":
            logger.info("[%s] skip (already exists): %s", year, dst_nc.name)
            return True
        logger.error("[%s] %s", year, prep)
        return False

    u_file = find_era5_file(era5_root, year, "u10")
    v_file = find_era5_file(era5_root, year, "v10")
    tp_file = find_era5_file(era5_root, year, "tp")

    logger.info("[%s] ERA5 files: u=%s v=%s tp=%s", year, u_file.name, v_file.name, tp_file.name)

    with h5py.File(u_file, "r") as fu, h5py.File(v_file, "r") as fv, h5py.File(tp_file, "r") as fp, Dataset(
        dst_nc, "r+"
    ) as ds:
        u_ds = fu["u10"]
        v_ds = fv["v10"]
        tp_ds = fp["tp"]
        u_time = np.asarray(fu["valid_time"][:], dtype=np.int64)
        v_time = np.asarray(fv["valid_time"][:], dtype=np.int64)
        tp_time = np.asarray(fp["valid_time"][:], dtype=np.int64)
        if not (np.array_equal(u_time, v_time) and np.array_equal(u_time, tp_time)):
            raise RuntimeError(f"ERA5 valid_time mismatch for year {year}")

        cache = DaySliceCache(u_ds, v_ds, tp_ds, max_items=16)
        max_day_idx = int(u_ds.shape[0]) - 1

        group_count = 0
        step_count = 0
        for gname, grp in ds.groups.items():
            group_count += 1
            if "time" not in grp.variables:
                logger.warning("[%s] group without time skipped: %s", year, gname)
                continue

            tvals = np.asarray(grp.variables["time"][:], dtype=np.float64)
            t_units = grp.variables["time"].units
            day_indices = to_day_indices(tvals, t_units, year, max_day_idx=max_day_idx)

            center_lat = np.asarray(grp.variables["center_lat"][:], dtype=np.float32)
            center_lon = np.asarray(grp.variables["center_lon"][:], dtype=np.float32)
            window_lat = np.asarray(grp.variables["window_lat"][:], dtype=np.float32)
            window_lon = np.asarray(grp.variables["window_lon"][:], dtype=np.float32)
            if window_lat.size != WINDOW_N or window_lon.size != WINDOW_N:
                raise RuntimeError(f"Unexpected window size in {gname}: {window_lat.size}x{window_lon.size}")

            wlat2d = window_lat[:, None]
            wlon2d = window_lon[None, :]

            var_u = ensure_var(grp, "era5_u10_daily_max")
            var_v = ensure_var(grp, "era5_v10_daily_max")
            var_w = ensure_var(grp, "era5_wind_speed_daily_proxy")
            var_tp = ensure_var(grp, "era5_tp_daily_sum_mm")
            var_hw = ensure_var(grp, "hazard_wind_power_daily")
            var_hr = ensure_var(grp, "hazard_rain_daily")
            var_hc = ensure_var(grp, "hazard_compound_daily")

            # Initialize to zero to avoid fill-value arithmetic later.
            var_u[:] = np.float32(0.0)
            var_v[:] = np.float32(0.0)
            var_w[:] = np.float32(0.0)
            var_tp[:] = np.float32(0.0)
            var_hw[:] = np.float32(0.0)
            var_hr[:] = np.float32(0.0)
            var_hc[:] = np.float32(0.0)

            for t_idx, day_idx in enumerate(day_indices):
                step_count += 1
                u2d, v2d, tp2d = cache.get(day_idx)

                c_lat = float(center_lat[t_idx])
                c_lon = float(center_lon[t_idx])
                lat_abs = c_lat + wlat2d
                lon_abs = c_lon + wlon2d
                lon_abs = np.mod(lon_abs, 360.0)

                lat_idx = np.rint((90.0 - lat_abs) / 0.25).astype(np.int32)
                lon_idx = np.rint(lon_abs / 0.25).astype(np.int32)
                np.clip(lat_idx, 0, ERA5_LAT_N - 1, out=lat_idx)
                np.clip(lon_idx, 0, ERA5_LON_N - 1, out=lon_idx)

                u_win = u2d[lat_idx, lon_idx].astype(np.float32)
                v_win = v2d[lat_idx, lon_idx].astype(np.float32)
                tp_mm = (tp2d[lat_idx, lon_idx] * np.float32(1000.0)).astype(np.float32)

                wind = np.sqrt(np.maximum(u_win * u_win + v_win * v_win, 0.0), dtype=np.float32)
                hazard_w = np.power(np.maximum(wind - np.float32(12.0), np.float32(0.0)), np.float32(3.0), dtype=np.float32)
                hazard_r = tp_mm
                hazard_comp = (0.5 * minmax_normalize_2d(hazard_w) + 0.5 * minmax_normalize_2d(hazard_r)).astype(
                    np.float32
                )

                var_u[t_idx, :, :] = u_win
                var_v[t_idx, :, :] = v_win
                var_w[t_idx, :, :] = wind
                var_tp[t_idx, :, :] = tp_mm
                var_hw[t_idx, :, :] = hazard_w
                var_hr[t_idx, :, :] = hazard_r
                var_hc[t_idx, :, :] = hazard_comp

        ds.emdat_era5_integration_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ds.emdat_era5_method_version = "v4_1"
        ds.emdat_output_root = str(output_nc_dir)
        ds.emdat_era5_u_file = u_file.name
        ds.emdat_era5_v_file = v_file.name
        ds.emdat_era5_tp_file = tp_file.name
        logger.info("[%s] done: groups=%s steps=%s output=%s", year, group_count, step_count, dst_nc.name)
    return True


def main() -> int:
    args = parse_args()
    years = parse_years_expr(args.years)
    logger = setup_logger("emdat_phase08_era5", LOG_DIR / "08_integrate_era5_daily_windows.log", args.log_level)

    logger.info("Phase 1 started | years=%s | output=%s", years, args.output_nc_dir)
    ok = True
    for year in years:
        try:
            ok = integrate_year(
                year=year,
                era5_root=args.era5_root,
                input_nc_dir=args.input_nc_dir,
                output_nc_dir=args.output_nc_dir,
                overwrite=bool(args.overwrite),
                logger=logger,
            ) and ok
        except Exception as exc:
            logger.exception("[%s] failed: %s", year, exc)
            ok = False
    logger.info("Phase 1 finished | success=%s", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
