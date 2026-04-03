#!/usr/bin/env python3
"""
Rebuild ocean variables in final yearly typhoon NetCDF files from split Copernicus inputs.

This script:
1. reads old final yearly NetCDF files under 04_final_output/emdat_attribution_integration
2. copies all non-ocean structure/content into a new output file
3. replaces the ocean layer with:
   - ocean_thetao(time, window_time, thetao_depth, window_lat, window_lon)
   - ocean_mlotst(time, window_time, window_lat, window_lon)
   - ocean_zos(time, window_time, window_lat, window_lon)

Old files remain untouched.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Tuple

import numpy as np
from netCDF4 import Dataset, Group, Variable, num2date

FILL_VALUE = np.float32(-9999.0)
BUFFER_DAYS = 7
WINDOW_TIME_N = BUFFER_DAYS * 2 + 1
WINDOW_N = 41
WINDOW_SIDE_NM = 200.0
KM_PER_NM = 1.852
WINDOW_DEG = (WINDOW_SIDE_NM * KM_PER_NM / 2.0) / 111.0
PRODUCT_NAME = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
SOURCE_START = date(2000, 1, 1)
SOURCE_END = date(2024, 12, 31)
OLD_OCEAN_VARS = {
    "ocean_thetao",
    "ocean_zos",
    "ocean_so",
    "ocean_uo",
    "ocean_vo",
    "ocean_mlotst",
}
THETAO_PATTERN = re.compile(r"glo_phy_thetao_(\d{8})_\d{8}\.nc$")
SURFACE_PATTERN = re.compile(r"glo_phy_surface_(\d{6})_(\d{6})\.nc$")

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPRO_ROOT = PROJECT_ROOT / "reproduction_v6"

DEFAULT_INPUT_FINAL_DIR = REPRO_ROOT / "04_final_output" / "emdat_attribution_integration"
DEFAULT_THETAO_DIR = PROJECT_ROOT / "data" / "raw" / "using" / "copernicus_download_split" / "thetao"
DEFAULT_SURFACE_DIR = PROJECT_ROOT / "data" / "raw" / "using" / "copernicus_download_split" / "surface"
DEFAULT_OUTPUT_DIR = REPRO_ROOT / "new_final_output" / "emdat_attribution_integration"
DEFAULT_LOG_FILE = REPRO_ROOT / "06_logs" / "copernicus_reintegration_split.log"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reintegrate Copernicus split ocean data into new final yearly NetCDF files."
    )
    p.add_argument("--years", type=str, default="2000-2024", help="Year expression, e.g. 2000-2024 or 2020,2021")
    p.add_argument("--input-final-dir", type=Path, default=DEFAULT_INPUT_FINAL_DIR)
    p.add_argument("--thetao-dir", type=Path, default=DEFAULT_THETAO_DIR)
    p.add_argument("--surface-dir", type=Path, default=DEFAULT_SURFACE_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing yearly outputs.")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    p.add_argument("--max-open-thetao", type=int, default=8, help="Max open daily thetao files.")
    p.add_argument("--max-open-surface", type=int, default=4, help="Max open quarterly surface files.")
    p.add_argument(
        "--group-filter",
        type=str,
        default="",
        help="Optional regex filter for typhoon group names, used for smoke runs.",
    )
    p.add_argument(
        "--group-limit",
        type=int,
        default=0,
        help="Optional limit on number of groups processed per year, used for smoke runs.",
    )
    return p.parse_args()


def setup_logger(log_file: Path, level: str) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("copernicus_reintegration_split")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def parse_years_expr(expr: str) -> List[int]:
    out = set()
    for part in (x.strip() for x in expr.split(",")):
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            lo = min(start, end)
            hi = max(start, end)
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return sorted(out)


def cfdate_to_date(dt_obj) -> date:
    return date(int(dt_obj.year), int(dt_obj.month), int(dt_obj.day))


def normalize_lon(lon: float) -> float:
    val = float(lon)
    if val < 0.0:
        val = val % 360.0
    return val


def copy_attrs(src, dst, skip: Iterable[str] = ()) -> None:
    skip_set = set(skip)
    for attr in src.ncattrs():
        if attr in skip_set:
            continue
        dst.setncattr(attr, src.getncattr(attr))


def resolve_fill_value(var: Variable):
    if "_FillValue" in var.ncattrs():
        return var.getncattr("_FillValue")
    if "missing_value" in var.ncattrs():
        return var.getncattr("missing_value")
    return None


def create_var_like(dst_grp: Group, src_var: Variable, name: Optional[str] = None) -> Variable:
    var_name = name or src_var.name
    kwargs = {}
    fill_value = resolve_fill_value(src_var)
    if fill_value is not None:
        kwargs["fill_value"] = fill_value
    try:
        filters = src_var.filters()
    except Exception:
        filters = {}
    if filters:
        kwargs["zlib"] = bool(filters.get("zlib", False))
        kwargs["shuffle"] = bool(filters.get("shuffle", False))
        if filters.get("complevel") is not None:
            kwargs["complevel"] = int(filters["complevel"])
    try:
        chunking = src_var.chunking()
    except Exception:
        chunking = None
    if isinstance(chunking, tuple):
        kwargs["chunksizes"] = chunking
    dst_var = dst_grp.createVariable(var_name, src_var.dtype, src_var.dimensions, **kwargs)
    copy_attrs(src_var, dst_var, skip={"_FillValue"})
    dst_var[:] = src_var[:]
    return dst_var


@dataclass
class YearStats:
    groups: int = 0
    timesteps: int = 0
    thetao_missing_days: int = 0
    surface_missing_days: int = 0
    thetao_all_fill_windows: int = 0
    mlotst_all_fill_windows: int = 0
    zos_all_fill_windows: int = 0


class DatasetHandleCache:
    def __init__(self, max_items: int):
        self.max_items = max_items
        self.cache: "OrderedDict[Path, Dataset]" = OrderedDict()

    def get(self, path: Path) -> Dataset:
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        ds = Dataset(path, "r")
        self.cache[path] = ds
        if len(self.cache) > self.max_items:
            _, old = self.cache.popitem(last=False)
            old.close()
        return ds

    def close(self) -> None:
        while self.cache:
            _, ds = self.cache.popitem(last=False)
            ds.close()


class CopernicusSplitSource:
    def __init__(
        self,
        thetao_dir: Path,
        surface_dir: Path,
        logger: logging.Logger,
        max_open_thetao: int = 8,
        max_open_surface: int = 4,
    ):
        self.thetao_dir = thetao_dir
        self.surface_dir = surface_dir
        self.logger = logger
        self.thetao_index: Dict[date, Path] = {}
        self.surface_index: Dict[date, Tuple[Path, int]] = {}
        self.thetao_cache = DatasetHandleCache(max_open_thetao)
        self.surface_cache = DatasetHandleCache(max_open_surface)
        self.latitudes: Optional[np.ndarray] = None
        self.longitudes: Optional[np.ndarray] = None
        self.thetao_depth: Optional[np.ndarray] = None
        self.thetao_meta: Dict[str, str] = {}
        self.mlotst_meta: Dict[str, str] = {}
        self.zos_meta: Dict[str, str] = {}
        self._build_indices()
        self._load_reference_metadata()

    def close(self) -> None:
        self.thetao_cache.close()
        self.surface_cache.close()

    def _build_indices(self) -> None:
        thetao_files = sorted(p for p in self.thetao_dir.rglob("*.nc") if p.is_file())
        if not thetao_files:
            raise FileNotFoundError(f"No thetao files found under: {self.thetao_dir}")
        for path in thetao_files:
            match = THETAO_PATTERN.match(path.name)
            if not match:
                continue
            day = datetime.strptime(match.group(1), "%Y%m%d").date()
            self.thetao_index[day] = path
        surface_files = sorted(p for p in self.surface_dir.glob("*.nc") if p.is_file())
        if not surface_files:
            raise FileNotFoundError(f"No surface files found under: {self.surface_dir}")
        for path in surface_files:
            if not SURFACE_PATTERN.match(path.name):
                continue
            with Dataset(path, "r") as ds:
                time_var = ds.variables["time"]
                times = num2date(time_var[:], units=time_var.units, calendar=getattr(time_var, "calendar", "standard"))
                for idx, dt_obj in enumerate(times):
                    self.surface_index[cfdate_to_date(dt_obj)] = (path, idx)
        self.logger.info(
            "Indexed Copernicus split files | thetao_days=%s | surface_days=%s",
            len(self.thetao_index),
            len(self.surface_index),
        )

    def _load_reference_metadata(self) -> None:
        thetao_path = self.thetao_index[min(self.thetao_index)]
        with Dataset(thetao_path, "r") as ds:
            self.latitudes = np.asarray(ds.variables["latitude"][:], dtype=np.float32)
            self.longitudes = np.asarray(ds.variables["longitude"][:], dtype=np.float32)
            self.thetao_depth = np.asarray(ds.variables["depth"][:], dtype=np.float32)
            self.thetao_meta = {
                "units": str(getattr(ds.variables["thetao"], "units", "degrees_C")),
                "standard_name": str(getattr(ds.variables["thetao"], "standard_name", "")),
                "source_long_name": str(getattr(ds.variables["thetao"], "long_name", "Temperature")),
                "depth_units": str(getattr(ds.variables["depth"], "units", "m")),
                "depth_long_name": str(getattr(ds.variables["depth"], "long_name", "Depth")),
                "depth_standard_name": str(getattr(ds.variables["depth"], "standard_name", "depth")),
            }
        sample_surface = self.surface_index[min(self.surface_index)][0]
        with Dataset(sample_surface, "r") as ds:
            for name, meta_store in (("mlotst", self.mlotst_meta), ("zos", self.zos_meta)):
                var = ds.variables[name]
                meta_store["units"] = str(getattr(var, "units", ""))
                meta_store["standard_name"] = str(getattr(var, "standard_name", ""))
                meta_store["source_long_name"] = str(getattr(var, "long_name", name))

    def in_global_range(self, target_day: date) -> bool:
        return SOURCE_START <= target_day <= SOURCE_END

    def _spatial_bounds(self, lat: float, lon: float) -> Tuple[int, int, int, int]:
        assert self.latitudes is not None
        assert self.longitudes is not None
        lon_norm = normalize_lon(lon)
        lat_start = int(np.searchsorted(self.latitudes, lat - WINDOW_DEG, side="left"))
        lat_end = int(np.searchsorted(self.latitudes, lat + WINDOW_DEG, side="right"))
        lon_start = int(np.searchsorted(self.longitudes, lon_norm - WINDOW_DEG, side="left"))
        lon_end = int(np.searchsorted(self.longitudes, lon_norm + WINDOW_DEG, side="right"))
        lat_start = max(0, lat_start)
        lat_end = min(len(self.latitudes), lat_end)
        lon_start = max(0, lon_start)
        lon_end = min(len(self.longitudes), lon_end)
        return lat_start, lat_end, lon_start, lon_end

    @staticmethod
    def _pad_2d(data) -> np.ndarray:
        out = np.full((WINDOW_N, WINDOW_N), FILL_VALUE, dtype=np.float32)
        if data is None:
            return out
        filled = np.asarray(np.ma.filled(data, FILL_VALUE), dtype=np.float32)
        if filled.ndim != 2:
            return out
        h = min(filled.shape[0], WINDOW_N)
        w = min(filled.shape[1], WINDOW_N)
        out[:h, :w] = filled[:h, :w]
        return out

    def _pad_3d_depth(self, data) -> np.ndarray:
        assert self.thetao_depth is not None
        out = np.full((len(self.thetao_depth), WINDOW_N, WINDOW_N), FILL_VALUE, dtype=np.float32)
        if data is None:
            return out
        filled = np.asarray(np.ma.filled(data, FILL_VALUE), dtype=np.float32)
        if filled.ndim != 3:
            return out
        d = min(filled.shape[0], len(self.thetao_depth))
        h = min(filled.shape[1], WINDOW_N)
        w = min(filled.shape[2], WINDOW_N)
        out[:d, :h, :w] = filled[:d, :h, :w]
        return out

    def extract_thetao_window(self, target_day: date, lat: float, lon: float) -> Tuple[np.ndarray, bool]:
        assert self.thetao_depth is not None
        if not self.in_global_range(target_day):
            return self._pad_3d_depth(None), False
        path = self.thetao_index.get(target_day)
        if path is None:
            return self._pad_3d_depth(None), False
        lat_start, lat_end, lon_start, lon_end = self._spatial_bounds(lat, lon)
        if lat_start >= lat_end or lon_start >= lon_end:
            return self._pad_3d_depth(None), True
        ds = self.thetao_cache.get(path)
        data = ds.variables["thetao"][0, :, lat_start:lat_end, lon_start:lon_end]
        return self._pad_3d_depth(data), True

    def extract_surface_window(self, var_name: str, target_day: date, lat: float, lon: float) -> Tuple[np.ndarray, bool]:
        if not self.in_global_range(target_day):
            return self._pad_2d(None), False
        surface_ref = self.surface_index.get(target_day)
        if surface_ref is None:
            return self._pad_2d(None), False
        lat_start, lat_end, lon_start, lon_end = self._spatial_bounds(lat, lon)
        if lat_start >= lat_end or lon_start >= lon_end:
            return self._pad_2d(None), True
        path, time_idx = surface_ref
        ds = self.surface_cache.get(path)
        data = ds.variables[var_name][time_idx, lat_start:lat_end, lon_start:lon_end]
        return self._pad_2d(data), True


def copy_root_structure(src: Dataset, dst: Dataset) -> None:
    for dim_name, dim in src.dimensions.items():
        dst.createDimension(dim_name, None if dim.isunlimited() else len(dim))
    for var_name, var in src.variables.items():
        create_var_like(dst, var, name=var_name)
    copy_attrs(src, dst)


def add_reintegration_attrs(ds: Dataset, args: argparse.Namespace, source: CopernicusSplitSource) -> None:
    ds.copernicus_reintegration_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ds.copernicus_reintegration_product = PRODUCT_NAME
    ds.copernicus_reintegration_thetao_dir = str(args.thetao_dir)
    ds.copernicus_reintegration_surface_dir = str(args.surface_dir)
    ds.copernicus_reintegration_output_root = str(args.output_dir)
    ds.copernicus_reintegration_window_side_nautical_miles = np.float32(WINDOW_SIDE_NM)
    ds.copernicus_reintegration_window_half_degree = np.float32(WINDOW_DEG)
    ds.copernicus_reintegration_time_offsets = ",".join(str(i) for i in range(-BUFFER_DAYS, BUFFER_DAYS + 1))
    ds.copernicus_reintegration_thetao_depth_count = np.int32(len(source.thetao_depth))
    ds.copernicus_reintegration_source_coverage = f"{SOURCE_START.isoformat()} to {SOURCE_END.isoformat()}"
    ds.copernicus_reintegration_ocean_variables = "ocean_thetao,ocean_mlotst,ocean_zos"


def create_new_ocean_variables(dst_grp: Group, source: CopernicusSplitSource) -> Tuple[Variable, Variable, Variable]:
    assert source.thetao_depth is not None
    if "thetao_depth" not in dst_grp.dimensions:
        dst_grp.createDimension("thetao_depth", len(source.thetao_depth))
    depth_var = dst_grp.createVariable("thetao_depth", "f4", ("thetao_depth",))
    depth_var[:] = source.thetao_depth
    depth_var.units = source.thetao_meta["depth_units"]
    depth_var.long_name = source.thetao_meta["depth_long_name"]
    depth_var.standard_name = source.thetao_meta["depth_standard_name"]

    thetao = dst_grp.createVariable(
        "ocean_thetao",
        "f4",
        ("time", "window_time", "thetao_depth", "window_lat", "window_lon"),
        zlib=True,
        shuffle=True,
        complevel=4,
        fill_value=FILL_VALUE,
        chunksizes=(1, WINDOW_TIME_N, len(source.thetao_depth), WINDOW_N, WINDOW_N),
    )
    thetao.units = source.thetao_meta["units"]
    thetao.standard_name = source.thetao_meta["standard_name"]
    thetao.long_name = "Typhoon-centered multi-depth temperature window"
    thetao.missing_value = FILL_VALUE

    mlotst = dst_grp.createVariable(
        "ocean_mlotst",
        "f4",
        ("time", "window_time", "window_lat", "window_lon"),
        zlib=True,
        shuffle=True,
        complevel=4,
        fill_value=FILL_VALUE,
        chunksizes=(1, WINDOW_TIME_N, WINDOW_N, WINDOW_N),
    )
    mlotst.units = source.mlotst_meta["units"]
    mlotst.standard_name = source.mlotst_meta["standard_name"]
    mlotst.long_name = source.mlotst_meta["source_long_name"]
    mlotst.missing_value = FILL_VALUE

    zos = dst_grp.createVariable(
        "ocean_zos",
        "f4",
        ("time", "window_time", "window_lat", "window_lon"),
        zlib=True,
        shuffle=True,
        complevel=4,
        fill_value=FILL_VALUE,
        chunksizes=(1, WINDOW_TIME_N, WINDOW_N, WINDOW_N),
    )
    zos.units = source.zos_meta["units"]
    zos.standard_name = source.zos_meta["standard_name"]
    zos.long_name = source.zos_meta["source_long_name"]
    zos.missing_value = FILL_VALUE

    return thetao, mlotst, zos


def fill_group_ocean_data(
    src_grp: Group,
    dst_grp: Group,
    source: CopernicusSplitSource,
    logger: logging.Logger,
    year_stats: YearStats,
) -> None:
    time_var = src_grp.variables["time"]
    center_lat = np.asarray(src_grp.variables["center_lat"][:], dtype=np.float32)
    center_lon = np.asarray(src_grp.variables["center_lon"][:], dtype=np.float32)
    time_values = np.asarray(time_var[:], dtype=np.float64)
    time_units = time_var.units
    calendar = getattr(time_var, "calendar", "standard")
    thetao_var, mlotst_var, zos_var = create_new_ocean_variables(dst_grp, source)
    dates = num2date(time_values, units=time_units, calendar=calendar)

    for t_idx, dt_obj in enumerate(dates):
        anchor_day = cfdate_to_date(dt_obj)
        lat = float(center_lat[t_idx])
        lon = float(center_lon[t_idx])
        thetao_window = np.full((WINDOW_TIME_N, len(source.thetao_depth), WINDOW_N, WINDOW_N), FILL_VALUE, dtype=np.float32)
        mlotst_window = np.full((WINDOW_TIME_N, WINDOW_N, WINDOW_N), FILL_VALUE, dtype=np.float32)
        zos_window = np.full((WINDOW_TIME_N, WINDOW_N, WINDOW_N), FILL_VALUE, dtype=np.float32)

        for out_idx, day_offset in enumerate(range(-BUFFER_DAYS, BUFFER_DAYS + 1)):
            target_day = anchor_day + timedelta(days=day_offset)
            thetao_arr, thetao_available = source.extract_thetao_window(target_day, lat, lon)
            mlotst_arr, mlotst_available = source.extract_surface_window("mlotst", target_day, lat, lon)
            zos_arr, zos_available = source.extract_surface_window("zos", target_day, lat, lon)
            thetao_window[out_idx] = thetao_arr
            mlotst_window[out_idx] = mlotst_arr
            zos_window[out_idx] = zos_arr
            if not thetao_available:
                year_stats.thetao_missing_days += 1
            if not (mlotst_available and zos_available):
                year_stats.surface_missing_days += 1

        thetao_var[t_idx, :, :, :, :] = thetao_window
        mlotst_var[t_idx, :, :, :] = mlotst_window
        zos_var[t_idx, :, :, :] = zos_window

        if not np.isfinite(thetao_window[thetao_window != FILL_VALUE]).any():
            year_stats.thetao_all_fill_windows += 1
        if not np.isfinite(mlotst_window[mlotst_window != FILL_VALUE]).any():
            year_stats.mlotst_all_fill_windows += 1
        if not np.isfinite(zos_window[zos_window != FILL_VALUE]).any():
            year_stats.zos_all_fill_windows += 1

        year_stats.timesteps += 1


def copy_group_without_old_ocean(src_grp: Group, dst_grp: Group) -> None:
    for dim_name, dim in src_grp.dimensions.items():
        dst_grp.createDimension(dim_name, None if dim.isunlimited() else len(dim))
    for var_name, var in src_grp.variables.items():
        if var_name in OLD_OCEAN_VARS:
            continue
        create_var_like(dst_grp, var, name=var_name)
    copy_attrs(src_grp, dst_grp)


def should_process_group(group_name: str, group_filter: Optional[re.Pattern[str]]) -> bool:
    if group_filter is None:
        return True
    return bool(group_filter.search(group_name))


def process_year(
    year: int,
    args: argparse.Namespace,
    source: CopernicusSplitSource,
    logger: logging.Logger,
) -> bool:
    src_nc = args.input_final_dir / f"typhoon_{year}_ocean_litpop_poplight_emdat.nc"
    dst_nc = args.output_dir / f"typhoon_{year}_ocean_litpop_poplight_emdat.nc"
    tmp_nc = dst_nc.with_suffix(dst_nc.suffix + ".tmp")
    if not src_nc.exists():
        logger.error("[%s] missing source final file: %s", year, src_nc)
        return False
    dst_nc.parent.mkdir(parents=True, exist_ok=True)
    if dst_nc.exists():
        if not args.overwrite:
            logger.info("[%s] skip existing output: %s", year, dst_nc.name)
            return True
        dst_nc.unlink()
    if tmp_nc.exists():
        tmp_nc.unlink()

    stats = YearStats()
    group_filter = re.compile(args.group_filter) if args.group_filter else None
    groups_written = 0
    logger.info("[%s] start -> %s", year, dst_nc.name)
    try:
        with Dataset(src_nc, "r") as src, Dataset(tmp_nc, "w", format="NETCDF4") as dst:
            copy_root_structure(src, dst)
            add_reintegration_attrs(dst, args, source)
            for gname, src_grp in src.groups.items():
                if not should_process_group(gname, group_filter):
                    continue
                if args.group_limit and groups_written >= args.group_limit:
                    break
                dst_grp = dst.createGroup(gname)
                copy_group_without_old_ocean(src_grp, dst_grp)
                fill_group_ocean_data(src_grp, dst_grp, source, logger, stats)
                stats.groups += 1
                groups_written += 1
            dst.sync()
        os.replace(tmp_nc, dst_nc)
    except Exception:
        if tmp_nc.exists():
            tmp_nc.unlink()
        raise

    logger.info(
        "[%s] done | groups=%s timesteps=%s thetao_missing_days=%s surface_missing_days=%s thetao_all_fill=%s mlotst_all_fill=%s zos_all_fill=%s",
        year,
        stats.groups,
        stats.timesteps,
        stats.thetao_missing_days,
        stats.surface_missing_days,
        stats.thetao_all_fill_windows,
        stats.mlotst_all_fill_windows,
        stats.zos_all_fill_windows,
    )
    return True


def main() -> int:
    args = parse_args()
    years = parse_years_expr(args.years)
    logger = setup_logger(DEFAULT_LOG_FILE, args.log_level)
    logger.info("Copernicus split reintegration started | years=%s | output=%s", years, args.output_dir)
    source = CopernicusSplitSource(
        thetao_dir=args.thetao_dir,
        surface_dir=args.surface_dir,
        logger=logger,
        max_open_thetao=args.max_open_thetao,
        max_open_surface=args.max_open_surface,
    )
    ok = True
    try:
        for year in years:
            try:
                ok = process_year(year, args, source, logger) and ok
            except Exception as exc:
                logger.exception("[%s] failed: %s", year, exc)
                ok = False
    finally:
        source.close()
    logger.info("Copernicus split reintegration finished | success=%s", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


