#!/usr/bin/env python3
"""Nightlight harmonization pipeline v2 (overlap years: 2012 + 2013)."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
import yaml
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit


STEP_ORDER = [
    "inventory",
    "save_plan",
    "viirs_monthly_mosaic",
    "viirs_annual_weighted",
    "viirs_annual_denoised",
    "dmsp_standardized",
    "viirs_kd_log",
    "sigmoid_fit",
    "viirs_dn_simulated",
    "temporal_consistency",
    "harmonized_assemble",
    "quality_report",
]


@dataclass
class GridSpec:
    crs: str
    resolution: float
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    nodata: float
    dtype: str
    blockxsize: int
    blockysize: int

    @property
    def width(self) -> int:
        return int(round((self.lon_max - self.lon_min) / self.resolution))

    @property
    def height(self) -> int:
        return int(round((self.lat_max - self.lat_min) / self.resolution))

    @property
    def transform(self):
        return rasterio.transform.from_origin(
            self.lon_min,
            self.lat_max,
            self.resolution,
            self.resolution,
        )


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nightlight harmonization pipeline v2")
    parser.add_argument(
        "--config",
        type=str,
        default="reproduction_v6/config/nightlight_harmonization.yaml",
        help="Config path",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="all",
        help="Comma-separated step list or 'all'",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--gdalbuildvrt", type=str, default="gdalbuildvrt", help="gdalbuildvrt executable")
    parser.add_argument("--gdalwarp", type=str, default="gdalwarp", help="gdalwarp executable")
    return parser.parse_args()


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("nightlight_harmonization_v2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config: {path}")
    cfg["_config_path"] = str(path)
    return cfg


def resolve_path(cfg: Dict, key: str) -> Path:
    value = cfg["paths"][key]
    p = Path(value)
    return p if p.is_absolute() else project_root() / p


def grid_spec(cfg: Dict) -> GridSpec:
    g = cfg["grid"]
    return GridSpec(
        crs=str(g["crs"]),
        resolution=float(g["resolution_deg"]),
        lon_min=float(g["lon_min"]),
        lon_max=float(g["lon_max"]),
        lat_min=float(g["lat_min"]),
        lat_max=float(g["lat_max"]),
        nodata=float(g["nodata"]),
        dtype=str(g["dtype"]),
        blockxsize=int(g.get("blockxsize", 512)),
        blockysize=int(g.get("blockysize", 512)),
    )


def parse_steps(steps_arg: str) -> List[str]:
    if steps_arg.strip().lower() == "all":
        return STEP_ORDER
    steps = [s.strip() for s in steps_arg.split(",") if s.strip()]
    unknown = [s for s in steps if s not in STEP_ORDER]
    if unknown:
        raise ValueError(f"Unknown steps: {unknown}; valid={STEP_ORDER}")
    return steps


def year_range(start: int, end: int) -> List[int]:
    return list(range(int(start), int(end) + 1))


def monthly_expected(year: int) -> List[str]:
    if int(year) == 2012:
        return [f"{m:02d}" for m in range(4, 13)]
    return [f"{m:02d}" for m in range(1, 13)]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict], headers: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(headers))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_cmd(cmd: Sequence[str], logger: logging.Logger) -> None:
    logger.info("RUN: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def dmsp_regex() -> re.Pattern:
    return re.compile(r"^(F\d{2})(\d{4})\.(v\d[a-z]?)\.global\.stable_lights\.avg_vis\.tif$", re.IGNORECASE)


def collect_dmsp_files(dmsp_dir: Path) -> Dict[int, Path]:
    pat = dmsp_regex()
    out: Dict[int, Path] = {}
    for fp in sorted(dmsp_dir.glob("*.tif")):
        m = pat.match(fp.name)
        if m:
            out[int(m.group(2))] = fp
    return out


def collect_month_files(month_dir: Path, cfg: Dict) -> Dict:
    tile_re = re.compile(str(cfg["viirs"]["tile_regex"]))
    avg_suffix = str(cfg["viirs"]["avg_suffix"])
    cf_suffix = str(cfg["viirs"]["cf_suffix"])

    avg_map: Dict[str, Path] = {}
    cf_map: Dict[str, Path] = {}
    ignored: List[str] = []
    unknown: List[str] = []
    nested_tif_count = 0

    for p in month_dir.glob("*"):
        if p.is_dir():
            nested_tif_count += len(list(p.glob("*.tif")))
            continue
        if p.suffix.lower() != ".tif":
            continue
        name = p.name
        if name.endswith(avg_suffix):
            m = tile_re.search(name)
            if m:
                avg_map[m.group(1)] = p
            else:
                unknown.append(name)
            continue
        if name.endswith(cf_suffix):
            m = tile_re.search(name)
            if m:
                cf_map[m.group(1)] = p
            else:
                unknown.append(name)
            continue
        if any(name.endswith(sfx) for sfx in cfg["viirs"].get("ignore_suffixes", [])):
            ignored.append(name)
        else:
            unknown.append(name)

    return {
        "avg_map": avg_map,
        "cf_map": cf_map,
        "ignored": ignored,
        "unknown": unknown,
        "nested_tif_count": nested_tif_count,
    }


def raster_stats(path: Path, nodata: Optional[float]) -> Dict[str, float]:
    with rasterio.open(path) as ds:
        min_v = float("inf")
        max_v = float("-inf")
        sum_v = 0.0
        sumsq_v = 0.0
        count = 0
        zero_count = 0
        for _, win in ds.block_windows(1):
            arr = ds.read(1, window=win)
            valid = np.isfinite(arr)
            if nodata is not None:
                valid &= arr != nodata
            if not np.any(valid):
                continue
            vals = arr[valid].astype(np.float64)
            min_v = min(min_v, float(vals.min()))
            max_v = max(max_v, float(vals.max()))
            sum_v += float(vals.sum())
            sumsq_v += float(np.square(vals).sum())
            count += int(vals.size)
            zero_count += int(np.count_nonzero(vals <= 0))

    if count == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "zero_frac": float("nan"),
            "count": 0,
        }

    mean_v = sum_v / count
    std_v = math.sqrt(max(sumsq_v / count - mean_v * mean_v, 0.0))
    return {
        "min": min_v,
        "max": max_v,
        "mean": mean_v,
        "std": std_v,
        "zero_frac": zero_count / count,
        "count": count,
    }


def inventory_inputs(cfg: Dict, logger: logging.Logger) -> Tuple[Path, Path]:
    logger.info("Step 1/12: inventory and consistency checks")
    output_root = resolve_path(cfg, "output_root")
    inv_dir = output_root / "00_inventory"
    ensure_dir(inv_dir)

    dmsp_dir = resolve_path(cfg, "dmsp_dir")
    viirs_root = resolve_path(cfg, "viirs_root")
    ycfg = cfg["years"]
    expected_dmsp_years = year_range(ycfg["dmsp_start"], ycfg["dmsp_end"])
    expected_viirs_years = year_range(ycfg["viirs_start"], ycfg["viirs_end"])
    target_tiles = set(cfg["viirs"]["target_tiles"])

    anomalies: List[Dict] = []
    dmsp_files = collect_dmsp_files(dmsp_dir)
    missing_dmsp = sorted(set(expected_dmsp_years) - set(dmsp_files.keys()))
    extra_dmsp = sorted(set(dmsp_files.keys()) - set(expected_dmsp_years))
    for year in missing_dmsp:
        anomalies.append(
            {"level": "error", "scope": "dmsp", "year": year, "month": "", "issue": "missing_dmsp_year", "detail": ""}
        )
    for year in extra_dmsp:
        anomalies.append(
            {
                "level": "warn",
                "scope": "dmsp",
                "year": year,
                "month": "",
                "issue": "unexpected_dmsp_year",
                "detail": str(dmsp_files[year]),
            }
        )

    viirs_years_report = []
    for year in expected_viirs_years:
        year_dir = viirs_root / str(year)
        if not year_dir.exists():
            anomalies.append(
                {
                    "level": "error",
                    "scope": "viirs",
                    "year": year,
                    "month": "",
                    "issue": "missing_year_dir",
                    "detail": str(year_dir),
                }
            )
            continue

        month_dirs = sorted([d for d in year_dir.iterdir() if d.is_dir() and re.match(r"^\d{2}$", d.name)])
        month_names = [d.name for d in month_dirs]
        expected_months = monthly_expected(year)
        missing_months = sorted(set(expected_months) - set(month_names))
        for mm in missing_months:
            anomalies.append(
                {
                    "level": "error",
                    "scope": "viirs",
                    "year": year,
                    "month": mm,
                    "issue": "missing_month_dir",
                    "detail": "",
                }
            )

        months_report = []
        for month_dir in month_dirs:
            info = collect_month_files(month_dir, cfg)
            avg_tiles = set(info["avg_map"].keys())
            cf_tiles = set(info["cf_map"].keys())
            missing_avg_tiles = sorted(target_tiles - avg_tiles)
            missing_cf_tiles = sorted(target_tiles - cf_tiles)
            if missing_avg_tiles:
                anomalies.append(
                    {
                        "level": "error",
                        "scope": "viirs",
                        "year": year,
                        "month": month_dir.name,
                        "issue": "missing_avg_tiles",
                        "detail": ",".join(missing_avg_tiles),
                    }
                )
            if missing_cf_tiles:
                anomalies.append(
                    {
                        "level": "error",
                        "scope": "viirs",
                        "year": year,
                        "month": month_dir.name,
                        "issue": "missing_cf_tiles",
                        "detail": ",".join(missing_cf_tiles),
                    }
                )
            if info["unknown"]:
                anomalies.append(
                    {
                        "level": "warn",
                        "scope": "viirs",
                        "year": year,
                        "month": month_dir.name,
                        "issue": "unknown_tif_files",
                        "detail": ",".join(sorted(info["unknown"])[:20]),
                    }
                )
            if info["nested_tif_count"] > 0:
                anomalies.append(
                    {
                        "level": "info",
                        "scope": "viirs",
                        "year": year,
                        "month": month_dir.name,
                        "issue": "nested_tif_ignored",
                        "detail": str(info["nested_tif_count"]),
                    }
                )

            months_report.append(
                {
                    "month": month_dir.name,
                    "avg_tile_count": len(avg_tiles),
                    "cf_tile_count": len(cf_tiles),
                    "ignored_file_count": len(info["ignored"]),
                    "unknown_file_count": len(info["unknown"]),
                    "nested_tif_count": info["nested_tif_count"],
                }
            )

        viirs_years_report.append(
            {
                "year": year,
                "month_count": len(month_names),
                "months": month_names,
                "missing_months": missing_months,
                "month_reports": months_report,
            }
        )

    inventory_payload = {
        "created_at": datetime.now().isoformat(),
        "config_path": cfg.get("_config_path", ""),
        "dmsp": {
            "dir": str(dmsp_dir),
            "expected_years": expected_dmsp_years,
            "found_years": sorted(dmsp_files.keys()),
            "missing_years": missing_dmsp,
            "extra_years": extra_dmsp,
            "files": {str(y): str(p) for y, p in sorted(dmsp_files.items())},
        },
        "viirs": {
            "root": str(viirs_root),
            "expected_years": expected_viirs_years,
            "target_tiles": sorted(target_tiles),
            "year_reports": viirs_years_report,
        },
        "anomaly_count": len(anomalies),
    }

    source_inventory_path = inv_dir / "source_inventory.json"
    anomalies_path = inv_dir / "input_anomalies.csv"
    write_json(source_inventory_path, inventory_payload)
    write_csv(
        anomalies_path,
        anomalies,
        headers=["level", "scope", "year", "month", "issue", "detail"],
    )
    logger.info("Inventory written: %s", source_inventory_path)
    logger.info("Anomalies written: %s (%d rows)", anomalies_path, len(anomalies))
    return source_inventory_path, anomalies_path


def build_plan_markdown(cfg: Dict) -> str:
    overlap = cfg["years"]["overlap_fit"]
    return f"""# Nightlight Harmonization Plan v2

## Summary
- DMSP years: 2000-2013
- VIIRS years: 2012-2024
- Overlap years for fit: {overlap[0]} and {overlap[1]}
- Final output directory: reproduction_v6/02_processed_data/nightlight_harmonization_v1/09_harmonized_ntl

## Steps
1. inventory
2. save_plan
3. viirs_monthly_mosaic
4. viirs_annual_weighted
5. viirs_annual_denoised
6. dmsp_standardized
7. viirs_kd_log
8. sigmoid_fit
9. viirs_dn_simulated
10. temporal_consistency
11. harmonized_assemble
12. quality_report
"""


def save_plan_file(cfg: Dict, logger: logging.Logger) -> Path:
    logger.info("Step 2/12: write plan markdown")
    plan_output = resolve_path(cfg, "plan_output")
    ensure_dir(plan_output.parent)
    plan_output.write_text(build_plan_markdown(cfg), encoding="utf-8")
    logger.info("Plan written: %s", plan_output)
    return plan_output


def gdal_create_monthly_mosaic(
    input_files: Sequence[Path],
    out_path: Path,
    grid: GridSpec,
    resampling: str,
    cfg: Dict,
    logger: logging.Logger,
    gdalbuildvrt_cmd: str,
    gdalwarp_cmd: str,
    overwrite: bool,
) -> None:
    if out_path.exists() and not overwrite:
        logger.info("Skip existing mosaic: %s", out_path.name)
        return
    ensure_dir(out_path.parent)
    compression = cfg["runtime"]["compression"]
    bigtiff = cfg["runtime"]["bigtiff"]

    with tempfile.TemporaryDirectory(prefix="viirs_mosaic_") as tmpdir:
        vrt_path = Path(tmpdir) / "mosaic.vrt"
        cmd_vrt = [gdalbuildvrt_cmd, str(vrt_path)] + [str(p) for p in input_files]
        run_cmd(cmd_vrt, logger)
        cmd_warp = [
            gdalwarp_cmd,
            "-multi",
            "-wo",
            "NUM_THREADS=ALL_CPUS",
            "-t_srs",
            grid.crs,
            "-te",
            str(grid.lon_min),
            str(grid.lat_min),
            str(grid.lon_max),
            str(grid.lat_max),
            "-tr",
            str(grid.resolution),
            str(grid.resolution),
            "-r",
            resampling,
            "-dstnodata",
            str(grid.nodata),
            "-ot",
            "Float32",
            "-co",
            f"COMPRESS={compression}",
            "-co",
            "TILED=YES",
            "-co",
            f"BIGTIFF={bigtiff}",
            str(vrt_path),
            str(out_path),
        ]
        if overwrite:
            cmd_warp.insert(1, "-overwrite")
        run_cmd(cmd_warp, logger)


def step_viirs_monthly_mosaic(
    cfg: Dict,
    grid: GridSpec,
    logger: logging.Logger,
    gdalbuildvrt_cmd: str,
    gdalwarp_cmd: str,
    overwrite: bool,
) -> None:
    logger.info("Step 3/12: VIIRS monthly mosaics")
    viirs_root = resolve_path(cfg, "viirs_root")
    out_root = resolve_path(cfg, "output_root") / "01_viirs_monthly_mosaic"
    ycfg = cfg["years"]
    years = year_range(ycfg["viirs_start"], ycfg["viirs_end"])
    target_tiles = set(cfg["viirs"]["target_tiles"])

    for year in years:
        year_dir = viirs_root / str(year)
        if not year_dir.exists():
            logger.warning("VIIRS year dir missing, skip: %s", year_dir)
            continue
        month_dirs = sorted([d for d in year_dir.iterdir() if d.is_dir() and re.match(r"^\d{2}$", d.name)])
        year_out = out_root / str(year)
        ensure_dir(year_out)
        for month_dir in month_dirs:
            info = collect_month_files(month_dir, cfg)
            avg_map = info["avg_map"]
            cf_map = info["cf_map"]
            missing_avg = sorted(target_tiles - set(avg_map))
            missing_cf = sorted(target_tiles - set(cf_map))
            yyyymm = f"{year}{month_dir.name}"
            qc = {
                "year": year,
                "month": month_dir.name,
                "yyyymm": yyyymm,
                "avg_tile_count": len(avg_map),
                "cf_tile_count": len(cf_map),
                "missing_avg_tiles": missing_avg,
                "missing_cf_tiles": missing_cf,
                "ignored_file_count": len(info["ignored"]),
                "unknown_file_count": len(info["unknown"]),
                "nested_tif_ignored": info["nested_tif_count"],
                "status": "ok",
            }
            if missing_avg or missing_cf:
                qc["status"] = "skip_missing_tiles"
                write_json(year_out / f"viirs_{yyyymm}_mosaic_qc.json", qc)
                logger.warning("Skip %s due to missing tiles avg=%s cf=%s", yyyymm, missing_avg, missing_cf)
                continue

            avg_out = year_out / f"viirs_{yyyymm}_avg_rade9h_mosaic_30as.tif"
            cf_out = year_out / f"viirs_{yyyymm}_cf_cvg_mosaic_30as.tif"
            qc_path = year_out / f"viirs_{yyyymm}_mosaic_qc.json"

            avg_files = [avg_map[tile] for tile in sorted(target_tiles)]
            cf_files = [cf_map[tile] for tile in sorted(target_tiles)]

            gdal_create_monthly_mosaic(
                input_files=avg_files,
                out_path=avg_out,
                grid=grid,
                resampling=cfg["viirs"].get("resampling_avg", "average"),
                cfg=cfg,
                logger=logger,
                gdalbuildvrt_cmd=gdalbuildvrt_cmd,
                gdalwarp_cmd=gdalwarp_cmd,
                overwrite=overwrite,
            )
            gdal_create_monthly_mosaic(
                input_files=cf_files,
                out_path=cf_out,
                grid=grid,
                resampling=cfg["viirs"].get("resampling_cf", "sum"),
                cfg=cfg,
                logger=logger,
                gdalbuildvrt_cmd=gdalbuildvrt_cmd,
                gdalwarp_cmd=gdalwarp_cmd,
                overwrite=overwrite,
            )
            qc["avg_output"] = str(avg_out)
            qc["cf_output"] = str(cf_out)
            write_json(qc_path, qc)


def step_viirs_annual_weighted(cfg: Dict, grid: GridSpec, logger: logging.Logger, overwrite: bool) -> Path:
    logger.info("Step 4/12: VIIRS annual weighted composites")
    monthly_root = resolve_path(cfg, "output_root") / "01_viirs_monthly_mosaic"
    out_root = resolve_path(cfg, "output_root") / "02_viirs_annual_weighted"
    ensure_dir(out_root)
    rows: List[Dict] = []
    ycfg = cfg["years"]

    for year in year_range(ycfg["viirs_start"], ycfg["viirs_end"]):
        months = monthly_expected(year)
        month_pairs: List[Tuple[str, Path, Path]] = []
        for m in months:
            yyyymm = f"{year}{m}"
            avg = monthly_root / str(year) / f"viirs_{yyyymm}_avg_rade9h_mosaic_30as.tif"
            cf = monthly_root / str(year) / f"viirs_{yyyymm}_cf_cvg_mosaic_30as.tif"
            if avg.exists() and cf.exists():
                month_pairs.append((m, avg, cf))

        out_path = out_root / f"viirs_{year}_annual_weighted_30as.tif"
        if out_path.exists() and not overwrite:
            stats = raster_stats(out_path, grid.nodata)
            rows.append(
                {
                    "year": year,
                    "months_used": ",".join(m for m, _, _ in month_pairs),
                    "month_count": len(month_pairs),
                    **stats,
                    "status": "skipped_existing",
                }
            )
            logger.info("Skip existing weighted annual: %s", out_path.name)
            continue

        if not month_pairs:
            logger.warning("No valid monthly mosaics for year %d; skip annual weighted", year)
            rows.append(
                {
                    "year": year,
                    "months_used": "",
                    "month_count": 0,
                    "min": "",
                    "max": "",
                    "mean": "",
                    "std": "",
                    "zero_frac": "",
                    "count": 0,
                    "status": "missing_months",
                }
            )
            continue

        avg_datasets = [rasterio.open(p_avg) for _, p_avg, _ in month_pairs]
        cf_datasets = [rasterio.open(p_cf) for _, _, p_cf in month_pairs]
        try:
            profile = avg_datasets[0].profile.copy()
            profile.update(
                {
                    "driver": "GTiff",
                    "dtype": "float32",
                    "count": 1,
                    "crs": grid.crs,
                    "transform": grid.transform,
                    "width": grid.width,
                    "height": grid.height,
                    "nodata": grid.nodata,
                    "compress": cfg["runtime"]["compression"],
                    "tiled": True,
                    "blockxsize": grid.blockxsize,
                    "blockysize": grid.blockysize,
                    "BIGTIFF": cfg["runtime"]["bigtiff"],
                }
            )
            with rasterio.open(out_path, "w", **profile) as dst:
                for _, win in avg_datasets[0].block_windows(1):
                    h = int(win.height)
                    w = int(win.width)
                    ws = np.zeros((h, w), dtype=np.float64)
                    cs = np.zeros((h, w), dtype=np.float64)
                    for ds_avg, ds_cf in zip(avg_datasets, cf_datasets):
                        a = ds_avg.read(1, window=win).astype(np.float64)
                        c = ds_cf.read(1, window=win).astype(np.float64)
                        valid = np.isfinite(a) & np.isfinite(c)
                        if ds_avg.nodata is not None:
                            valid &= a != ds_avg.nodata
                        if ds_cf.nodata is not None:
                            valid &= c != ds_cf.nodata
                        c = np.where(valid, c, 0.0)
                        ws += np.where(valid, a * c, 0.0)
                        cs += c
                    out = np.where(cs > 0, ws / cs, 0.0).astype(np.float32)
                    dst.write(out, 1, window=win)
            stats = raster_stats(out_path, grid.nodata)
            rows.append(
                {
                    "year": year,
                    "months_used": ",".join(m for m, _, _ in month_pairs),
                    "month_count": len(month_pairs),
                    **stats,
                    "status": "ok",
                }
            )
            logger.info("Weighted annual complete: %s", out_path.name)
        finally:
            for ds in avg_datasets + cf_datasets:
                if not ds.closed:
                    ds.close()

    qc_path = out_root / "viirs_annual_weighted_qc.csv"
    write_csv(
        qc_path,
        rows,
        headers=["year", "months_used", "month_count", "min", "max", "mean", "std", "zero_frac", "count", "status"],
    )
    logger.info("Annual weighted QC written: %s", qc_path)
    return qc_path


def compute_high_lat_threshold(path: Path, grid: GridSpec, cfg: Dict) -> float:
    high_lat = float(cfg["noise"]["high_lat_abs_threshold"])
    sigma_mult = float(cfg["noise"]["aurora_sigma_multiplier"])
    total = 0.0
    total2 = 0.0
    count = 0

    with rasterio.open(path) as ds:
        for _, win in ds.block_windows(1):
            arr = ds.read(1, window=win).astype(np.float64)
            row_off = int(win.row_off)
            h = int(win.height)
            lats = grid.lat_max - (row_off + np.arange(h) + 0.5) * grid.resolution
            high_rows = np.where(np.abs(lats) >= high_lat)[0]
            if high_rows.size == 0:
                continue
            sub = arr[high_rows, :]
            vals = sub[np.isfinite(sub) & (sub > 0)]
            if vals.size == 0:
                continue
            total += float(vals.sum())
            total2 += float(np.square(vals).sum())
            count += int(vals.size)
    if count < 1000:
        return float("inf")
    mu = total / count
    sigma = math.sqrt(max(total2 / count - mu * mu, 0.0))
    return mu + sigma_mult * sigma


def step_viirs_annual_denoised(cfg: Dict, grid: GridSpec, logger: logging.Logger, overwrite: bool) -> Path:
    logger.info("Step 5/12: VIIRS annual denoising")
    in_root = resolve_path(cfg, "output_root") / "02_viirs_annual_weighted"
    out_root = resolve_path(cfg, "output_root") / "03_viirs_annual_denoised"
    ensure_dir(out_root)
    rows: List[Dict] = []
    low_thr = float(cfg["noise"]["low_radiance_threshold"])
    ycfg = cfg["years"]

    for year in year_range(ycfg["viirs_start"], ycfg["viirs_end"]):
        in_path = in_root / f"viirs_{year}_annual_weighted_30as.tif"
        out_path = out_root / f"viirs_{year}_annual_denoised_30as.tif"
        if not in_path.exists():
            logger.warning("Missing weighted annual input: %s", in_path)
            continue
        if out_path.exists() and not overwrite:
            stats = raster_stats(out_path, grid.nodata)
            rows.append(
                {
                    "year": year,
                    "aurora_threshold": "",
                    "low_removed": "",
                    "high_lat_removed": "",
                    **stats,
                    "status": "skipped_existing",
                }
            )
            logger.info("Skip existing denoised file: %s", out_path.name)
            continue

        aurora_thr = compute_high_lat_threshold(in_path, grid, cfg)
        low_removed = 0
        high_removed = 0

        with rasterio.open(in_path) as src:
            profile = src.profile.copy()
            profile.update(
                {
                    "dtype": "float32",
                    "nodata": grid.nodata,
                    "compress": cfg["runtime"]["compression"],
                    "tiled": True,
                    "blockxsize": grid.blockxsize,
                    "blockysize": grid.blockysize,
                    "BIGTIFF": cfg["runtime"]["bigtiff"],
                }
            )
            with rasterio.open(out_path, "w", **profile) as dst:
                for _, win in src.block_windows(1):
                    arr = src.read(1, window=win).astype(np.float32)
                    arr[arr < 0] = 0
                    low_mask = (arr > 0) & (arr < low_thr)
                    low_removed += int(np.count_nonzero(low_mask))
                    arr[low_mask] = 0

                    if np.isfinite(aurora_thr):
                        row_off = int(win.row_off)
                        h = int(win.height)
                        lats = grid.lat_max - (row_off + np.arange(h) + 0.5) * grid.resolution
                        high_rows = np.where(np.abs(lats) >= float(cfg["noise"]["high_lat_abs_threshold"]))[0]
                        if high_rows.size > 0:
                            sub = arr[high_rows, :]
                            mask = sub > aurora_thr
                            high_removed += int(np.count_nonzero(mask))
                            sub[mask] = 0
                            arr[high_rows, :] = sub
                    dst.write(arr, 1, window=win)

        stats = raster_stats(out_path, grid.nodata)
        rows.append(
            {
                "year": year,
                "aurora_threshold": "" if not np.isfinite(aurora_thr) else f"{aurora_thr:.6f}",
                "low_removed": low_removed,
                "high_lat_removed": high_removed,
                **stats,
                "status": "ok",
            }
        )
        logger.info("Denoised annual complete: %s", out_path.name)

    qc_path = out_root / "viirs_denoise_stats.csv"
    write_csv(
        qc_path,
        rows,
        headers=[
            "year",
            "aurora_threshold",
            "low_removed",
            "high_lat_removed",
            "min",
            "max",
            "mean",
            "std",
            "zero_frac",
            "count",
            "status",
        ],
    )
    logger.info("VIIRS denoise stats written: %s", qc_path)
    return qc_path


def gdal_warp_to_grid(
    src_path: Path,
    dst_path: Path,
    grid: GridSpec,
    cfg: Dict,
    logger: logging.Logger,
    gdalwarp_cmd: str,
    resampling: str,
    overwrite: bool,
) -> None:
    ensure_dir(dst_path.parent)
    cmd = [
        gdalwarp_cmd,
        "-multi",
        "-wo",
        "NUM_THREADS=ALL_CPUS",
        "-t_srs",
        grid.crs,
        "-te",
        str(grid.lon_min),
        str(grid.lat_min),
        str(grid.lon_max),
        str(grid.lat_max),
        "-tr",
        str(grid.resolution),
        str(grid.resolution),
        "-r",
        resampling,
        "-dstnodata",
        str(grid.nodata),
        "-ot",
        "Float32",
        "-co",
        f"COMPRESS={cfg['runtime']['compression']}",
        "-co",
        "TILED=YES",
        "-co",
        f"BIGTIFF={cfg['runtime']['bigtiff']}",
        str(src_path),
        str(dst_path),
    ]
    if overwrite:
        cmd.insert(1, "-overwrite")
    run_cmd(cmd, logger)


def clamp_raster(path: Path, min_v: float, max_v: float, nodata: float) -> None:
    with rasterio.open(path, "r+") as ds:
        for _, win in ds.block_windows(1):
            arr = ds.read(1, window=win).astype(np.float32)
            valid = np.isfinite(arr)
            if nodata is not None:
                valid &= arr != nodata
            arr[valid] = np.clip(arr[valid], min_v, max_v)
            ds.write(arr, 1, window=win)


def step_dmsp_standardized(
    cfg: Dict,
    grid: GridSpec,
    logger: logging.Logger,
    gdalwarp_cmd: str,
    overwrite: bool,
) -> Path:
    logger.info("Step 6/12: DMSP standardization (2000-2013)")
    out_root = resolve_path(cfg, "output_root") / "04_dmsp_standardized"
    ensure_dir(out_root)
    dmsp_dir = resolve_path(cfg, "dmsp_dir")
    dmsp_files = collect_dmsp_files(dmsp_dir)
    years = year_range(cfg["years"]["dmsp_start"], cfg["years"]["dmsp_end"])
    rows: List[Dict] = []

    for year in years:
        src = dmsp_files.get(year)
        if src is None:
            logger.warning("Missing DMSP year: %d", year)
            rows.append(
                {
                    "year": year,
                    "source": "",
                    "status": "missing_source",
                    "min": "",
                    "max": "",
                    "mean": "",
                    "std": "",
                    "zero_frac": "",
                    "count": "",
                }
            )
            continue
        dst = out_root / f"dmsp_{year}_dn_std_30as.tif"
        if dst.exists() and not overwrite:
            stats = raster_stats(dst, grid.nodata)
            rows.append({"year": year, "source": str(src), "status": "skipped_existing", **stats})
            logger.info("Skip existing DMSP standard: %s", dst.name)
            continue
        gdal_warp_to_grid(
            src_path=src,
            dst_path=dst,
            grid=grid,
            cfg=cfg,
            logger=logger,
            gdalwarp_cmd=gdalwarp_cmd,
            resampling="near",
            overwrite=overwrite,
        )
        clamp_raster(dst, min_v=0.0, max_v=63.0, nodata=grid.nodata)
        stats = raster_stats(dst, grid.nodata)
        rows.append({"year": year, "source": str(src), "status": "ok", **stats})
        logger.info("DMSP standardized: %s", dst.name)

    qc_path = out_root / "dmsp_standardized_qc.csv"
    write_csv(
        qc_path,
        rows,
        headers=["year", "source", "status", "min", "max", "mean", "std", "zero_frac", "count"],
    )
    logger.info("DMSP QC written: %s", qc_path)
    return qc_path


def clamp_window(win: Window, width: int, height: int) -> Window:
    col_off = max(0, int(win.col_off))
    row_off = max(0, int(win.row_off))
    w = int(min(win.width, width - col_off))
    h = int(min(win.height, height - row_off))
    return Window(col_off, row_off, max(0, w), max(0, h))


def gaussian_blur_blockwise(
    src_path: Path,
    dst_path: Path,
    sigma: float,
    cfg: Dict,
    logger: logging.Logger,
    overwrite: bool,
) -> None:
    if dst_path.exists() and not overwrite:
        logger.info("Skip existing KD raster: %s", dst_path.name)
        return
    ensure_dir(dst_path.parent)
    halo = max(1, int(math.ceil(3 * sigma)))
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update({"dtype": "float32", "compress": cfg["runtime"]["compression"], "tiled": True, "BIGTIFF": cfg["runtime"]["bigtiff"]})
        with rasterio.open(dst_path, "w", **profile) as dst:
            for _, win in src.block_windows(1):
                col0 = int(win.col_off)
                row0 = int(win.row_off)
                w = int(win.width)
                h = int(win.height)
                ewin = clamp_window(Window(col0 - halo, row0 - halo, w + 2 * halo, h + 2 * halo), src.width, src.height)
                arr = src.read(1, window=ewin).astype(np.float32)
                if src.nodata is not None:
                    arr[arr == src.nodata] = 0.0
                arr = np.nan_to_num(arr, nan=0.0)
                blurred = gaussian_filter(arr, sigma=sigma, mode="reflect")
                rr = row0 - int(ewin.row_off)
                cc = col0 - int(ewin.col_off)
                core = blurred[rr : rr + h, cc : cc + w].astype(np.float32)
                dst.write(core, 1, window=win)


def log1p_blockwise(src_path: Path, dst_path: Path, cfg: Dict, logger: logging.Logger, overwrite: bool) -> None:
    if dst_path.exists() and not overwrite:
        logger.info("Skip existing LogVKD raster: %s", dst_path.name)
        return
    ensure_dir(dst_path.parent)
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update({"dtype": "float32", "compress": cfg["runtime"]["compression"], "tiled": True, "BIGTIFF": cfg["runtime"]["bigtiff"]})
        with rasterio.open(dst_path, "w", **profile) as dst:
            for _, win in src.block_windows(1):
                arr = src.read(1, window=win).astype(np.float32)
                arr = np.nan_to_num(arr, nan=0.0)
                arr[arr < 0] = 0
                out = np.log1p(arr).astype(np.float32)
                dst.write(out, 1, window=win)


def step_viirs_kd_log(cfg: Dict, logger: logging.Logger, overwrite: bool) -> None:
    logger.info("Step 7/12: VIIRS KD + LogVKD")
    in_root = resolve_path(cfg, "output_root") / "03_viirs_annual_denoised"
    out_root = resolve_path(cfg, "output_root") / "05_viirs_kd_log"
    ensure_dir(out_root)
    sigma = float(cfg["noise"]["kd_sigma_pixels"])
    ycfg = cfg["years"]
    for year in year_range(ycfg["viirs_start"], ycfg["viirs_end"]):
        in_path = in_root / f"viirs_{year}_annual_denoised_30as.tif"
        if not in_path.exists():
            logger.warning("Missing VIIRS denoised annual: %s", in_path)
            continue
        kd_path = out_root / f"viirs_{year}_kd_30as.tif"
        log_path = out_root / f"viirs_{year}_logvkd_30as.tif"
        gaussian_blur_blockwise(in_path, kd_path, sigma=sigma, cfg=cfg, logger=logger, overwrite=overwrite)
        log1p_blockwise(kd_path, log_path, cfg=cfg, logger=logger, overwrite=overwrite)
        logger.info("KD+Log completed for year %d", year)


def sigmoid_func(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    return a + b / (1.0 + np.exp(c * (x - d)))


def wp_window_indices(grid: GridSpec, cfg: Dict) -> Window:
    wp = cfg["fitting"]["wp_extent"]
    col0 = int(math.floor((float(wp["lon_min"]) - grid.lon_min) / grid.resolution))
    col1 = int(math.ceil((float(wp["lon_max"]) - grid.lon_min) / grid.resolution))
    row0 = int(math.floor((grid.lat_max - float(wp["lat_max"])) / grid.resolution))
    row1 = int(math.ceil((grid.lat_max - float(wp["lat_min"])) / grid.resolution))
    return clamp_window(Window(col0, row0, col1 - col0, row1 - row0), grid.width, grid.height)


def load_overlap_samples(dmsp_path: Path, log_path: Path, window: Window, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    with rasterio.open(dmsp_path) as ds_d, rasterio.open(log_path) as ds_l:
        d = ds_d.read(1, window=window).astype(np.float32)
        x = ds_l.read(1, window=window).astype(np.float32)
    d = d[::stride, ::stride]
    x = x[::stride, ::stride]
    mask = np.isfinite(d) & np.isfinite(x) & (d >= 0) & (x >= 0)
    return x[mask].astype(np.float64), d[mask].astype(np.float64)


def fit_sigmoid(x: np.ndarray, y: np.ndarray, cfg: Dict) -> Tuple[np.ndarray, Dict[str, float]]:
    fit_cfg = cfg["fitting"]
    p0 = fit_cfg["sigmoid_initial"]
    init = [float(p0["a"]), float(p0["b"]), float(p0["c"]), float(p0["d"])]
    b = fit_cfg["sigmoid_bounds"]
    lower = [float(b["lower"]["a"]), float(b["lower"]["b"]), float(b["lower"]["c"]), float(b["lower"]["d"])]
    upper = [float(b["upper"]["a"]), float(b["upper"]["b"]), float(b["upper"]["c"]), float(b["upper"]["d"])]
    params, _ = curve_fit(sigmoid_func, x, y, p0=init, bounds=(lower, upper), maxfev=50000)
    pred = sigmoid_func(x, *params)
    residual = y - pred
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    mae = float(np.mean(np.abs(residual)))
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum(np.square(y - y_mean)))
    ss_res = float(np.sum(np.square(residual)))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return params, {"rmse": rmse, "mae": mae, "r2": r2}


def step_sigmoid_fit(cfg: Dict, grid: GridSpec, logger: logging.Logger) -> Tuple[Path, np.ndarray]:
    logger.info("Step 8/12: overlap-year sigmoid fitting (2012+2013)")
    dmsp_root = resolve_path(cfg, "output_root") / "04_dmsp_standardized"
    log_root = resolve_path(cfg, "output_root") / "05_viirs_kd_log"
    out_root = resolve_path(cfg, "output_root") / "06_sigmoid_fit"
    ensure_dir(out_root)

    overlap_years = [int(y) for y in cfg["years"]["overlap_fit"]]
    stride = int(cfg["fitting"].get("sample_stride", 8))
    win = wp_window_indices(grid, cfg)
    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    by_year_rows: List[Dict] = []
    diag_rows: List[Dict] = []

    for year in overlap_years:
        dmsp_path = dmsp_root / f"dmsp_{year}_dn_std_30as.tif"
        log_path = log_root / f"viirs_{year}_logvkd_30as.tif"
        if not dmsp_path.exists() or not log_path.exists():
            raise FileNotFoundError(f"Missing overlap inputs for year {year}: {dmsp_path}, {log_path}")
        x, y = load_overlap_samples(dmsp_path, log_path, win, stride)
        if x.size < 1000:
            raise RuntimeError(f"Insufficient overlap samples for year {year}: {x.size}")
        all_x.append(x)
        all_y.append(y)
        p_year, d_year = fit_sigmoid(x, y, cfg)
        by_year_rows.append(
            {
                "year": year,
                "a": float(p_year[0]),
                "b": float(p_year[1]),
                "c": float(p_year[2]),
                "d": float(p_year[3]),
                "samples": int(x.size),
                "rmse": d_year["rmse"],
                "mae": d_year["mae"],
                "r2": d_year["r2"],
            }
        )
        diag_rows.append({"scope": f"year_{year}", "samples": int(x.size), "rmse": d_year["rmse"], "mae": d_year["mae"], "r2": d_year["r2"]})

    x_all = np.concatenate(all_x, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    max_samples = 2_000_000
    if x_all.size > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(x_all.size, size=max_samples, replace=False)
        x_fit = x_all[idx]
        y_fit = y_all[idx]
    else:
        x_fit = x_all
        y_fit = y_all

    p_all, d_all = fit_sigmoid(x_fit, y_fit, cfg)
    diag_rows.append({"scope": "overlap_2012_2013_combined", "samples": int(x_fit.size), "rmse": d_all["rmse"], "mae": d_all["mae"], "r2": d_all["r2"]})

    fit_json = out_root / "sigmoid_wp_fit_overlap2012_2013.json"
    fit_csv = out_root / "sigmoid_wp_fit_by_year.csv"
    diag_csv = out_root / "sigmoid_fit_diagnostics.csv"
    write_json(
        fit_json,
        {
            "created_at": datetime.now().isoformat(),
            "overlap_years": overlap_years,
            "sample_stride": stride,
            "fit_window": {"col_off": int(win.col_off), "row_off": int(win.row_off), "width": int(win.width), "height": int(win.height)},
            "combined_fit": {"a": float(p_all[0]), "b": float(p_all[1]), "c": float(p_all[2]), "d": float(p_all[3]), "samples_used": int(x_fit.size), "diagnostics": d_all},
            "per_year_fit": by_year_rows,
        },
    )
    write_csv(fit_csv, by_year_rows, headers=["year", "a", "b", "c", "d", "samples", "rmse", "mae", "r2"])
    write_csv(diag_csv, diag_rows, headers=["scope", "samples", "rmse", "mae", "r2"])
    logger.info("Sigmoid fit written: %s", fit_json)
    return fit_json, p_all


def step_viirs_dn_simulated(cfg: Dict, logger: logging.Logger, params: np.ndarray, overwrite: bool) -> Path:
    logger.info("Step 9/12: apply sigmoid mapping to 2014-2024")
    in_root = resolve_path(cfg, "output_root") / "05_viirs_kd_log"
    out_root = resolve_path(cfg, "output_root") / "07_viirs_dn_simulated"
    ensure_dir(out_root)
    rows: List[Dict] = []
    ycfg = cfg["years"]
    years = year_range(ycfg["viirs_apply_start"], ycfg["viirs_apply_end"])

    for year in years:
        in_path = in_root / f"viirs_{year}_logvkd_30as.tif"
        out_path = out_root / f"dn_simulated_{year}_30as.tif"
        if not in_path.exists():
            logger.warning("Missing LogVKD input for year %d: %s", year, in_path)
            continue
        if out_path.exists() and not overwrite:
            stats = raster_stats(out_path, nodata=None)
            rows.append({"year": year, **stats, "status": "skipped_existing"})
            logger.info("Skip existing DN simulated: %s", out_path.name)
            continue

        with rasterio.open(in_path) as src:
            profile = src.profile.copy()
            profile.update({"dtype": "float32", "compress": cfg["runtime"]["compression"], "tiled": True, "BIGTIFF": cfg["runtime"]["bigtiff"]})
            with rasterio.open(out_path, "w", **profile) as dst:
                for _, win in src.block_windows(1):
                    x = src.read(1, window=win).astype(np.float32)
                    x = np.nan_to_num(x, nan=0.0)
                    dn = sigmoid_func(x.astype(np.float64), *params).astype(np.float32)
                    dn = np.clip(dn, 0, 63)
                    dst.write(dn, 1, window=win)
        stats = raster_stats(out_path, nodata=None)
        rows.append({"year": year, **stats, "status": "ok"})
        logger.info("DN simulated: %s", out_path.name)

    stats_path = out_root / "dn_simulated_stats.csv"
    write_csv(stats_path, rows, headers=["year", "min", "max", "mean", "std", "zero_frac", "count", "status"])
    logger.info("DN simulation stats: %s", stats_path)
    return stats_path


def step_temporal_consistency(cfg: Dict, logger: logging.Logger, overwrite: bool) -> Path:
    logger.info("Step 10/12: temporal consistency correction (2013/2014 focus)")
    sim_root = resolve_path(cfg, "output_root") / "07_viirs_dn_simulated"
    dmsp_root = resolve_path(cfg, "output_root") / "04_dmsp_standardized"
    out_root = resolve_path(cfg, "output_root") / "08_temporal_consistency"
    ensure_dir(out_root)
    tcfg = cfg["temporal"]
    tol = float(tcfg.get("drop_tolerance", 0.0))
    enforce = bool(tcfg.get("enforce_non_decreasing", True))
    alpha = float(tcfg.get("smoothing_alpha", 0.5))
    ycfg = cfg["years"]
    years = year_range(ycfg["viirs_apply_start"], ycfg["viirs_apply_end"])
    rows: List[Dict] = []

    prev_path: Optional[Path] = dmsp_root / "dmsp_2013_dn_std_30as.tif"
    if not prev_path.exists():
        raise FileNotFoundError(f"Temporal correction requires DMSP 2013: {prev_path}")

    for year in years:
        curr_path = sim_root / f"dn_simulated_{year}_30as.tif"
        out_path = out_root / f"dn_tscorr_{year}_30as.tif"
        if not curr_path.exists():
            logger.warning("Missing simulated DN for year %d: %s", year, curr_path)
            continue
        if out_path.exists() and not overwrite:
            stats = raster_stats(out_path, nodata=None)
            rows.append({"year": year, "prev_source": str(prev_path), "corrected_pixels": "", "total_pixels": "", "corrected_frac": "", **stats, "status": "skipped_existing"})
            prev_path = out_path
            logger.info("Skip existing temporal corrected: %s", out_path.name)
            continue

        corrected_pixels = 0
        total_pixels = 0
        with rasterio.open(prev_path) as ds_prev, rasterio.open(curr_path) as ds_curr:
            profile = ds_curr.profile.copy()
            profile.update({"dtype": "float32", "compress": cfg["runtime"]["compression"], "tiled": True, "BIGTIFF": cfg["runtime"]["bigtiff"]})
            with rasterio.open(out_path, "w", **profile) as dst:
                for _, win in ds_curr.block_windows(1):
                    prev = ds_prev.read(1, window=win).astype(np.float32)
                    curr = ds_curr.read(1, window=win).astype(np.float32)
                    valid = np.isfinite(prev) & np.isfinite(curr)
                    corrected = curr.copy()
                    drop_mask = valid & (curr + tol < prev)
                    if enforce:
                        corrected[drop_mask] = prev[drop_mask]
                    else:
                        corrected[drop_mask] = curr[drop_mask] + alpha * (prev[drop_mask] - curr[drop_mask])
                    corrected = np.clip(corrected, 0, 63)
                    corrected_pixels += int(np.count_nonzero(drop_mask))
                    total_pixels += int(np.count_nonzero(valid))
                    dst.write(corrected, 1, window=win)

        stats = raster_stats(out_path, nodata=None)
        rows.append(
            {
                "year": year,
                "prev_source": str(prev_path),
                "corrected_pixels": corrected_pixels,
                "total_pixels": total_pixels,
                "corrected_frac": corrected_pixels / total_pixels if total_pixels else float("nan"),
                **stats,
                "status": "ok",
            }
        )
        logger.info("Temporal corrected: %s (corrected=%d)", out_path.name, corrected_pixels)
        prev_path = out_path

    log_path = out_root / "temporal_corrections_log.csv"
    write_csv(log_path, rows, headers=["year", "prev_source", "corrected_pixels", "total_pixels", "corrected_frac", "min", "max", "mean", "std", "zero_frac", "count", "status"])
    logger.info("Temporal correction log: %s", log_path)
    return log_path


def maybe_hardlink(src: Path, dst: Path) -> str:
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)
        return "hardlink"
    except Exception:
        shutil.copy2(src, dst)
        return "copy"


def step_harmonized_assemble(cfg: Dict, logger: logging.Logger, overwrite: bool) -> Tuple[Path, Path]:
    logger.info("Step 11/12: assemble harmonized dataset 2000-2024")
    dmsp_root = resolve_path(cfg, "output_root") / "04_dmsp_standardized"
    corr_root = resolve_path(cfg, "output_root") / "08_temporal_consistency"
    out_root = resolve_path(cfg, "output_root") / "09_harmonized_ntl"
    ensure_dir(out_root)
    ycfg = cfg["years"]
    years = year_range(ycfg["dmsp_start"], ycfg["viirs_end"])
    manifest_rows: List[Dict] = []
    summary_rows: List[Dict] = []

    for year in years:
        if year <= ycfg["dmsp_end"]:
            src = dmsp_root / f"dmsp_{year}_dn_std_30as.tif"
            method = "dmsp_standardized"
        else:
            src = corr_root / f"dn_tscorr_{year}_30as.tif"
            method = "viirs_simulated_temporal_corrected"
        dst = out_root / f"ntl_harmonized_{year}_dn_30as.tif"
        if not src.exists():
            manifest_rows.append({"year": year, "source": str(src), "output": str(dst), "method": method, "link_mode": "missing_source", "status": "missing"})
            continue
        if dst.exists() and overwrite:
            dst.unlink()
        link_mode = maybe_hardlink(src, dst)
        stats = raster_stats(dst, nodata=None)
        manifest_rows.append({"year": year, "source": str(src), "output": str(dst), "method": method, "link_mode": link_mode, "status": "ok"})
        summary_rows.append({"year": year, **stats, "method": method})

    manifest_json = out_root / "harmonized_manifest.json"
    summary_csv = out_root / "harmonized_yearly_summary.csv"
    write_json(manifest_json, {"created_at": datetime.now().isoformat(), "rows": manifest_rows})
    write_csv(summary_csv, summary_rows, headers=["year", "method", "min", "max", "mean", "std", "zero_frac", "count"])
    logger.info("Harmonized manifest: %s", manifest_json)
    logger.info("Harmonized summary: %s", summary_csv)
    return manifest_json, summary_csv


def js_distance(a: np.ndarray, b: np.ndarray, bins: int = 256) -> float:
    hist_range = (0.0, 63.0)
    pa, _ = np.histogram(a, bins=bins, range=hist_range)
    pb, _ = np.histogram(b, bins=bins, range=hist_range)
    pa = pa.astype(np.float64) + 1e-12
    pb = pb.astype(np.float64) + 1e-12
    pa /= pa.sum()
    pb /= pb.sum()
    m = 0.5 * (pa + pb)
    kld1 = np.sum(pa * np.log(pa / m))
    kld2 = np.sum(pb * np.log(pb / m))
    return float(np.sqrt(0.5 * (kld1 + kld2)))


def load_strided_samples(path: Path, stride: int = 32) -> np.ndarray:
    with rasterio.open(path) as ds:
        arr = ds.read(1)
    arr = arr[::stride, ::stride]
    vals = arr[np.isfinite(arr)]
    return vals.astype(np.float64)


def step_quality_report(cfg: Dict, logger: logging.Logger) -> Tuple[Path, Path, Path]:
    logger.info("Step 12/12: quality metrics and report")
    harmonized_root = resolve_path(cfg, "output_root") / "09_harmonized_ntl"
    report_root = resolve_path(cfg, "report_root")
    ensure_dir(report_root)
    summary_csv = harmonized_root / "harmonized_yearly_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing harmonized summary: {summary_csv}")

    rows: List[Dict] = []
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            year = int(r["year"])
            min_v = float(r["min"])
            max_v = float(r["max"])
            rows.append(
                {
                    "year": year,
                    "min": min_v,
                    "max": max_v,
                    "mean": float(r["mean"]),
                    "std": float(r["std"]),
                    "zero_frac": float(r["zero_frac"]),
                    "count": int(float(r["count"])),
                    "range_pass": (min_v >= -1e-6 and max_v <= 63.000001),
                }
            )

    qc_csv = report_root / "harmonized_qc_metrics.csv"
    write_csv(qc_csv, rows, headers=["year", "min", "max", "mean", "std", "zero_frac", "count", "range_pass"])

    y_prev = 2013
    y_curr = 2014
    p_prev = harmonized_root / f"ntl_harmonized_{y_prev}_dn_30as.tif"
    p_curr = harmonized_root / f"ntl_harmonized_{y_curr}_dn_30as.tif"
    if p_prev.exists() and p_curr.exists():
        prev_vals = load_strided_samples(p_prev, stride=32)
        curr_vals = load_strided_samples(p_curr, stride=32)
        n = min(prev_vals.size, curr_vals.size)
        prev_vals = prev_vals[:n]
        curr_vals = curr_vals[:n]
        mean_ratio = float(curr_vals.mean() / max(prev_vals.mean(), 1e-9))
        corr = float(np.corrcoef(prev_vals, curr_vals)[0, 1]) if n > 1 else float("nan")
        js = js_distance(prev_vals, curr_vals)
    else:
        mean_ratio = float("nan")
        corr = float("nan")
        js = float("nan")

    transition_csv = report_root / "harmonized_transition_2013_2014.csv"
    write_csv(
        transition_csv,
        [{"year_prev": y_prev, "year_curr": y_curr, "mean_ratio_curr_over_prev": mean_ratio, "correlation": corr, "js_distance": js}],
        headers=["year_prev", "year_curr", "mean_ratio_curr_over_prev", "correlation", "js_distance"],
    )

    all_range_pass = all(bool(r["range_pass"]) for r in rows)
    report_md = report_root / "harmonized_ntl_report.md"
    report_text = "\n".join(
        [
            "# Harmonized NTL Report",
            "",
            f"- created_at: {datetime.now().isoformat()}",
            f"- years_covered: {min(r['year'] for r in rows)}-{max(r['year'] for r in rows)}" if rows else "- years_covered: n/a",
            f"- range_check_pass: {all_range_pass}",
            f"- transition_2013_2014_mean_ratio: {mean_ratio}",
            f"- transition_2013_2014_corr: {corr}",
            f"- transition_2013_2014_js_distance: {js}",
            "",
            "## Files",
            f"- {qc_csv}",
            f"- {transition_csv}",
        ]
    )
    report_md.write_text(report_text, encoding="utf-8")
    logger.info("Quality metrics: %s", qc_csv)
    logger.info("Transition metrics: %s", transition_csv)
    logger.info("Report markdown: %s", report_md)
    return qc_csv, transition_csv, report_md


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = project_root() / cfg_path
    cfg = load_config(cfg_path)
    cfg["runtime"]["overwrite"] = bool(args.overwrite or cfg["runtime"].get("overwrite", False))
    overwrite = bool(cfg["runtime"]["overwrite"])

    log_file = resolve_path(cfg, "output_root") / "00_inventory" / "harmonization_run.log"
    logger = setup_logger(log_file)
    logger.info("=" * 72)
    logger.info("Nightlight harmonization v2 started")
    logger.info("Config: %s", cfg_path)
    logger.info("Overwrite: %s", overwrite)
    logger.info("=" * 72)

    grid = grid_spec(cfg)
    steps = parse_steps(args.steps)
    logger.info("Steps: %s", steps)

    fit_params: Optional[np.ndarray] = None
    fit_json_path = resolve_path(cfg, "output_root") / "06_sigmoid_fit" / "sigmoid_wp_fit_overlap2012_2013.json"

    for step in steps:
        if step == "inventory":
            inventory_inputs(cfg, logger)
        elif step == "save_plan":
            save_plan_file(cfg, logger)
        elif step == "viirs_monthly_mosaic":
            step_viirs_monthly_mosaic(
                cfg=cfg,
                grid=grid,
                logger=logger,
                gdalbuildvrt_cmd=args.gdalbuildvrt,
                gdalwarp_cmd=args.gdalwarp,
                overwrite=overwrite,
            )
        elif step == "viirs_annual_weighted":
            step_viirs_annual_weighted(cfg, grid, logger, overwrite)
        elif step == "viirs_annual_denoised":
            step_viirs_annual_denoised(cfg, grid, logger, overwrite)
        elif step == "dmsp_standardized":
            step_dmsp_standardized(cfg, grid, logger, args.gdalwarp, overwrite)
        elif step == "viirs_kd_log":
            step_viirs_kd_log(cfg, logger, overwrite)
        elif step == "sigmoid_fit":
            _, fit_params = step_sigmoid_fit(cfg, grid, logger)
        elif step == "viirs_dn_simulated":
            if fit_params is None:
                if fit_json_path.exists():
                    payload = json.loads(fit_json_path.read_text(encoding="utf-8"))
                    fit = payload["combined_fit"]
                    fit_params = np.array([fit["a"], fit["b"], fit["c"], fit["d"]], dtype=np.float64)
                    logger.info("Loaded existing fit params from %s", fit_json_path)
                else:
                    raise RuntimeError("Missing fit params. Run sigmoid_fit first.")
            step_viirs_dn_simulated(cfg, logger, fit_params, overwrite)
        elif step == "temporal_consistency":
            step_temporal_consistency(cfg, logger, overwrite)
        elif step == "harmonized_assemble":
            step_harmonized_assemble(cfg, logger, overwrite)
        elif step == "quality_report":
            step_quality_report(cfg, logger)

    logger.info("=" * 72)
    logger.info("Nightlight harmonization v2 completed")
    logger.info("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
