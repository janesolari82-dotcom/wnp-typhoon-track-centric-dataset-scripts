#!/usr/bin/env python3
"""LitPop asset allocation pipeline (2000-2024, global full extent)."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.windows import Window


STEP_ORDER = [
    "inventory",
    "save_plan",
    "prepare_gdp",
    "compute_denominator",
    "allocate_litpop",
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
    width: int
    height: int
    nodata: float
    dtype: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LitPop asset allocation (2000-2024)")
    parser.add_argument(
        "--config",
        type=str,
        default="reproduction_v6/config/litpop_2000_2024.yaml",
        help="Config path",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="all",
        help="Comma-separated step list or 'all'",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    return parser.parse_args()


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("litpop_2000_2024")
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


def parse_steps(steps_arg: str) -> List[str]:
    if steps_arg.strip().lower() == "all":
        return STEP_ORDER
    steps = [s.strip() for s in steps_arg.split(",") if s.strip()]
    unknown = [s for s in steps if s not in STEP_ORDER]
    if unknown:
        raise ValueError(f"Unknown steps: {unknown}; valid={STEP_ORDER}")
    return steps


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_rows_csv(path: Path, rows: Sequence[Dict], headers: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(headers))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_df_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8")


def all_exist(paths: Iterable[Path]) -> bool:
    return all(p.exists() for p in paths)


def year_list(cfg: Dict) -> List[int]:
    start = int(cfg["years"]["start"])
    end = int(cfg["years"]["end"])
    return list(range(start, end + 1))


def grid_spec(cfg: Dict) -> GridSpec:
    g = cfg["grid"]
    return GridSpec(
        crs=str(g["crs"]),
        resolution=float(g["resolution_deg"]),
        lon_min=float(g["lon_min"]),
        lon_max=float(g["lon_max"]),
        lat_min=float(g["lat_min"]),
        lat_max=float(g["lat_max"]),
        width=int(g["width"]),
        height=int(g["height"]),
        nodata=float(g["nodata"]),
        dtype=str(g["dtype"]),
    )


def output_paths(cfg: Dict) -> Dict[str, Path]:
    root = resolve_path(cfg, "output_root")
    return {
        "root": root,
        "inventory_dir": root / "00_inventory",
        "annual_dir": root / "01_litpop_annual",
        "table_dir": root / "02_tables",
        "report_dir": root / "06_report",
    }


def inventory_files(cfg: Dict) -> Dict[str, Path]:
    out = output_paths(cfg)
    return {
        "source_inventory": out["inventory_dir"] / "source_inventory.json",
        "input_anomalies": out["inventory_dir"] / "input_anomalies.csv",
        "run_log": out["inventory_dir"] / "litpop_2000_2024_run.log",
    }


def annual_files(cfg: Dict) -> Dict[str, Path]:
    out = output_paths(cfg)
    return {
        "manifest": out["annual_dir"] / "litpop_manifest.json",
        "summary": out["annual_dir"] / "litpop_yearly_summary.csv",
    }


def table_files(cfg: Dict) -> Dict[str, Path]:
    out = output_paths(cfg)
    return {
        "gdp_long": out["table_dir"] / "gdp_country_2000_2024_long.csv",
        "gdp_wide": out["table_dir"] / "gdp_country_2000_2024_wide.csv",
        "missing_gdp": out["table_dir"] / "missing_gdp_country_year.csv",
        "denominator": out["table_dir"] / "litpop_denominator_country_year.csv",
        "zero_denominator": out["table_dir"] / "zero_denominator_country_year.csv",
    }


def report_files(cfg: Dict) -> Dict[str, Path]:
    out = output_paths(cfg)
    return {
        "country_conservation": out["report_dir"] / "litpop_country_conservation.csv",
        "qc_metrics": out["report_dir"] / "litpop_qc_metrics.csv",
        "qa_summary": out["report_dir"] / "qa_summary.json",
        "report_md": out["report_dir"] / "litpop_2000_2024_report.md",
    }


def build_anomaly(level: str, scope: str, item: str, issue: str, detail: str) -> Dict:
    return {"level": level, "scope": scope, "item": item, "issue": issue, "detail": detail}


def ntl_file_map(cfg: Dict) -> Dict[int, Path]:
    root = resolve_path(cfg, "ntl_root")
    pat = str(cfg["inputs"]["ntl_pattern"])
    return {y: root / pat.format(year=y) for y in year_list(cfg)}


def pop_file_map(cfg: Dict) -> Dict[int, Path]:
    root = resolve_path(cfg, "pop_root")
    pat = str(cfg["inputs"]["pop_pattern"])
    return {y: root / pat.format(year=y) for y in year_list(cfg)}


def almost_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def read_raster_meta(path: Path) -> Dict:
    with rasterio.open(path) as ds:
        t = ds.transform
        return {
            "path": str(path),
            "width": int(ds.width),
            "height": int(ds.height),
            "crs": str(ds.crs) if ds.crs is not None else "",
            "nodata": None if ds.nodata is None else float(ds.nodata),
            "dtype": str(ds.dtypes[0]),
            "transform": [float(t.a), float(t.b), float(t.c), float(t.d), float(t.e), float(t.f)],
            "res_x": float(t.a),
            "res_y": float(abs(t.e)),
            "bounds": {
                "left": float(ds.bounds.left),
                "bottom": float(ds.bounds.bottom),
                "right": float(ds.bounds.right),
                "top": float(ds.bounds.top),
            },
        }


def validate_series_metas(
    meta_by_year: Dict[int, Dict],
    ref_expected: Dict,
    label: str,
    anomalies: List[Dict],
) -> None:
    years = sorted(meta_by_year.keys())
    if not years:
        return

    ref = meta_by_year[years[0]]
    for year in years[1:]:
        m = meta_by_year[year]
        if m["crs"] != ref["crs"]:
            anomalies.append(build_anomaly("error", label, str(year), "crs_mismatch", f"{m['crs']} vs {ref['crs']}"))
        if m["width"] != ref["width"] or m["height"] != ref["height"]:
            anomalies.append(
                build_anomaly(
                    "error",
                    label,
                    str(year),
                    "shape_mismatch",
                    f"{m['width']}x{m['height']} vs {ref['width']}x{ref['height']}",
                )
            )
        if m["nodata"] != ref["nodata"]:
            anomalies.append(
                build_anomaly(
                    "error",
                    label,
                    str(year),
                    "nodata_mismatch",
                    f"{m['nodata']} vs {ref['nodata']}",
                )
            )
        for idx, (a, b) in enumerate(zip(m["transform"], ref["transform"])):
            if not almost_equal(a, b, tol=1e-12):
                anomalies.append(
                    build_anomaly(
                        "error",
                        label,
                        str(year),
                        "transform_mismatch",
                        f"idx={idx}, {a} vs {b}",
                    )
                )
                break

    exp = ref_expected
    if ref["crs"] != str(exp["crs"]):
        anomalies.append(build_anomaly("error", label, "reference", "unexpected_crs", ref["crs"]))
    if ref["width"] != int(exp["width"]) or ref["height"] != int(exp["height"]):
        anomalies.append(
            build_anomaly(
                "error",
                label,
                "reference",
                "unexpected_shape",
                f"{ref['width']}x{ref['height']} expected {exp['width']}x{exp['height']}",
            )
        )
    if not almost_equal(ref["res_x"], float(exp["resolution"]), tol=1e-12) or not almost_equal(
        ref["res_y"], float(exp["resolution"]), tol=1e-12
    ):
        anomalies.append(
            build_anomaly(
                "error",
                label,
                "reference",
                "unexpected_resolution",
                f"{ref['res_x']},{ref['res_y']} expected {exp['resolution']}",
            )
        )
    b = ref["bounds"]
    if not almost_equal(b["left"], float(exp["lon_min"]), tol=1e-9):
        anomalies.append(build_anomaly("error", label, "reference", "unexpected_left", str(b["left"])))
    if not almost_equal(b["right"], float(exp["lon_max"]), tol=1e-9):
        anomalies.append(build_anomaly("error", label, "reference", "unexpected_right", str(b["right"])))
    if not almost_equal(b["bottom"], float(exp["lat_min"]), tol=1e-9):
        anomalies.append(build_anomaly("error", label, "reference", "unexpected_bottom", str(b["bottom"])))
    if not almost_equal(b["top"], float(exp["lat_max"]), tol=1e-9):
        anomalies.append(build_anomaly("error", label, "reference", "unexpected_top", str(b["top"])))


def scan_country_ids(path: Path) -> Dict:
    unique_ids: set[int] = set()
    min_id = None
    max_id = None
    with rasterio.open(path) as ds:
        for _, window in ds.block_windows(1):
            arr = ds.read(1, window=window).astype(np.int64, copy=False)
            vals = np.unique(arr)
            for v in vals.tolist():
                vv = int(v)
                if vv <= 0:
                    continue
                unique_ids.add(vv)
                if min_id is None or vv < min_id:
                    min_id = vv
                if max_id is None or vv > max_id:
                    max_id = vv
    return {
        "unique_count": int(len(unique_ids)),
        "min_id": 0 if min_id is None else int(min_id),
        "max_id": 0 if max_id is None else int(max_id),
    }


def load_worldbank_header(gdp_csv: Path) -> Tuple[int, List[str]]:
    with gdp_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if len(row) >= 2 and row[0] == "Country Name" and row[1] == "Country Code":
                return idx, row
    raise ValueError(f"Cannot find header row in GDP CSV: {gdp_csv}")


def collect_inventory(cfg: Dict, logger: logging.Logger) -> Tuple[Path, Path, List[Dict], Dict]:
    invf = inventory_files(cfg)
    years = year_list(cfg)
    g = grid_spec(cfg)
    ntl_win = cfg["ntl_window"]
    ntl_row_off = int(ntl_win["global_row_offset"])
    ntl_height = int(ntl_win["height"])

    anomalies: List[Dict] = []
    ntl_files = ntl_file_map(cfg)
    pop_files = pop_file_map(cfg)
    country_raster = resolve_path(cfg, "country_id_raster")
    country_lookup = resolve_path(cfg, "country_lookup_csv")
    gdp_csv = resolve_path(cfg, "gdp_csv")

    ntl_meta: Dict[int, Dict] = {}
    pop_meta: Dict[int, Dict] = {}

    for year in years:
        ntl = ntl_files[year]
        pop = pop_files[year]
        if not ntl.exists():
            anomalies.append(build_anomaly("error", "input", f"ntl_{year}", "missing_file", str(ntl)))
        if not pop.exists():
            anomalies.append(build_anomaly("error", "input", f"pop_{year}", "missing_file", str(pop)))
        if ntl.exists():
            try:
                ntl_meta[year] = read_raster_meta(ntl)
            except Exception as exc:
                anomalies.append(build_anomaly("error", "input", f"ntl_{year}", "unreadable", str(exc)))
        if pop.exists():
            try:
                pop_meta[year] = read_raster_meta(pop)
            except Exception as exc:
                anomalies.append(build_anomaly("error", "input", f"pop_{year}", "unreadable", str(exc)))

    ntl_expected = {
        "crs": g.crs,
        "width": g.width,
        "height": ntl_height,
        "resolution": g.resolution,
        "lon_min": g.lon_min,
        "lon_max": g.lon_max,
        "lat_min": -65.0,
        "lat_max": 75.0,
    }
    pop_expected = {
        "crs": g.crs,
        "width": g.width,
        "height": g.height,
        "resolution": g.resolution,
        "lon_min": g.lon_min,
        "lon_max": g.lon_max,
        "lat_min": g.lat_min,
        "lat_max": g.lat_max,
    }

    validate_series_metas(ntl_meta, ntl_expected, "ntl", anomalies)
    validate_series_metas(pop_meta, pop_expected, "population", anomalies)

    country_meta = {}
    country_id_stats = {}
    if not country_raster.exists():
        anomalies.append(build_anomaly("error", "input", "country_id_raster", "missing_file", str(country_raster)))
    else:
        try:
            country_meta = read_raster_meta(country_raster)
            validate_series_metas({0: country_meta}, pop_expected, "country_id_raster", anomalies)
            if country_meta.get("nodata") != 0.0:
                anomalies.append(
                    build_anomaly(
                        "error",
                        "country_id_raster",
                        "nodata",
                        "unexpected_nodata",
                        str(country_meta.get("nodata")),
                    )
                )
            country_id_stats = scan_country_ids(country_raster)
        except Exception as exc:
            anomalies.append(build_anomaly("error", "country_id_raster", "read", "unreadable", str(exc)))

    lookup_info = {}
    if not country_lookup.exists():
        anomalies.append(build_anomaly("error", "input", "country_lookup_csv", "missing_file", str(country_lookup)))
    else:
        try:
            lookup = pd.read_csv(country_lookup)
            required_cols = {"country_id", "iso3_final"}
            missing_cols = [c for c in required_cols if c not in lookup.columns]
            if missing_cols:
                anomalies.append(
                    build_anomaly("error", "country_lookup_csv", "columns", "missing_columns", ",".join(missing_cols))
                )
            else:
                lookup_ids = lookup["country_id"].astype(int)
                lookup_info = {
                    "row_count": int(len(lookup)),
                    "unique_count": int(lookup_ids.nunique()),
                    "min_id": int(lookup_ids.min()),
                    "max_id": int(lookup_ids.max()),
                }
                if country_id_stats:
                    if lookup_info["unique_count"] != country_id_stats["unique_count"]:
                        anomalies.append(
                            build_anomaly(
                                "error",
                                "country_lookup_csv",
                                "country_id",
                                "unique_count_mismatch",
                                f"lookup={lookup_info['unique_count']}, raster={country_id_stats['unique_count']}",
                            )
                        )
                    if lookup_info["min_id"] != country_id_stats["min_id"]:
                        anomalies.append(
                            build_anomaly(
                                "error",
                                "country_lookup_csv",
                                "country_id",
                                "min_id_mismatch",
                                f"lookup={lookup_info['min_id']}, raster={country_id_stats['min_id']}",
                            )
                        )
                    if lookup_info["max_id"] != country_id_stats["max_id"]:
                        anomalies.append(
                            build_anomaly(
                                "error",
                                "country_lookup_csv",
                                "country_id",
                                "max_id_mismatch",
                                f"lookup={lookup_info['max_id']}, raster={country_id_stats['max_id']}",
                            )
                        )
        except Exception as exc:
            anomalies.append(build_anomaly("error", "country_lookup_csv", "read", "unreadable", str(exc)))

    gdp_header_idx = None
    gdp_header_cols: List[str] = []
    if not gdp_csv.exists():
        anomalies.append(build_anomaly("error", "input", "gdp_csv", "missing_file", str(gdp_csv)))
    else:
        try:
            gdp_header_idx, gdp_header_cols = load_worldbank_header(gdp_csv)
            required = ["Country Name", "Country Code", "Indicator Code"] + [str(y) for y in years]
            missing = [c for c in required if c not in gdp_header_cols]
            if missing:
                anomalies.append(
                    build_anomaly("error", "gdp_csv", "header", "missing_required_columns", ",".join(missing))
                )
        except Exception as exc:
            anomalies.append(build_anomaly("error", "gdp_csv", "header", "invalid", str(exc)))

    if ntl_row_off < 0 or ntl_height <= 0 or (ntl_row_off + ntl_height > g.height):
        anomalies.append(
            build_anomaly(
                "error",
                "ntl_window",
                "window",
                "invalid_window",
                f"row_off={ntl_row_off}, height={ntl_height}, grid_height={g.height}",
            )
        )

    payload = {
        "created_at": datetime.now().isoformat(),
        "config_path": cfg.get("_config_path", ""),
        "years": years,
        "inputs": {
            "ntl_root": str(resolve_path(cfg, "ntl_root")),
            "pop_root": str(resolve_path(cfg, "pop_root")),
            "gdp_csv": str(gdp_csv),
            "country_id_raster": str(country_raster),
            "country_lookup_csv": str(country_lookup),
            "methodology_doc": str(resolve_path(cfg, "methodology_doc")),
            "reference_pdf": str(resolve_path(cfg, "reference_pdf")),
        },
        "ntl_window": {"global_row_offset": ntl_row_off, "height": ntl_height},
        "ntl_meta_by_year": {str(y): ntl_meta[y] for y in sorted(ntl_meta.keys())},
        "pop_meta_by_year": {str(y): pop_meta[y] for y in sorted(pop_meta.keys())},
        "country_id_meta": country_meta,
        "country_id_raster_stats": country_id_stats,
        "country_lookup_stats": lookup_info,
        "gdp_header_row_index_0based": gdp_header_idx,
        "gdp_header_column_count": len(gdp_header_cols),
        "anomaly_count": len(anomalies),
        "error_count": sum(1 for a in anomalies if a["level"] == "error"),
    }
    write_json(invf["source_inventory"], payload)
    write_rows_csv(
        invf["input_anomalies"],
        anomalies,
        headers=["level", "scope", "item", "issue", "detail"],
    )
    logger.info("Inventory written: %s", invf["source_inventory"])
    logger.info("Anomalies written: %s (%d rows)", invf["input_anomalies"], len(anomalies))
    return invf["source_inventory"], invf["input_anomalies"], anomalies, payload


def step_inventory(cfg: Dict, logger: logging.Logger, overwrite: bool) -> Tuple[Path, Path]:
    logger.info("Step 1/6: inventory")
    invf = inventory_files(cfg)
    if not overwrite and all_exist([invf["source_inventory"], invf["input_anomalies"]]):
        logger.info("Skip existing inventory outputs (overwrite=false).")
        return invf["source_inventory"], invf["input_anomalies"]

    _, _, anomalies, _ = collect_inventory(cfg, logger)
    err_count = sum(1 for a in anomalies if a["level"] == "error")
    if err_count > 0:
        raise RuntimeError(f"Inventory failed with {err_count} error(s).")
    return invf["source_inventory"], invf["input_anomalies"]


def build_plan_markdown(cfg: Dict) -> str:
    years = year_list(cfg)
    return f"""# LitPop Execution Plan

## Scope
- Years: {years[0]}-{years[-1]}
- Grid: EPSG:4326, 30 arc-second, full global extent
- Parameters: m=1, n=1

## Inputs
- NTL: {resolve_path(cfg, "ntl_root")}
- Population: {resolve_path(cfg, "pop_root")}
- GDP: {resolve_path(cfg, "gdp_csv")}
- Country raster: {resolve_path(cfg, "country_id_raster")}
- Country lookup: {resolve_path(cfg, "country_lookup_csv")}

## Steps
1. inventory
2. save_plan
3. prepare_gdp
4. compute_denominator
5. allocate_litpop
6. quality_report
"""


def step_save_plan(cfg: Dict, logger: logging.Logger, overwrite: bool) -> Path:
    logger.info("Step 2/6: save_plan")
    plan_output = resolve_path(cfg, "plan_output")
    if plan_output.exists() and not overwrite:
        logger.info("Skip existing plan file (overwrite=false): %s", plan_output)
        return plan_output
    ensure_dir(plan_output.parent)
    plan_output.write_text(build_plan_markdown(cfg), encoding="utf-8")
    logger.info("Plan written: %s", plan_output)
    return plan_output


def step_prepare_gdp(cfg: Dict, logger: logging.Logger, overwrite: bool) -> Tuple[Path, Path, Path]:
    logger.info("Step 3/6: prepare_gdp")
    tbf = table_files(cfg)
    outputs = [tbf["gdp_long"], tbf["gdp_wide"], tbf["missing_gdp"]]
    if not overwrite and all_exist(outputs):
        logger.info("Skip existing GDP outputs (overwrite=false).")
        return tbf["gdp_long"], tbf["gdp_wide"], tbf["missing_gdp"]

    years = year_list(cfg)
    year_cols = [str(y) for y in years]
    lookup = pd.read_csv(resolve_path(cfg, "country_lookup_csv"))
    if "country_id" not in lookup.columns or "iso3_final" not in lookup.columns:
        raise ValueError("country_id_lookup.csv missing columns country_id/iso3_final.")
    country_df = (
        lookup[["country_id", "iso3_final"]]
        .copy()
        .dropna(subset=["country_id", "iso3_final"])
        .drop_duplicates()
        .sort_values("country_id")
    )
    country_df["country_id"] = country_df["country_id"].astype(int)
    country_df["iso3_final"] = country_df["iso3_final"].astype(str)

    wb = pd.read_csv(resolve_path(cfg, "gdp_csv"), skiprows=4, dtype=str, encoding="utf-8-sig")
    required = ["Country Code", "Indicator Code"] + year_cols
    missing_cols = [c for c in required if c not in wb.columns]
    if missing_cols:
        raise ValueError(f"GDP CSV missing required columns: {missing_cols}")
    wb = wb.loc[wb["Indicator Code"] == "NY.GDP.MKTP.CD", ["Country Code"] + year_cols].copy()
    wb = wb.rename(columns={"Country Code": "iso3_final"})
    wb["iso3_final"] = wb["iso3_final"].astype(str).str.strip()
    for col in year_cols:
        wb[col] = pd.to_numeric(wb[col], errors="coerce").astype("float64")
    wb_long = wb.melt(id_vars=["iso3_final"], value_vars=year_cols, var_name="year", value_name="gdp_raw")
    wb_long["year"] = wb_long["year"].astype(int)

    grid = country_df.assign(_k=1).merge(pd.DataFrame({"year": years, "_k": 1}), on="_k").drop(columns="_k")
    long_df = grid.merge(wb_long, on=["iso3_final", "year"], how="left")
    long_df["is_missing_gdp"] = long_df["gdp_raw"].isna().astype(int)
    long_df["gdp_current_usd"] = long_df["gdp_raw"].fillna(0.0).astype("float64")
    long_df = long_df[["country_id", "iso3_final", "year", "gdp_current_usd", "is_missing_gdp"]]
    long_df = long_df.sort_values(["country_id", "year"]).reset_index(drop=True)

    missing_df = long_df.loc[long_df["is_missing_gdp"] == 1, ["country_id", "iso3_final", "year"]].copy()
    write_df_csv(tbf["missing_gdp"], missing_df)

    wide = long_df.pivot(index=["country_id", "iso3_final"], columns="year", values="gdp_current_usd").reset_index()
    for y in years:
        if y not in wide.columns:
            wide[y] = 0.0
    wide = wide[["country_id", "iso3_final"] + years].sort_values("country_id").reset_index(drop=True)
    wide.columns = ["country_id", "iso3_final"] + year_cols

    write_df_csv(tbf["gdp_long"], long_df)
    write_df_csv(tbf["gdp_wide"], wide)
    logger.info("GDP long: %s (%d rows)", tbf["gdp_long"], len(long_df))
    logger.info("GDP wide: %s (%d rows)", tbf["gdp_wide"], len(wide))
    logger.info("Missing GDP rows: %s (%d rows)", tbf["missing_gdp"], len(missing_df))
    return tbf["gdp_long"], tbf["gdp_wide"], tbf["missing_gdp"]


def array_window_iter(height: int, width: int, block_rows: int, block_cols: int) -> Iterable[Window]:
    for row_off in range(0, height, block_rows):
        h = min(block_rows, height - row_off)
        for col_off in range(0, width, block_cols):
            w = min(block_cols, width - col_off)
            yield Window(col_off=col_off, row_off=row_off, width=w, height=h)


def sanitize_nonnegative(arr: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    out = arr.astype(np.float64, copy=False)
    if nodata is not None:
        out = np.where(out == nodata, 0.0, out)
    out = np.where(np.isfinite(out), out, 0.0)
    out = np.where(out < 0.0, 0.0, out)
    return out


def init_stats() -> Dict[str, float]:
    return {
        "min": float("inf"),
        "max": float("-inf"),
        "sum": 0.0,
        "sumsq": 0.0,
        "count": 0.0,
        "zero_count": 0.0,
        "finite_issue_count": 0.0,
    }


def update_stats(st: Dict[str, float], arr: np.ndarray) -> None:
    vals = arr.astype(np.float64, copy=False)
    finite = np.isfinite(vals)
    if not np.all(finite):
        st["finite_issue_count"] += float(vals.size - np.count_nonzero(finite))
    vals = vals[finite]
    if vals.size == 0:
        return
    st["min"] = min(st["min"], float(vals.min()))
    st["max"] = max(st["max"], float(vals.max()))
    st["sum"] += float(vals.sum())
    st["sumsq"] += float(np.square(vals).sum())
    st["count"] += float(vals.size)
    st["zero_count"] += float(np.count_nonzero(vals <= 0.0))


def finalize_stats(year: int, st: Dict[str, float]) -> Dict:
    count = int(st["count"])
    if count <= 0:
        return {
            "year": int(year),
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "sum": float("nan"),
            "zero_frac": float("nan"),
            "count": 0,
            "finite_issue_count": int(st["finite_issue_count"]),
        }
    mean_v = st["sum"] / st["count"]
    std_v = math.sqrt(max(st["sumsq"] / st["count"] - mean_v * mean_v, 0.0))
    return {
        "year": int(year),
        "min": float(st["min"]),
        "max": float(st["max"]),
        "mean": float(mean_v),
        "std": float(std_v),
        "sum": float(st["sum"]),
        "zero_frac": float(st["zero_count"] / st["count"]),
        "count": int(count),
        "finite_issue_count": int(st["finite_issue_count"]),
    }


def read_country_lookup(cfg: Dict) -> pd.DataFrame:
    lookup = pd.read_csv(resolve_path(cfg, "country_lookup_csv"))
    req = {"country_id", "iso3_final"}
    missing = [c for c in req if c not in lookup.columns]
    if missing:
        raise ValueError(f"country_id_lookup.csv missing columns: {missing}")
    lookup = lookup.copy()
    lookup["country_id"] = lookup["country_id"].astype(int)
    lookup["iso3_final"] = lookup["iso3_final"].astype(str)
    return lookup.sort_values("country_id").reset_index(drop=True)


def build_year_country_arrays(
    gdp_long: pd.DataFrame,
    year: int,
    max_country_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    gdp_arr = np.zeros(max_country_id + 1, dtype=np.float64)
    missing_arr = np.ones(max_country_id + 1, dtype=np.int8)
    sub = gdp_long.loc[gdp_long["year"] == int(year), ["country_id", "gdp_current_usd", "is_missing_gdp"]]
    for row in sub.itertuples(index=False):
        cid = int(row.country_id)
        if cid < 0 or cid > max_country_id:
            continue
        gdp_arr[cid] = float(row.gdp_current_usd)
        missing_arr[cid] = int(row.is_missing_gdp)
    return gdp_arr, missing_arr


def step_compute_denominator(cfg: Dict, logger: logging.Logger, overwrite: bool) -> Tuple[Path, Path]:
    logger.info("Step 4/6: compute_denominator")
    tbf = table_files(cfg)
    if not tbf["gdp_long"].exists():
        raise FileNotFoundError(f"Missing GDP long table: {tbf['gdp_long']}")
    outputs = [tbf["denominator"], tbf["zero_denominator"]]
    if not overwrite and all_exist(outputs):
        logger.info("Skip existing denominator outputs (overwrite=false).")
        return tbf["denominator"], tbf["zero_denominator"]

    years = year_list(cfg)
    ntl_files = ntl_file_map(cfg)
    pop_files = pop_file_map(cfg)
    country_raster_path = resolve_path(cfg, "country_id_raster")
    ntl_row_off = int(cfg["ntl_window"]["global_row_offset"])
    ntl_height = int(cfg["ntl_window"]["height"])
    runtime = cfg["runtime"]
    block_rows = int(runtime["block_rows"])
    block_cols = int(runtime["block_cols"])

    lookup = read_country_lookup(cfg)
    max_country_id = int(lookup["country_id"].max())
    gdp_long = pd.read_csv(tbf["gdp_long"])
    gdp_long["country_id"] = gdp_long["country_id"].astype(int)
    gdp_long["year"] = gdp_long["year"].astype(int)
    gdp_long["gdp_current_usd"] = pd.to_numeric(gdp_long["gdp_current_usd"], errors="coerce").fillna(0.0).astype(float)
    gdp_long["is_missing_gdp"] = gdp_long["is_missing_gdp"].astype(int)

    rows: List[Dict] = []
    zero_rows: List[Dict] = []

    with rasterio.open(country_raster_path) as country_ds:
        for year in years:
            denom = np.zeros(max_country_id + 1, dtype=np.float64)
            gdp_arr, missing_arr = build_year_country_arrays(gdp_long, year, max_country_id)
            with rasterio.open(ntl_files[year]) as ntl_ds, rasterio.open(pop_files[year]) as pop_ds:
                if ntl_ds.height != ntl_height:
                    raise RuntimeError(f"Unexpected NTL height for {year}: {ntl_ds.height}, expected={ntl_height}")
                for window in array_window_iter(ntl_ds.height, ntl_ds.width, block_rows=block_rows, block_cols=block_cols):
                    ntl_arr = sanitize_nonnegative(ntl_ds.read(1, window=window), ntl_ds.nodata)
                    global_window = Window(
                        col_off=window.col_off,
                        row_off=ntl_row_off + window.row_off,
                        width=window.width,
                        height=window.height,
                    )
                    pop_arr = sanitize_nonnegative(pop_ds.read(1, window=global_window), pop_ds.nodata)
                    cid_arr = country_ds.read(1, window=global_window).astype(np.int64, copy=False)
                    valid = cid_arr > 0
                    if not np.any(valid):
                        continue
                    weight = ntl_arr * pop_arr
                    weight = np.where(np.isfinite(weight), weight, 0.0)
                    denom += np.bincount(
                        cid_arr[valid].ravel(),
                        weights=weight[valid].ravel(),
                        minlength=max_country_id + 1,
                    )

            year_rows = 0
            year_zero = 0
            for r in lookup.itertuples(index=False):
                cid = int(r.country_id)
                gdp_v = float(gdp_arr[cid])
                is_missing = int(missing_arr[cid])
                den_v = float(denom[cid])
                flag = bool(den_v <= 0.0 and gdp_v > 0.0)
                row = {
                    "country_id": cid,
                    "iso3_final": str(r.iso3_final),
                    "year": int(year),
                    "denominator": den_v,
                    "gdp_input": gdp_v,
                    "is_missing_gdp": is_missing,
                    "gdp_positive_with_zero_denominator": flag,
                }
                rows.append(row)
                year_rows += 1
                if flag:
                    zero_rows.append(row)
                    year_zero += 1
            logger.info(
                "Denominator year %d: countries=%d, zero_denominator_with_positive_gdp=%d",
                year,
                year_rows,
                year_zero,
            )

    den_cols = [
        "country_id",
        "iso3_final",
        "year",
        "denominator",
        "gdp_input",
        "is_missing_gdp",
        "gdp_positive_with_zero_denominator",
    ]
    den_df = pd.DataFrame(rows, columns=den_cols).sort_values(["year", "country_id"]).reset_index(drop=True)
    zero_df = pd.DataFrame(zero_rows, columns=den_cols).sort_values(["year", "country_id"]).reset_index(drop=True)
    write_df_csv(tbf["denominator"], den_df)
    write_df_csv(tbf["zero_denominator"], zero_df)
    logger.info("Denominator table: %s (%d rows)", tbf["denominator"], len(den_df))
    logger.info("Zero-denominator table: %s (%d rows)", tbf["zero_denominator"], len(zero_df))
    return tbf["denominator"], tbf["zero_denominator"]


def output_profile(reference_ds: rasterio.io.DatasetReader, cfg: Dict) -> Dict:
    runtime = cfg["runtime"]
    g = grid_spec(cfg)
    profile = reference_ds.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype=g.dtype,
        nodata=g.nodata,
        compress=str(runtime["compression"]),
        tiled=bool(runtime["tiled"]),
        blockxsize=int(runtime["blockxsize"]),
        blockysize=int(runtime["blockysize"]),
        bigtiff=str(runtime["bigtiff"]),
    )
    return profile


def raster_stats(path: Path, nodata: Optional[float]) -> Dict[str, float]:
    st = init_stats()
    with rasterio.open(path) as ds:
        for _, window in ds.block_windows(1):
            arr = ds.read(1, window=window).astype(np.float64, copy=False)
            finite = np.isfinite(arr)
            if nodata is not None:
                finite &= arr != nodata
            vals = arr[finite]
            if vals.size == 0:
                continue
            st["min"] = min(st["min"], float(vals.min()))
            st["max"] = max(st["max"], float(vals.max()))
            st["sum"] += float(vals.sum())
            st["sumsq"] += float(np.square(vals).sum())
            st["count"] += float(vals.size)
            st["zero_count"] += float(np.count_nonzero(vals <= 0.0))
            st["finite_issue_count"] += float(arr.size - np.count_nonzero(np.isfinite(arr)))
    return finalize_stats(0, st)


def step_allocate_litpop(cfg: Dict, logger: logging.Logger, overwrite: bool) -> Tuple[Path, Path]:
    logger.info("Step 5/6: allocate_litpop")
    tbf = table_files(cfg)
    anf = annual_files(cfg)
    if not tbf["gdp_long"].exists():
        raise FileNotFoundError(f"Missing GDP long table: {tbf['gdp_long']}")
    if not tbf["denominator"].exists():
        raise FileNotFoundError(f"Missing denominator table: {tbf['denominator']}")

    years = year_list(cfg)
    out_dir = output_paths(cfg)["annual_dir"]
    ensure_dir(out_dir)
    output_files = {y: out_dir / f"litpop_{y}_30as.tif" for y in years}

    years_write: List[int] = []
    years_skip: List[int] = []
    for y in years:
        if output_files[y].exists() and not overwrite:
            years_skip.append(y)
            logger.info("Skip existing output (overwrite=false): %s", output_files[y].name)
        else:
            years_write.append(y)

    ntl_files = ntl_file_map(cfg)
    pop_files = pop_file_map(cfg)
    country_path = resolve_path(cfg, "country_id_raster")
    ntl_row_off = int(cfg["ntl_window"]["global_row_offset"])
    ntl_height = int(cfg["ntl_window"]["height"])
    runtime = cfg["runtime"]
    block_rows = int(runtime["block_rows"])
    block_cols = int(runtime["block_cols"])

    lookup = read_country_lookup(cfg)
    max_country_id = int(lookup["country_id"].max())
    gdp_long = pd.read_csv(tbf["gdp_long"])
    gdp_long["country_id"] = gdp_long["country_id"].astype(int)
    gdp_long["year"] = gdp_long["year"].astype(int)
    gdp_long["gdp_current_usd"] = pd.to_numeric(gdp_long["gdp_current_usd"], errors="coerce").fillna(0.0).astype(float)
    gdp_long["is_missing_gdp"] = gdp_long["is_missing_gdp"].astype(int)

    den_df = pd.read_csv(tbf["denominator"])
    den_df["country_id"] = den_df["country_id"].astype(int)
    den_df["year"] = den_df["year"].astype(int)
    den_df["denominator"] = pd.to_numeric(den_df["denominator"], errors="coerce").fillna(0.0).astype(float)

    generated_years: List[int] = []
    stats_by_year: Dict[int, Dict] = {}
    zero_den_flags: List[Dict] = []

    with rasterio.open(country_path) as country_ds:
        profile = output_profile(country_ds, cfg)
        for year in years_write:
            logger.info("Allocating LitPop year %d", year)
            gdp_arr, missing_arr = build_year_country_arrays(gdp_long, year, max_country_id)
            den_arr = np.zeros(max_country_id + 1, dtype=np.float64)
            den_sub = den_df.loc[den_df["year"] == year, ["country_id", "denominator"]]
            for r in den_sub.itertuples(index=False):
                cid = int(r.country_id)
                if 0 <= cid <= max_country_id:
                    den_arr[cid] = float(r.denominator)

            st = init_stats()
            out_path = output_files[year]
            with rasterio.open(ntl_files[year]) as ntl_ds, rasterio.open(pop_files[year]) as pop_ds, rasterio.open(
                out_path, "w", **profile
            ) as out_ds:
                for window in array_window_iter(country_ds.height, country_ds.width, block_rows=block_rows, block_cols=block_cols):
                    out_block = np.zeros((int(window.height), int(window.width)), dtype=np.float64)
                    overlap_top = max(int(window.row_off), ntl_row_off)
                    overlap_bottom = min(int(window.row_off + window.height), ntl_row_off + ntl_height)

                    if overlap_bottom > overlap_top:
                        local_top = overlap_top - int(window.row_off)
                        local_bottom = overlap_bottom - int(window.row_off)
                        overlap_h = overlap_bottom - overlap_top

                        ntl_window = Window(
                            col_off=window.col_off,
                            row_off=overlap_top - ntl_row_off,
                            width=window.width,
                            height=overlap_h,
                        )
                        global_overlap = Window(
                            col_off=window.col_off,
                            row_off=overlap_top,
                            width=window.width,
                            height=overlap_h,
                        )

                        ntl_arr = sanitize_nonnegative(ntl_ds.read(1, window=ntl_window), ntl_ds.nodata)
                        pop_arr = sanitize_nonnegative(pop_ds.read(1, window=global_overlap), pop_ds.nodata)
                        cid_arr = country_ds.read(1, window=global_overlap).astype(np.int64, copy=False)

                        valid = cid_arr > 0
                        if np.any(valid):
                            weight = ntl_arr * pop_arr
                            weight = np.where(np.isfinite(weight), weight, 0.0)
                            denom_vals = den_arr[cid_arr]
                            gdp_vals = gdp_arr[cid_arr]
                            missing_vals = missing_arr[cid_arr]
                            alloc = np.zeros_like(weight, dtype=np.float64)
                            can_alloc = valid & (missing_vals == 0) & (denom_vals > 0.0) & (gdp_vals > 0.0) & (weight > 0.0)
                            if np.any(can_alloc):
                                alloc[can_alloc] = gdp_vals[can_alloc] * (weight[can_alloc] / denom_vals[can_alloc])
                            out_block[local_top:local_bottom, :] = alloc

                    out_block = np.where(np.isfinite(out_block), out_block, 0.0)
                    out_block = np.where(out_block < 0.0, 0.0, out_block)
                    out_f32 = out_block.astype(np.float32, copy=False)
                    out_ds.write(out_f32, 1, window=window)
                    update_stats(st, out_f32)

            stats_by_year[year] = finalize_stats(year, st)
            generated_years.append(year)

            flag_sub = den_df.loc[(den_df["year"] == year) & (den_df["denominator"] <= 0.0), ["country_id"]]
            if len(flag_sub) > 0:
                for cid in flag_sub["country_id"].astype(int).tolist():
                    if 0 <= cid <= max_country_id and gdp_arr[cid] > 0.0:
                        iso = lookup.loc[lookup["country_id"] == cid, "iso3_final"]
                        iso3 = str(iso.iloc[0]) if len(iso) > 0 else ""
                        zero_den_flags.append({"year": year, "country_id": cid, "iso3_final": iso3, "gdp_input": gdp_arr[cid]})

    for year in years_skip:
        st = raster_stats(output_files[year], nodata=float(cfg["grid"]["nodata"]))
        st["year"] = year
        stats_by_year[year] = st

    summary_rows = [stats_by_year[y] for y in years]
    write_df_csv(anf["summary"], pd.DataFrame(summary_rows))
    manifest = {
        "created_at": datetime.now().isoformat(),
        "config_path": cfg.get("_config_path", ""),
        "algorithm": "A_i = GDP_c * (L_i^m * P_i^n) / sum_j_in_c(L_j^m * P_j^n), m=1, n=1",
        "years": years,
        "generated_years": sorted(generated_years),
        "skipped_existing_years": sorted(years_skip),
        "overwrite": bool(overwrite),
        "ntl_window": cfg["ntl_window"],
        "output_files": {str(y): str(output_files[y]) for y in years},
        "summary_csv": str(anf["summary"]),
        "zero_denominator_positive_gdp_case_count": len(zero_den_flags),
    }
    write_json(anf["manifest"], manifest)

    logger.info("LitPop summary: %s", anf["summary"])
    logger.info("LitPop manifest: %s", anf["manifest"])
    if zero_den_flags:
        logger.warning(
            "Found %d country-year cases with positive GDP but zero denominator; outputs written as 0 for those cases.",
            len(zero_den_flags),
        )
    return anf["manifest"], anf["summary"]


def compute_country_sums_from_outputs(cfg: Dict, years: List[int], output_files: Dict[int, Path]) -> pd.DataFrame:
    lookup = read_country_lookup(cfg)
    max_country_id = int(lookup["country_id"].max())
    country_path = resolve_path(cfg, "country_id_raster")
    runtime = cfg["runtime"]
    block_rows = int(runtime["block_rows"])
    block_cols = int(runtime["block_cols"])

    rows: List[Dict] = []
    with rasterio.open(country_path) as country_ds:
        for year in years:
            sums = np.zeros(max_country_id + 1, dtype=np.float64)
            with rasterio.open(output_files[year]) as out_ds:
                for window in array_window_iter(country_ds.height, country_ds.width, block_rows=block_rows, block_cols=block_cols):
                    arr = out_ds.read(1, window=window).astype(np.float64, copy=False)
                    finite = np.isfinite(arr)
                    if out_ds.nodata is not None:
                        finite &= arr != out_ds.nodata
                    vals = np.where(finite, arr, 0.0)
                    vals = np.where(vals < 0.0, 0.0, vals)
                    cid = country_ds.read(1, window=window).astype(np.int64, copy=False)
                    valid = cid > 0
                    if not np.any(valid):
                        continue
                    sums += np.bincount(cid[valid].ravel(), weights=vals[valid].ravel(), minlength=max_country_id + 1)
            for r in lookup.itertuples(index=False):
                cid = int(r.country_id)
                rows.append(
                    {
                        "country_id": cid,
                        "iso3_final": str(r.iso3_final),
                        "year": int(year),
                        "litpop_sum": float(sums[cid]),
                    }
                )
    return pd.DataFrame(rows)


def step_quality_report(cfg: Dict, logger: logging.Logger, overwrite: bool) -> Tuple[Path, Path, Path, Path]:
    logger.info("Step 6/6: quality_report")
    rf = report_files(cfg)
    if not overwrite and all_exist([rf["country_conservation"], rf["qc_metrics"], rf["qa_summary"], rf["report_md"]]):
        logger.info("Skip existing report outputs (overwrite=false).")
        return rf["country_conservation"], rf["qc_metrics"], rf["qa_summary"], rf["report_md"]

    tbf = table_files(cfg)
    anf = annual_files(cfg)
    required = [tbf["gdp_long"], tbf["denominator"], anf["summary"], anf["manifest"]]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing prerequisite files for quality_report: {missing}")

    years = year_list(cfg)
    out_dir = output_paths(cfg)["annual_dir"]
    output_files = {y: out_dir / f"litpop_{y}_30as.tif" for y in years}
    missing_out = [output_files[y] for y in years if not output_files[y].exists()]
    if missing_out:
        raise FileNotFoundError(f"Missing LitPop outputs: {missing_out}")

    gdp_long = pd.read_csv(tbf["gdp_long"])
    gdp_long["country_id"] = gdp_long["country_id"].astype(int)
    gdp_long["year"] = gdp_long["year"].astype(int)
    gdp_long["gdp_current_usd"] = pd.to_numeric(gdp_long["gdp_current_usd"], errors="coerce").fillna(0.0).astype(float)
    gdp_long["is_missing_gdp"] = gdp_long["is_missing_gdp"].astype(int)

    den = pd.read_csv(tbf["denominator"])
    den["country_id"] = den["country_id"].astype(int)
    den["year"] = den["year"].astype(int)
    den["denominator"] = pd.to_numeric(den["denominator"], errors="coerce").fillna(0.0).astype(float)

    litpop_sum = compute_country_sums_from_outputs(cfg, years, output_files)
    cons = gdp_long.merge(den[["country_id", "year", "denominator"]], on=["country_id", "year"], how="left")
    cons = cons.merge(litpop_sum, on=["country_id", "iso3_final", "year"], how="left")
    cons["denominator"] = cons["denominator"].fillna(0.0)
    cons["litpop_sum"] = cons["litpop_sum"].fillna(0.0)
    cons["abs_error"] = np.abs(cons["litpop_sum"] - cons["gdp_current_usd"])
    valid = (cons["gdp_current_usd"] > 0.0) & (cons["denominator"] > 0.0) & (cons["is_missing_gdp"] == 0)
    cons["rel_error"] = np.where(valid, cons["abs_error"] / np.maximum(cons["gdp_current_usd"], 1e-12), np.nan)
    cons["is_valid_conservation"] = valid.astype(int)
    cons_out = cons[
        [
            "country_id",
            "iso3_final",
            "year",
            "gdp_current_usd",
            "litpop_sum",
            "denominator",
            "is_missing_gdp",
            "is_valid_conservation",
            "abs_error",
            "rel_error",
        ]
    ].sort_values(["year", "country_id"])
    write_df_csv(rf["country_conservation"], cons_out)

    summary = pd.read_csv(anf["summary"])
    summary["year"] = summary["year"].astype(int)
    qc_rows: List[Dict] = []
    for year in years:
        sub = cons_out.loc[cons_out["year"] == year]
        valid_sub = sub.loc[sub["is_valid_conservation"] == 1]
        gdp_total = float(sub["gdp_current_usd"].sum())
        lit_total = float(sub["litpop_sum"].sum())
        total_abs_error = abs(lit_total - gdp_total)
        total_rel_error = total_abs_error / max(gdp_total, 1e-12)
        valid_gdp = float(valid_sub["gdp_current_usd"].sum())
        valid_abs = float(valid_sub["abs_error"].sum())
        valid_rel = valid_abs / max(valid_gdp, 1e-12) if valid_gdp > 0 else float("nan")
        ysum = summary.loc[summary["year"] == year].iloc[0].to_dict()
        qc_rows.append(
            {
                "year": int(year),
                "gdp_total_input": gdp_total,
                "litpop_total_sum": lit_total,
                "total_abs_error": total_abs_error,
                "total_rel_error": total_rel_error,
                "valid_country_count": int(len(valid_sub)),
                "missing_gdp_country_count": int((sub["is_missing_gdp"] == 1).sum()),
                "zero_denominator_country_count": int((sub["denominator"] <= 0.0).sum()),
                "valid_abs_error_sum": valid_abs,
                "valid_rel_error_sum_over_gdp": valid_rel,
                "raster_min": float(ysum["min"]),
                "raster_max": float(ysum["max"]),
                "raster_mean": float(ysum["mean"]),
                "raster_std": float(ysum["std"]),
                "raster_sum": float(ysum["sum"]),
                "raster_zero_frac": float(ysum["zero_frac"]),
                "raster_count": int(ysum["count"]),
                "raster_finite_issue_count": int(ysum["finite_issue_count"]),
            }
        )
    qc_df = pd.DataFrame(qc_rows).sort_values("year").reset_index(drop=True)
    write_df_csv(rf["qc_metrics"], qc_df)

    zero_den = pd.read_csv(tbf["zero_denominator"]) if tbf["zero_denominator"].exists() else pd.DataFrame()
    qa_summary = {
        "created_at": datetime.now().isoformat(),
        "config_path": cfg.get("_config_path", ""),
        "years": [int(years[0]), int(years[-1])],
        "qc": {
            "mean_total_rel_error": float(qc_df["total_rel_error"].mean()),
            "max_total_rel_error": float(qc_df["total_rel_error"].max()),
            "mean_valid_rel_error": float(qc_df["valid_rel_error_sum_over_gdp"].dropna().mean()),
        },
        "missing_gdp_country_year_count": int((gdp_long["is_missing_gdp"] == 1).sum()),
        "zero_denominator_positive_gdp_count": int(len(zero_den)),
        "output_root": str(resolve_path(cfg, "output_root")),
    }
    write_json(rf["qa_summary"], qa_summary)

    report_lines = [
        "# LitPop 2000-2024 Report",
        "",
        f"- created_at: {qa_summary['created_at']}",
        f"- config: {qa_summary['config_path']}",
        f"- output_root: {qa_summary['output_root']}",
        f"- years: {years[0]}-{years[-1]}",
        "",
        "## Core Settings",
        "- global full extent output (43200x21600, EPSG:4326, 30 arc-second)",
        "- NTL coverage outside area set to 0",
        "- GDP missing values set to 0 with trace records",
        "- LitPop exponents fixed to m=1, n=1",
        "",
        "## QC Summary",
        f"- mean total relative error: {qa_summary['qc']['mean_total_rel_error']}",
        f"- max total relative error: {qa_summary['qc']['max_total_rel_error']}",
        f"- mean valid-country relative error: {qa_summary['qc']['mean_valid_rel_error']}",
        f"- missing GDP country-year count: {qa_summary['missing_gdp_country_year_count']}",
        f"- zero denominator with positive GDP count: {qa_summary['zero_denominator_positive_gdp_count']}",
        "",
        "## Output Files",
        f"- country_conservation: `{rf['country_conservation']}`",
        f"- qc_metrics: `{rf['qc_metrics']}`",
        f"- qa_summary: `{rf['qa_summary']}`",
        f"- annual_summary: `{anf['summary']}`",
        f"- manifest: `{anf['manifest']}`",
    ]
    ensure_dir(rf["report_md"].parent)
    rf["report_md"].write_text("\n".join(report_lines), encoding="utf-8")

    logger.info("Country conservation: %s", rf["country_conservation"])
    logger.info("QC metrics: %s", rf["qc_metrics"])
    logger.info("QA summary: %s", rf["qa_summary"])
    logger.info("Report markdown: %s", rf["report_md"])
    return rf["country_conservation"], rf["qc_metrics"], rf["qa_summary"], rf["report_md"]


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = project_root() / cfg_path

    cfg = load_config(cfg_path)
    cfg["runtime"]["overwrite"] = bool(args.overwrite or cfg["runtime"].get("overwrite", False))
    overwrite = bool(cfg["runtime"]["overwrite"])

    logger = setup_logger(inventory_files(cfg)["run_log"])
    logger.info("=" * 72)
    logger.info("LitPop pipeline started")
    logger.info("Config: %s", cfg_path)
    logger.info("Overwrite: %s", overwrite)
    logger.info("=" * 72)

    steps = parse_steps(args.steps)
    logger.info("Steps: %s", steps)

    for step in steps:
        if step == "inventory":
            step_inventory(cfg, logger, overwrite=overwrite)
        elif step == "save_plan":
            step_save_plan(cfg, logger, overwrite=overwrite)
        elif step == "prepare_gdp":
            step_prepare_gdp(cfg, logger, overwrite=overwrite)
        elif step == "compute_denominator":
            step_compute_denominator(cfg, logger, overwrite=overwrite)
        elif step == "allocate_litpop":
            step_allocate_litpop(cfg, logger, overwrite=overwrite)
        elif step == "quality_report":
            step_quality_report(cfg, logger, overwrite=overwrite)

    logger.info("=" * 72)
    logger.info("LitPop pipeline completed")
    logger.info("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
