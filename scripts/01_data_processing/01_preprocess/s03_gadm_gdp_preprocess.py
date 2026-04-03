#!/usr/bin/env python3
"""GADM + World Bank GDP preprocessing for LitPop (v1)."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
import yaml


STEP_ORDER = [
    "inventory",
    "build_mapping",
    "prepare_gdp",
    "join_and_validate",
    "export_vector",
    "rasterize_country_id",
    "report",
]

ZXX_PARENT_DEFAULT = {
    "Z01": "IND",
    "Z02": "CHN",
    "Z03": "CHN",
    "Z04": "IND",
    "Z05": "IND",
    "Z06": "PAK",
    "Z07": "IND",
    "Z08": "CHN",
    "Z09": "IND",
}


@dataclass
class RasterGridSpec:
    crs: str
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    width: int
    height: int
    resolution_deg: float
    nodata: int
    dtype: str
    compression: str
    tiled: bool
    bigtiff: str
    blockxsize: int
    blockysize: int


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GADM + WorldBank GDP preprocessing for LitPop")
    parser.add_argument(
        "--config",
        type=str,
        default="reproduction_v6/config/gadm_gdp_preprocess.yaml",
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
    logger = logging.getLogger("gadm_gdp_preprocess")
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
    p = Path(cfg["paths"][key])
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


def build_anomaly(level: str, scope: str, item: str, issue: str, detail: str) -> Dict:
    return {"level": level, "scope": scope, "item": item, "issue": issue, "detail": detail}


def year_list(cfg: Dict) -> List[int]:
    y0 = int(cfg["years"]["start"])
    y1 = int(cfg["years"]["end"])
    return list(range(y0, y1 + 1))


def mapping_rules(cfg: Dict) -> Dict[str, str]:
    custom = cfg.get("mapping", {}).get("disputed_to_parent", {})
    rules = dict(ZXX_PARENT_DEFAULT)
    for k, v in custom.items():
        rules[str(k)] = str(v)
    return rules


def grid_spec(cfg: Dict) -> RasterGridSpec:
    rcfg = cfg["runtime"]["raster"]
    return RasterGridSpec(
        crs=str(rcfg["crs"]),
        lon_min=float(rcfg["lon_min"]),
        lon_max=float(rcfg["lon_max"]),
        lat_min=float(rcfg["lat_min"]),
        lat_max=float(rcfg["lat_max"]),
        width=int(rcfg["width"]),
        height=int(rcfg["height"]),
        resolution_deg=float(rcfg["resolution_deg"]),
        nodata=int(rcfg["nodata"]),
        dtype=str(rcfg["dtype"]),
        compression=str(rcfg["compression"]),
        tiled=bool(rcfg["tiled"]),
        bigtiff=str(rcfg["bigtiff"]),
        blockxsize=int(rcfg["blockxsize"]),
        blockysize=int(rcfg["blockysize"]),
    )


def output_paths(cfg: Dict) -> Dict[str, Path]:
    root = resolve_path(cfg, "output_root")
    return {
        "root": root,
        "inventory_dir": root / "00_inventory",
        "mapping_dir": root / "01_mapping",
        "gdp_dir": root / "02_gdp",
        "join_dir": root / "03_join",
        "vector_dir": root / "04_vector",
        "raster_dir": root / "05_raster",
        "report_dir": root / "06_report",
    }


def inventory_files(cfg: Dict) -> Dict[str, Path]:
    outs = output_paths(cfg)
    return {
        "source_inventory": outs["inventory_dir"] / "source_inventory.json",
        "input_anomalies": outs["inventory_dir"] / "input_anomalies.csv",
    }


def mapping_files(cfg: Dict) -> Dict[str, Path]:
    outs = output_paths(cfg)
    return {
        "raw": outs["mapping_dir"] / "gadm_adm0_iso_mapping_raw.csv",
        "clean": outs["mapping_dir"] / "gadm_adm0_iso_mapping_clean.csv",
        "zxx_log": outs["mapping_dir"] / "zxx_reassignment_log.csv",
    }


def gdp_files(cfg: Dict) -> Dict[str, Path]:
    outs = output_paths(cfg)
    return {
        "long": outs["gdp_dir"] / "gdp_worldbank_2000_2024_long.csv",
        "wide": outs["gdp_dir"] / "gdp_worldbank_2000_2024_wide.csv",
    }


def join_files(cfg: Dict) -> Dict[str, Path]:
    outs = output_paths(cfg)
    return {
        "joined_long": outs["join_dir"] / "gadm_gdp_joined_2000_2024_long.csv",
        "country_wide": outs["join_dir"] / "gadm_gdp_country_2000_2024_wide.csv",
        "unmatched_gdp": outs["join_dir"] / "unmatched_gdp_codes.csv",
        "unmatched_gadm": outs["join_dir"] / "unmatched_gadm_codes.csv",
        "coverage": outs["join_dir"] / "join_coverage_by_year.csv",
    }


def vector_files(cfg: Dict) -> Dict[str, Path]:
    outs = output_paths(cfg)
    return {
        "adm0_clean": outs["vector_dir"] / "gadm_adm0_clean.gpkg",
        "adm0_iso3": outs["vector_dir"] / "gadm_adm0_dissolved_iso3.gpkg",
    }


def raster_files(cfg: Dict) -> Dict[str, Path]:
    outs = output_paths(cfg)
    return {
        "country_id_raster": outs["raster_dir"] / "country_id_30as_global.tif",
        "country_lookup": outs["raster_dir"] / "country_id_lookup.csv",
    }


def report_files(cfg: Dict) -> Dict[str, Path]:
    outs = output_paths(cfg)
    return {
        "report_md": outs["report_dir"] / "gadm_gdp_preprocess_report.md",
        "qa_summary": outs["report_dir"] / "qa_summary.json",
    }


def all_exist(paths: Iterable[Path]) -> bool:
    return all(p.exists() for p in paths)


def safe_unlink(path: Path) -> None:
    if path.exists():
        path.unlink()


def load_worldbank_header(gdp_csv: Path) -> Tuple[int, List[str]]:
    with gdp_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if len(row) >= 2 and row[0] == "Country Name" and row[1] == "Country Code":
                return idx, row
    raise ValueError(f"Cannot find header row in GDP CSV: {gdp_csv}")


def read_adm0(cfg: Dict) -> gpd.GeoDataFrame:
    return gpd.read_file(resolve_path(cfg, "gadm_gpkg"), layer="ADM_0")


def collect_inventory(cfg: Dict, logger: logging.Logger) -> Tuple[Path, Path, List[Dict], Dict]:
    inv = inventory_files(cfg)
    gadm_path = resolve_path(cfg, "gadm_gpkg")
    gdp_path = resolve_path(cfg, "gdp_csv")
    anomalies: List[Dict] = []

    if not gadm_path.exists():
        anomalies.append(build_anomaly("error", "input", "gadm_gpkg", "missing_file", str(gadm_path)))
    if not gdp_path.exists():
        anomalies.append(build_anomaly("error", "input", "gdp_csv", "missing_file", str(gdp_path)))

    layer_names: List[str] = []
    adm0_stats: Dict[str, object] = {}
    if gadm_path.exists():
        try:
            layer_info = pyogrio.list_layers(gadm_path)
            layer_names = [str(row[0]) for row in layer_info]
            if "ADM_0" not in layer_names:
                anomalies.append(build_anomaly("error", "gadm", "ADM_0", "missing_layer", f"layers={layer_names}"))
            else:
                gdf = read_adm0(cfg)
                missing_cols = [c for c in ["GID_0", "COUNTRY", "geometry"] if c not in gdf.columns]
                if missing_cols:
                    anomalies.append(build_anomaly("error", "gadm", "ADM_0", "missing_columns", ",".join(missing_cols)))
                adm0_stats = {
                    "feature_count": int(len(gdf)),
                    "unique_country_count": int(gdf["COUNTRY"].nunique()),
                    "unique_gid0_count": int(gdf["GID_0"].nunique()),
                    "crs": None if gdf.crs is None else str(gdf.crs),
                    "bounds": [float(x) for x in gdf.total_bounds],
                }
                if adm0_stats["crs"] != "EPSG:4326":
                    anomalies.append(
                        build_anomaly(
                            "warning",
                            "gadm",
                            "ADM_0",
                            "unexpected_crs",
                            f"expected=EPSG:4326 actual={adm0_stats['crs']}",
                        )
                    )
        except Exception as exc:
            anomalies.append(build_anomaly("error", "gadm", "layer_scan", "unreadable", str(exc)))

    header_row_idx = None
    gdp_header: List[str] = []
    years = year_list(cfg)
    if gdp_path.exists():
        try:
            header_row_idx, gdp_header = load_worldbank_header(gdp_path)
            required = ["Country Name", "Country Code", "Indicator Code"] + [str(y) for y in years]
            missing = [c for c in required if c not in gdp_header]
            if missing:
                anomalies.append(
                    build_anomaly(
                        "error",
                        "gdp",
                        "header",
                        "missing_required_columns",
                        ",".join(missing),
                    )
                )
        except Exception as exc:
            anomalies.append(build_anomaly("error", "gdp", "header", "unreadable", str(exc)))

    payload = {
        "created_at": datetime.now().isoformat(),
        "config_path": cfg.get("_config_path", ""),
        "inputs": {
            "gadm_gpkg": str(gadm_path),
            "gdp_csv": str(gdp_path),
            "analysis_doc": str(resolve_path(cfg, "analysis_doc")),
            "litpop_doc": str(resolve_path(cfg, "litpop_doc")),
        },
        "layers": layer_names,
        "adm0_stats": adm0_stats,
        "gdp_header_row_index_0based": header_row_idx,
        "gdp_header_columns_count": len(gdp_header),
        "years": years,
        "anomaly_count": len(anomalies),
        "error_count": sum(1 for a in anomalies if a["level"] == "error"),
    }

    write_json(inv["source_inventory"], payload)
    write_rows_csv(inv["input_anomalies"], anomalies, headers=["level", "scope", "item", "issue", "detail"])
    logger.info("Inventory written: %s", inv["source_inventory"])
    logger.info("Anomalies written: %s (%d rows)", inv["input_anomalies"], len(anomalies))
    return inv["source_inventory"], inv["input_anomalies"], anomalies, payload


def step_inventory(cfg: Dict, logger: logging.Logger, overwrite: bool) -> Tuple[Path, Path]:
    logger.info("Step 1/7: inventory")
    inv_paths = inventory_files(cfg)
    if not overwrite and all_exist(inv_paths.values()):
        logger.info("Skip existing inventory outputs (overwrite=false).")
        return inv_paths["source_inventory"], inv_paths["input_anomalies"]

    _, _, anomalies, _ = collect_inventory(cfg, logger)
    err_count = sum(1 for a in anomalies if a["level"] == "error")
    if err_count > 0:
        raise RuntimeError(f"Inventory failed with {err_count} error(s).")
    return inv_paths["source_inventory"], inv_paths["input_anomalies"]


def step_build_mapping(cfg: Dict, logger: logging.Logger, overwrite: bool) -> Tuple[Path, Path, Path]:
    logger.info("Step 2/7: build_mapping")
    paths = mapping_files(cfg)
    if not overwrite and all_exist(paths.values()):
        logger.info("Skip existing mapping outputs (overwrite=false).")
        return paths["raw"], paths["clean"], paths["zxx_log"]

    gdf = read_adm0(cfg).reset_index(drop=True)
    gdf["feature_id"] = np.arange(1, len(gdf) + 1, dtype=np.int32)
    if "GID_0" not in gdf.columns or "COUNTRY" not in gdf.columns:
        raise ValueError("ADM_0 missing required columns GID_0/COUNTRY.")

    rules = mapping_rules(cfg)
    raw_df = pd.DataFrame(
        {
            "feature_id": gdf["feature_id"].astype(int),
            "gid0_raw": gdf["GID_0"].astype(str),
            "country": gdf["COUNTRY"].astype(str),
            "iso3_raw": gdf["GID_0"].astype(str),
            "is_disputed_raw": gdf["GID_0"].astype(str).isin(rules).astype(int),
        }
    )
    clean_df = raw_df.copy()
    clean_df["iso3_final"] = clean_df["gid0_raw"].map(rules).fillna(clean_df["gid0_raw"])
    clean_df["is_disputed"] = clean_df["gid0_raw"].isin(rules).astype(int)
    clean_df["dispute_parent"] = clean_df["gid0_raw"].map(rules).fillna("")
    clean_df = clean_df[
        ["feature_id", "gid0_raw", "country", "iso3_final", "is_disputed", "dispute_parent"]
    ].sort_values(["iso3_final", "feature_id"])

    zxx_log = clean_df.loc[clean_df["is_disputed"] == 1, ["gid0_raw", "country", "iso3_final"]].copy()
    zxx_log["rule"] = zxx_log["gid0_raw"] + "->" + zxx_log["iso3_final"]
    zxx_log = zxx_log.sort_values(["iso3_final", "gid0_raw"])

    write_df_csv(paths["raw"], raw_df.sort_values(["iso3_raw", "feature_id"]))
    write_df_csv(paths["clean"], clean_df)
    write_df_csv(paths["zxx_log"], zxx_log)

    logger.info("Raw mapping: %s", paths["raw"])
    logger.info("Clean mapping: %s", paths["clean"])
    logger.info("Zxx reassignment log: %s (%d rows)", paths["zxx_log"], len(zxx_log))
    return paths["raw"], paths["clean"], paths["zxx_log"]


def step_prepare_gdp(cfg: Dict, logger: logging.Logger, overwrite: bool) -> Tuple[Path, Path]:
    logger.info("Step 3/7: prepare_gdp")
    paths = gdp_files(cfg)
    if not overwrite and all_exist(paths.values()):
        logger.info("Skip existing GDP outputs (overwrite=false).")
        return paths["long"], paths["wide"]

    gdp_csv = resolve_path(cfg, "gdp_csv")
    years = year_list(cfg)
    year_cols = [str(y) for y in years]

    wb = pd.read_csv(gdp_csv, skiprows=4, dtype=str, encoding="utf-8-sig")
    required = ["Country Name", "Country Code", "Indicator Code"] + year_cols
    missing = [c for c in required if c not in wb.columns]
    if missing:
        raise ValueError(f"GDP file missing required columns: {missing}")

    wb = wb.loc[wb["Indicator Code"] == "NY.GDP.MKTP.CD", required].copy()
    wb = wb.rename(columns={"Country Name": "country_name_wb", "Country Code": "iso3"})
    wb["iso3"] = wb["iso3"].astype(str).str.strip()
    wb["country_name_wb"] = wb["country_name_wb"].astype(str).str.strip()

    for col in year_cols:
        wb[col] = pd.to_numeric(wb[col], errors="coerce").astype("float64")

    wide_df = wb[["iso3", "country_name_wb"] + year_cols].sort_values("iso3").reset_index(drop=True)
    long_df = wide_df.melt(
        id_vars=["iso3", "country_name_wb"],
        value_vars=year_cols,
        var_name="year",
        value_name="gdp_current_usd",
    )
    long_df["year"] = long_df["year"].astype(int)
    long_df["gdp_current_usd"] = long_df["gdp_current_usd"].astype("float64")
    long_df = long_df.sort_values(["iso3", "year"]).reset_index(drop=True)

    write_df_csv(paths["long"], long_df)
    write_df_csv(paths["wide"], wide_df)

    logger.info("GDP long: %s (%d rows)", paths["long"], len(long_df))
    logger.info("GDP wide: %s (%d rows)", paths["wide"], len(wide_df))
    return paths["long"], paths["wide"]


def step_join_and_validate(cfg: Dict, logger: logging.Logger, overwrite: bool) -> Tuple[Path, Path, Path, Path, Path]:
    logger.info("Step 4/7: join_and_validate")
    paths = join_files(cfg)
    if not overwrite and all_exist(paths.values()):
        logger.info("Skip existing join outputs (overwrite=false).")
        return (
            paths["joined_long"],
            paths["country_wide"],
            paths["unmatched_gdp"],
            paths["unmatched_gadm"],
            paths["coverage"],
        )

    mfiles = mapping_files(cfg)
    gfiles = gdp_files(cfg)
    if not mfiles["clean"].exists():
        raise FileNotFoundError(f"Missing mapping file: {mfiles['clean']}")
    if not gfiles["long"].exists():
        raise FileNotFoundError(f"Missing GDP long file: {gfiles['long']}")
    if not gfiles["wide"].exists():
        raise FileNotFoundError(f"Missing GDP wide file: {gfiles['wide']}")

    years = year_list(cfg)
    year_cols = [str(y) for y in years]
    mapping_df = pd.read_csv(mfiles["clean"], dtype={"gid0_raw": str, "country": str, "iso3_final": str})
    gdp_long = pd.read_csv(gfiles["long"], dtype={"iso3": str, "country_name_wb": str})
    gdp_long["year"] = pd.to_numeric(gdp_long["year"], errors="coerce").astype("Int64")
    gdp_long = gdp_long.dropna(subset=["year"]).copy()
    gdp_long["year"] = gdp_long["year"].astype(int)
    gdp_long["gdp_current_usd"] = pd.to_numeric(gdp_long["gdp_current_usd"], errors="coerce").astype("float64")
    gdp_long = gdp_long[gdp_long["year"].isin(years)].copy()

    mapping_base = mapping_df[
        ["feature_id", "gid0_raw", "country", "iso3_final", "is_disputed", "dispute_parent"]
    ].copy()
    mapping_base["__k"] = 1
    years_df = pd.DataFrame({"year": years}, dtype=int)
    years_df["__k"] = 1
    joined_base = mapping_base.merge(years_df, on="__k", how="inner").drop(columns="__k")

    gdp_join = gdp_long.rename(columns={"iso3": "iso3_final"})
    joined_long = joined_base.merge(
        gdp_join[["iso3_final", "country_name_wb", "year", "gdp_current_usd"]],
        on=["iso3_final", "year"],
        how="left",
    )
    joined_long = joined_long.sort_values(["iso3_final", "feature_id", "year"]).reset_index(drop=True)
    write_df_csv(paths["joined_long"], joined_long)

    iso3_codes = sorted(mapping_base["iso3_final"].dropna().astype(str).unique().tolist())
    country_year = pd.MultiIndex.from_product([iso3_codes, years], names=["iso3_final", "year"]).to_frame(index=False)
    gdp_country = gdp_join[["iso3_final", "country_name_wb", "year", "gdp_current_usd"]].copy()
    gdp_country = gdp_country.sort_values(["iso3_final", "year"]).drop_duplicates(["iso3_final", "year"], keep="first")
    country_year = country_year.merge(gdp_country, on=["iso3_final", "year"], how="left")

    country_name = (
        gdp_country.dropna(subset=["country_name_wb"])
        .sort_values(["iso3_final", "year"])
        .drop_duplicates(["iso3_final"], keep="first")[["iso3_final", "country_name_wb"]]
    )
    country_wide = country_year.pivot(index="iso3_final", columns="year", values="gdp_current_usd").reset_index()
    for y in years:
        if y not in country_wide.columns:
            country_wide[y] = np.nan
    country_wide = country_wide[["iso3_final"] + years]
    country_wide = country_wide.merge(country_name, on="iso3_final", how="left")
    country_wide = country_wide[["iso3_final", "country_name_wb"] + years]
    country_wide.columns = ["iso3_final", "country_name_wb"] + year_cols
    country_wide = country_wide.sort_values("iso3_final").reset_index(drop=True)
    write_df_csv(paths["country_wide"], country_wide)

    gdp_wide = pd.read_csv(gfiles["wide"], dtype={"iso3": str, "country_name_wb": str})
    gdp_codes = sorted(set(gdp_wide["iso3"].dropna().astype(str)))
    gadm_codes = iso3_codes

    unmatched_gdp = gdp_wide[gdp_wide["iso3"].isin(sorted(set(gdp_codes) - set(gadm_codes)))][
        ["iso3", "country_name_wb"]
    ].drop_duplicates()
    unmatched_gdp = unmatched_gdp.sort_values("iso3").reset_index(drop=True)
    write_df_csv(paths["unmatched_gdp"], unmatched_gdp)

    gadm_name_map = mapping_base.groupby("iso3_final", as_index=False)["country"].first()
    unmatched_gadm = gadm_name_map[gadm_name_map["iso3_final"].isin(sorted(set(gadm_codes) - set(gdp_codes)))]
    unmatched_gadm = unmatched_gadm.rename(columns={"country": "country_gadm"}).sort_values("iso3_final").reset_index(drop=True)
    write_df_csv(paths["unmatched_gadm"], unmatched_gadm)

    coverage_rows: List[Dict] = []
    total_countries = len(iso3_codes)
    for y in years:
        sub = country_year[country_year["year"] == y]
        matched = int(sub["gdp_current_usd"].notna().sum())
        missing = int(total_countries - matched)
        coverage_rows.append(
            {
                "year": int(y),
                "total_countries": int(total_countries),
                "matched_countries": int(matched),
                "missing_countries": int(missing),
                "match_rate": float(matched / total_countries) if total_countries > 0 else float("nan"),
                "missing_rate": float(missing / total_countries) if total_countries > 0 else float("nan"),
            }
        )
    coverage_df = pd.DataFrame(coverage_rows)
    write_df_csv(paths["coverage"], coverage_df)

    logger.info("Joined long: %s (%d rows)", paths["joined_long"], len(joined_long))
    logger.info("Country wide: %s (%d rows)", paths["country_wide"], len(country_wide))
    logger.info("Unmatched GDP codes: %s (%d rows)", paths["unmatched_gdp"], len(unmatched_gdp))
    logger.info("Unmatched GADM codes: %s (%d rows)", paths["unmatched_gadm"], len(unmatched_gadm))
    logger.info("Join coverage: %s", paths["coverage"])
    return (
        paths["joined_long"],
        paths["country_wide"],
        paths["unmatched_gdp"],
        paths["unmatched_gadm"],
        paths["coverage"],
    )


def _country_name_by_iso3(adm0_clean: gpd.GeoDataFrame) -> pd.DataFrame:
    preferred = (
        adm0_clean.loc[adm0_clean["is_disputed"] == 0, ["iso3_final", "COUNTRY"]]
        .drop_duplicates("iso3_final", keep="first")
        .rename(columns={"COUNTRY": "country_name"})
    )
    fallback = (
        adm0_clean[["iso3_final", "COUNTRY"]]
        .drop_duplicates("iso3_final", keep="first")
        .rename(columns={"COUNTRY": "country_name"})
    )
    out = fallback.set_index("iso3_final")
    out.update(preferred.set_index("iso3_final"))
    return out.reset_index()


def step_export_vector(cfg: Dict, logger: logging.Logger, overwrite: bool) -> Tuple[Path, Path]:
    logger.info("Step 5/7: export_vector")
    paths = vector_files(cfg)
    if not overwrite and all_exist(paths.values()):
        logger.info("Skip existing vector outputs (overwrite=false).")
        return paths["adm0_clean"], paths["adm0_iso3"]

    mfiles = mapping_files(cfg)
    if not mfiles["clean"].exists():
        raise FileNotFoundError(f"Missing mapping file: {mfiles['clean']}")
    mapping_df = pd.read_csv(mfiles["clean"], dtype={"gid0_raw": str, "iso3_final": str, "country": str})

    adm0 = read_adm0(cfg).reset_index(drop=True)
    adm0["feature_id"] = np.arange(1, len(adm0) + 1, dtype=np.int32)
    adm0_clean = adm0.merge(
        mapping_df[["feature_id", "gid0_raw", "iso3_final", "is_disputed", "dispute_parent"]],
        on="feature_id",
        how="left",
        validate="one_to_one",
    )
    if adm0_clean["iso3_final"].isna().any():
        raise RuntimeError("Mapping join failed: some ADM_0 features have null iso3_final.")

    keep_cols = [
        "feature_id",
        "GID_0",
        "COUNTRY",
        "gid0_raw",
        "iso3_final",
        "is_disputed",
        "dispute_parent",
        "geometry",
    ]
    adm0_clean = adm0_clean[keep_cols]
    ensure_dir(paths["adm0_clean"].parent)
    safe_unlink(paths["adm0_clean"])
    adm0_clean.to_file(paths["adm0_clean"], layer="ADM_0_CLEAN", driver="GPKG")

    dissolved = adm0_clean[["iso3_final", "geometry"]].dissolve(by="iso3_final", as_index=False)
    parts_count = (
        adm0_clean.groupby("iso3_final", as_index=False)
        .size()
        .rename(columns={"size": "parts_count"})
    )
    disputed_parts_count = (
        adm0_clean.loc[adm0_clean["is_disputed"] == 1]
        .groupby("iso3_final", as_index=False)
        .size()
        .rename(columns={"size": "disputed_parts_count"})
    )
    country_name = _country_name_by_iso3(adm0_clean)
    dissolved = dissolved.merge(parts_count, on="iso3_final", how="left")
    dissolved = dissolved.merge(disputed_parts_count, on="iso3_final", how="left")
    dissolved = dissolved.merge(country_name, on="iso3_final", how="left")
    dissolved["disputed_parts_count"] = dissolved["disputed_parts_count"].fillna(0).astype(int)
    dissolved["parts_count"] = dissolved["parts_count"].astype(int)
    dissolved = dissolved.sort_values("iso3_final").reset_index(drop=True)

    safe_unlink(paths["adm0_iso3"])
    dissolved.to_file(paths["adm0_iso3"], layer="ADM_0_ISO3", driver="GPKG")

    logger.info("Vector clean layer: %s", paths["adm0_clean"])
    logger.info("Vector dissolved layer: %s (%d features)", paths["adm0_iso3"], len(dissolved))
    return paths["adm0_clean"], paths["adm0_iso3"]


def find_gdal_rasterize(cfg: Dict) -> str:
    preferred = str(cfg["runtime"].get("gdal_rasterize_bin", "gdal_rasterize"))
    candidates = [preferred, "gdal_rasterize", "gdal_rasterize.exe"]
    for cand in candidates:
        found = shutil.which(cand)
        if found:
            return found
    raise FileNotFoundError("Cannot find gdal_rasterize executable in PATH.")


def raster_nonzero_stats(raster_path: Path) -> Tuple[int, int, float]:
    with rasterio.open(raster_path) as ds:
        total = int(ds.width * ds.height)
        nonzero = 0
        for _, window in ds.block_windows(1):
            arr = ds.read(1, window=window)
            nonzero += int(np.count_nonzero(arr != 0))
    frac = float(nonzero / total) if total > 0 else float("nan")
    return nonzero, total, frac


def step_rasterize_country_id(cfg: Dict, logger: logging.Logger, overwrite: bool) -> Tuple[Path, Path]:
    logger.info("Step 6/7: rasterize_country_id")
    vfiles = vector_files(cfg)
    rfiles = raster_files(cfg)
    if not overwrite and all_exist(rfiles.values()):
        logger.info("Skip existing raster outputs (overwrite=false).")
        return rfiles["country_id_raster"], rfiles["country_lookup"]
    if not vfiles["adm0_iso3"].exists():
        raise FileNotFoundError(f"Missing dissolved vector: {vfiles['adm0_iso3']}")

    gdf_iso3 = gpd.read_file(vfiles["adm0_iso3"], layer="ADM_0_ISO3")
    if "iso3_final" not in gdf_iso3.columns:
        raise ValueError("ADM_0_ISO3 layer missing iso3_final field.")

    gdf_iso3["iso3_final"] = gdf_iso3["iso3_final"].astype(str)
    iso3_sorted = sorted(gdf_iso3["iso3_final"].dropna().unique().tolist())
    id_map = {iso3: idx + 1 for idx, iso3 in enumerate(iso3_sorted)}
    gdf_iso3["country_id"] = gdf_iso3["iso3_final"].map(id_map).astype(np.int32)

    lookup_cols = ["country_id", "iso3_final"]
    for opt_col in ["country_name", "parts_count", "disputed_parts_count"]:
        if opt_col in gdf_iso3.columns:
            lookup_cols.append(opt_col)
    lookup_df = gdf_iso3[lookup_cols].drop_duplicates("iso3_final").sort_values("country_id").reset_index(drop=True)
    write_df_csv(rfiles["country_lookup"], lookup_df)

    safe_unlink(vfiles["adm0_iso3"])
    gdf_iso3.to_file(vfiles["adm0_iso3"], layer="ADM_0_ISO3", driver="GPKG")

    grid = grid_spec(cfg)
    gdal_rasterize = find_gdal_rasterize(cfg)
    ensure_dir(rfiles["country_id_raster"].parent)
    tmp_raster = rfiles["country_id_raster"].with_suffix(".tif.tmp")
    safe_unlink(tmp_raster)

    cmd = [
        gdal_rasterize,
        "-of",
        "GTiff",
        "-l",
        "ADM_0_ISO3",
        "-a",
        "country_id",
        "-ot",
        "Int32",
        "-a_nodata",
        str(grid.nodata),
        "-init",
        str(grid.nodata),
        "-te",
        str(grid.lon_min),
        str(grid.lat_min),
        str(grid.lon_max),
        str(grid.lat_max),
        "-ts",
        str(grid.width),
        str(grid.height),
        "-co",
        f"COMPRESS={grid.compression}",
        "-co",
        f"TILED={'YES' if grid.tiled else 'NO'}",
        "-co",
        f"BIGTIFF={grid.bigtiff}",
        "-co",
        f"BLOCKXSIZE={grid.blockxsize}",
        "-co",
        f"BLOCKYSIZE={grid.blockysize}",
        str(vfiles["adm0_iso3"]),
        str(tmp_raster),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "gdal_rasterize failed:\n"
            + f"command={' '.join(cmd)}\n"
            + f"stdout={proc.stdout}\n"
            + f"stderr={proc.stderr}"
        )

    if rfiles["country_id_raster"].exists():
        safe_unlink(rfiles["country_id_raster"])
    tmp_raster.replace(rfiles["country_id_raster"])

    with rasterio.open(rfiles["country_id_raster"]) as ds:
        if ds.width != grid.width or ds.height != grid.height:
            raise RuntimeError(f"Unexpected raster shape: {ds.width}x{ds.height}")
        if str(ds.crs) != grid.crs:
            raise RuntimeError(f"Unexpected raster CRS: {ds.crs}, expected={grid.crs}")
        if ds.nodata is None or int(ds.nodata) != int(grid.nodata):
            raise RuntimeError(f"Unexpected raster nodata: {ds.nodata}, expected={grid.nodata}")

    nonzero, total, frac = raster_nonzero_stats(rfiles["country_id_raster"])
    if nonzero <= 0:
        raise RuntimeError("Rasterization failed: nonzero pixel count is zero.")

    logger.info("Country lookup: %s (%d rows)", rfiles["country_lookup"], len(lookup_df))
    logger.info(
        "Country ID raster: %s | nonzero=%d total=%d frac=%.6f",
        rfiles["country_id_raster"],
        nonzero,
        total,
        frac,
    )
    return rfiles["country_id_raster"], rfiles["country_lookup"]


def step_report(cfg: Dict, logger: logging.Logger, overwrite: bool) -> Tuple[Path, Path]:
    logger.info("Step 7/7: report")
    rfiles = report_files(cfg)
    if not overwrite and all_exist(rfiles.values()):
        logger.info("Skip existing report outputs (overwrite=false).")
        return rfiles["report_md"], rfiles["qa_summary"]

    inv = inventory_files(cfg)
    mfiles = mapping_files(cfg)
    jfiles = join_files(cfg)
    vfiles = vector_files(cfg)
    rafiles = raster_files(cfg)

    required = [
        inv["source_inventory"],
        mfiles["zxx_log"],
        jfiles["coverage"],
        jfiles["unmatched_gdp"],
        jfiles["unmatched_gadm"],
        vfiles["adm0_iso3"],
        rafiles["country_id_raster"],
        rafiles["country_lookup"],
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files for report: {missing}")

    inventory_payload = json.loads(inv["source_inventory"].read_text(encoding="utf-8"))
    zxx_log = pd.read_csv(mfiles["zxx_log"], dtype=str)
    coverage = pd.read_csv(jfiles["coverage"])
    unmatched_gdp = pd.read_csv(jfiles["unmatched_gdp"], dtype=str)
    unmatched_gadm = pd.read_csv(jfiles["unmatched_gadm"], dtype=str)
    lookup = pd.read_csv(rafiles["country_lookup"])

    parent_counts = {}
    if len(zxx_log) > 0 and "iso3_final" in zxx_log.columns:
        parent_counts = {
            str(k): int(v)
            for k, v in zxx_log["iso3_final"].value_counts().sort_index().to_dict().items()
        }

    cov_rates = coverage["match_rate"].astype(float)
    cov_summary = {
        "years": int(len(coverage)),
        "min_match_rate": float(cov_rates.min()),
        "max_match_rate": float(cov_rates.max()),
        "mean_match_rate": float(cov_rates.mean()),
    }

    with rasterio.open(rafiles["country_id_raster"]) as ds:
        raster_meta = {
            "path": str(rafiles["country_id_raster"]),
            "width": int(ds.width),
            "height": int(ds.height),
            "crs": str(ds.crs),
            "nodata": int(ds.nodata) if ds.nodata is not None else None,
            "dtype": str(ds.dtypes[0]),
            "transform": [float(ds.transform.a), float(ds.transform.b), float(ds.transform.c), float(ds.transform.d), float(ds.transform.e), float(ds.transform.f)],
        }
    nonzero, total, frac = raster_nonzero_stats(rafiles["country_id_raster"])
    raster_meta["nonzero_pixels"] = int(nonzero)
    raster_meta["total_pixels"] = int(total)
    raster_meta["nonzero_fraction"] = float(frac)

    qa = {
        "created_at": datetime.now().isoformat(),
        "config_path": cfg.get("_config_path", ""),
        "input_stats": inventory_payload.get("adm0_stats", {}),
        "zxx_reassignment": {
            "count": int(len(zxx_log)),
            "by_parent_iso3": parent_counts,
        },
        "gdp_join_coverage": cov_summary,
        "unmatched_keys": {
            "unmatched_gdp_codes": int(len(unmatched_gdp)),
            "unmatched_gadm_codes": int(len(unmatched_gadm)),
        },
        "raster": raster_meta,
        "country_lookup_rows": int(len(lookup)),
    }
    write_json(rfiles["qa_summary"], qa)

    report_lines = [
        "# GADM + GDP Preprocess Report (LitPop Pre-step, v1)",
        "",
        f"- created_at: {qa['created_at']}",
        f"- output_root: {resolve_path(cfg, 'output_root')}",
        f"- years: {cfg['years']['start']}-{cfg['years']['end']}",
        "",
        "## Inputs",
        f"- GADM file: `{resolve_path(cfg, 'gadm_gpkg')}`",
        f"- GDP file: `{resolve_path(cfg, 'gdp_csv')}`",
        f"- ADM_0 features: {qa['input_stats'].get('feature_count', 'n/a')}",
        f"- ADM_0 unique COUNTRY: {qa['input_stats'].get('unique_country_count', 'n/a')}",
        f"- ADM_0 unique GID_0: {qa['input_stats'].get('unique_gid0_count', 'n/a')}",
        "",
        "## Zxx Reassignment",
        f"- total reassigned features: {qa['zxx_reassignment']['count']}",
        f"- by parent ISO3: {qa['zxx_reassignment']['by_parent_iso3']}",
        "",
        "## GDP Join Coverage",
        f"- years covered: {qa['gdp_join_coverage']['years']}",
        f"- min match_rate: {qa['gdp_join_coverage']['min_match_rate']:.6f}",
        f"- max match_rate: {qa['gdp_join_coverage']['max_match_rate']:.6f}",
        f"- mean match_rate: {qa['gdp_join_coverage']['mean_match_rate']:.6f}",
        f"- unmatched GDP codes: {qa['unmatched_keys']['unmatched_gdp_codes']}",
        f"- unmatched GADM codes: {qa['unmatched_keys']['unmatched_gadm_codes']}",
        "",
        "## Raster QA",
        f"- raster path: `{raster_meta['path']}`",
        f"- shape: {raster_meta['width']} x {raster_meta['height']}",
        f"- crs: {raster_meta['crs']}",
        f"- nodata: {raster_meta['nodata']}",
        f"- dtype: {raster_meta['dtype']}",
        f"- nonzero fraction: {raster_meta['nonzero_fraction']:.6f}",
        "",
        "## LitPop Applicability Conclusion",
        "- ADM_0  `iso3_final`， `Zxx` 。",
        "-  GDP (2000-2024, NY.GDP.MKTP.CD)  `iso3_final` 。",
        "-  dissolve  30ID， LitPop  C 。",
        "",
        "## Output Files",
        f"- inventory: `{inv['source_inventory']}`",
        f"- mapping_clean: `{mfiles['clean']}`",
        f"- gdp_long: `{gdp_files(cfg)['long']}`",
        f"- joined_long: `{jfiles['joined_long']}`",
        f"- vector_iso3: `{vfiles['adm0_iso3']}`",
        f"- country_id_raster: `{rafiles['country_id_raster']}`",
        f"- lookup: `{rafiles['country_lookup']}`",
        f"- qa_summary: `{rfiles['qa_summary']}`",
    ]
    ensure_dir(rfiles["report_md"].parent)
    rfiles["report_md"].write_text("\n".join(report_lines), encoding="utf-8")

    logger.info("QA summary: %s", rfiles["qa_summary"])
    logger.info("Report markdown: %s", rfiles["report_md"])
    return rfiles["report_md"], rfiles["qa_summary"]


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = project_root() / cfg_path

    cfg = load_config(cfg_path)
    cfg["runtime"]["overwrite"] = bool(args.overwrite or cfg["runtime"].get("overwrite", False))
    overwrite = bool(cfg["runtime"]["overwrite"])

    log_file = output_paths(cfg)["inventory_dir"] / "gadm_gdp_preprocess_run.log"
    logger = setup_logger(log_file)
    logger.info("=" * 72)
    logger.info("GADM + GDP preprocess pipeline started")
    logger.info("Config: %s", cfg_path)
    logger.info("Overwrite: %s", overwrite)
    logger.info("=" * 72)

    steps = parse_steps(args.steps)
    logger.info("Steps: %s", steps)

    for step in steps:
        if step == "inventory":
            step_inventory(cfg, logger, overwrite=overwrite)
        elif step == "build_mapping":
            step_build_mapping(cfg, logger, overwrite=overwrite)
        elif step == "prepare_gdp":
            step_prepare_gdp(cfg, logger, overwrite=overwrite)
        elif step == "join_and_validate":
            step_join_and_validate(cfg, logger, overwrite=overwrite)
        elif step == "export_vector":
            step_export_vector(cfg, logger, overwrite=overwrite)
        elif step == "rasterize_country_id":
            step_rasterize_country_id(cfg, logger, overwrite=overwrite)
        elif step == "report":
            step_report(cfg, logger, overwrite=overwrite)

    logger.info("=" * 72)
    logger.info("GADM + GDP preprocess pipeline completed")
    logger.info("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
