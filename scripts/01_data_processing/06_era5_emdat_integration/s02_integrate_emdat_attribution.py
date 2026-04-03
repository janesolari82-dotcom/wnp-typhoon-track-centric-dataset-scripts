#!/usr/bin/env python3
"""
Phase 2: EM-DAT matching, spatial fallback, and conservative allocation.

Reads phase-1 NetCDF outputs in:
  reproduction_v6/04_final_output/emdat_attribution_integration

Writes:
  - emdat_loss_allocated_usd / emdat_match_count / emdat_loss_weight_norm
  - audit CSV files under .../audit
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Ensure GDAL runtime data paths are defined before importing GDAL-backed modules.
_conda_prefix = os.environ.get("CONDA_PREFIX")
if _conda_prefix and "GDAL_DATA" not in os.environ:
    _p = Path(_conda_prefix) / "Library" / "share" / "gdal"
    if _p.exists():
        os.environ["GDAL_DATA"] = str(_p)
if _conda_prefix and "PROJ_LIB" not in os.environ:
    _p = Path(_conda_prefix) / "Library" / "share" / "proj"
    if _p.exists():
        os.environ["PROJ_LIB"] = str(_p)

import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from netCDF4 import Dataset, num2date
from rasterio.windows import Window
from shapely.ops import unary_union

try:
    from shapely import contains_xy as _contains_xy
except Exception:  # pragma: no cover
    _contains_xy = None

from s00_common import (
    COUNTRY_LOOKUP,
    COUNTRY_RASTER,
    EMDAT_FILE,
    GADM_GPKG,
    LOG_DIR,
    OUTPUT_AUDIT_DIR,
    OUTPUT_NC_DIR,
    STOPWORDS,
    cfdate_to_date,
    end_of_month,
    jaccard,
    name_tokens,
    normalize_text,
    parse_years_expr,
    setup_logger,
)

WINDOW_N = 41
AGG_FACTOR = 10
PATCH_N = WINDOW_N * AGG_FACTOR
HALF_PATCH = PATCH_N // 2
FLOAT_FILL = -9999.0


ALIAS_MAP = {
    "CIRAMON": "CIMARON",
    "TROPCAL": "TROPICAL",
    "TOPICAL": "TROPICAL",
    "TYPHOONN": "TYPHOON",
    "PAENG": "NALGAE",
    "LABUYO": "UTOR",
    "YOLANDA": "HAIYAN",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EM-DAT attribution allocation (phase 2).")
    p.add_argument("--years", type=str, default="2013", help="Year expression, e.g. 2013 or 2000-2024.")
    p.add_argument("--pad-days", type=int, default=1)
    p.add_argument("--match-policy", type=str, default="hybrid_abc", choices=["hybrid_abc"])
    p.add_argument("--overwrite", action="store_true", help="Reset EM-DAT allocation vars if already present.")
    p.add_argument("--log-level", type=str, default="INFO")
    p.add_argument("--emdat-file", type=Path, default=EMDAT_FILE)
    p.add_argument("--gadm-gpkg", type=Path, default=GADM_GPKG)
    p.add_argument("--country-raster", type=Path, default=COUNTRY_RASTER)
    p.add_argument("--country-lookup", type=Path, default=COUNTRY_LOOKUP)
    p.add_argument("--input-nc-dir", type=Path, default=OUTPUT_NC_DIR)
    p.add_argument("--audit-dir", type=Path, default=OUTPUT_AUDIT_DIR)
    return p.parse_args()


def normalize_storm_tokens(text: str) -> List[str]:
    toks = name_tokens(text)
    out = []
    for t in toks:
        out.append(ALIAS_MAP.get(t, t))
    return [t for t in out if t not in STOPWORDS]


def storm_name_score(a: str, b: str) -> float:
    ta = normalize_storm_tokens(a)
    tb = normalize_storm_tokens(b)
    if not ta or not tb:
        return 0.0
    ja = jaccard(ta, tb)
    sa = " ".join(ta)
    sb = " ".join(tb)
    seq = SequenceMatcher(None, sa, sb).ratio()
    return float(max(ja, seq))


def safe_json_loads(text: str):
    if text is None:
        return None
    s = str(text).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


@dataclass
class GroupMeta:
    name: str
    storm_name: str
    name_tokens: List[str]
    start_date: date
    end_date: date
    dates: List[date]
    day_counts: Dict[date, int]

    def overlap_indices(self, d0: date, d1: date) -> List[int]:
        idxs: List[int] = []
        for i, d in enumerate(self.dates):
            if d0 <= d <= d1:
                idxs.append(i)
        return idxs


class GADMIndex:
    def __init__(self, gpkg_path: Path):
        self.gpkg_path = gpkg_path
        self.gid1_to_geom: Dict[str, object] = {}
        self.gid2_to_geom: Dict[str, object] = {}
        self.adm1_name_index: Dict[Tuple[str, str], List[object]] = {}
        self.adm2_name_index: Dict[Tuple[str, str], List[object]] = {}
        self.loaded = False

    @staticmethod
    def _norm_key(x: str) -> str:
        return normalize_text(x)

    @staticmethod
    def _is_valid_geom(geom) -> bool:
        return geom is not None and (not getattr(geom, "is_empty", False))

    def load(self):
        if self.loaded:
            return
        if not self.gpkg_path.exists():
            raise FileNotFoundError(f"Failed to open GADM gpkg: {self.gpkg_path}")

        gdf1 = gpd.read_file(self.gpkg_path, layer="ADM_1", engine="pyogrio")
        for row in gdf1.itertuples(index=False):
            gid0 = str(getattr(row, "GID_0", "") or "").upper()
            gid1 = str(getattr(row, "GID_1", "") or "").upper()
            name1 = str(getattr(row, "NAME_1", "") or "")
            shp = getattr(row, "geometry", None)
            if self._is_valid_geom(shp) and gid1:
                self.gid1_to_geom[gid1] = shp
                key = (gid0, self._norm_key(name1))
                self.adm1_name_index.setdefault(key, []).append(shp)

        gdf2 = gpd.read_file(self.gpkg_path, layer="ADM_2", engine="pyogrio")
        for row in gdf2.itertuples(index=False):
            gid0 = str(getattr(row, "GID_0", "") or "").upper()
            gid2 = str(getattr(row, "GID_2", "") or "").upper()
            name2 = str(getattr(row, "NAME_2", "") or "")
            shp = getattr(row, "geometry", None)
            if self._is_valid_geom(shp) and gid2:
                self.gid2_to_geom[gid2] = shp
                key = (gid0, self._norm_key(name2))
                self.adm2_name_index.setdefault(key, []).append(shp)

        self.loaded = True

    def geom_from_gadm_units(self, text: str) -> Optional[object]:
        if not text or str(text).lower() == "nan":
            return None
        self.load()

        geoms: List[object] = []
        obj = safe_json_loads(text)
        if isinstance(obj, list):
            for item in obj:
                if not isinstance(item, dict):
                    continue
                gid1 = str(item.get("gid_1") or "").upper().strip()
                gid2 = str(item.get("gid_2") or "").upper().strip()
                if gid2 and gid2 in self.gid2_to_geom:
                    geoms.append(self.gid2_to_geom[gid2])
                if gid1 and gid1 in self.gid1_to_geom:
                    geoms.append(self.gid1_to_geom[gid1])

        if not geoms:
            s = str(text)
            p_gid2 = re.compile(r"[A-Z]{3}\.[A-Z0-9]+(?:\.[A-Z0-9]+)?_[0-9]")
            p_gid1 = re.compile(r"[A-Z]{3}\.[A-Z0-9]+(?:_[0-9])?")
            for gid in p_gid2.findall(s):
                if gid in self.gid2_to_geom:
                    geoms.append(self.gid2_to_geom[gid])
            for gid in p_gid1.findall(s):
                if gid in self.gid1_to_geom:
                    geoms.append(self.gid1_to_geom[gid])

        if not geoms:
            return None
        return unary_union(geoms)

    def geom_from_admin_units(self, iso: str, text: str) -> Optional[object]:
        if not text or str(text).lower() == "nan":
            return None
        self.load()
        gid0 = str(iso or "").upper()
        obj = safe_json_loads(text)
        if not isinstance(obj, list):
            return None

        geoms: List[object] = []
        for item in obj:
            if not isinstance(item, dict):
                continue
            n2 = str(item.get("adm2_name") or "").strip()
            n1 = str(item.get("adm1_name") or "").strip()
            if n2 and "ADMINISTRATIVE UNIT NOT AVAILABLE" not in normalize_text(n2):
                key = (gid0, self._norm_key(n2))
                geoms.extend(self.adm2_name_index.get(key, []))
            if n1 and "ADMINISTRATIVE UNIT NOT AVAILABLE" not in normalize_text(n1):
                key = (gid0, self._norm_key(n1))
                geoms.extend(self.adm1_name_index.get(key, []))

        if not geoms:
            return None
        return unary_union(geoms)

    def special_iso_geom(self, iso: str) -> Optional[object]:
        self.load()
        iso = str(iso or "").upper()
        if iso == "HKG":
            g1 = self.gid1_to_geom.get("CHN.HKG")
            if g1 is not None:
                return g1
            geoms = [g for k, g in self.gid2_to_geom.items() if k.startswith("HKG.")]
            if geoms:
                return unary_union(geoms)
        if iso == "MAC":
            g1 = self.gid1_to_geom.get("CHN.MAC")
            if g1 is not None:
                return g1
            geoms = [g for k, g in self.gid2_to_geom.items() if k.startswith("MAC.")]
            if geoms:
                return unary_union(geoms)
        return None


@dataclass
class SpatialStrategy:
    source: str
    detail: str
    geom: Optional[object] = None
    country_id: Optional[int] = None


def contains_mask(geom, lon_abs: np.ndarray, lat_abs: np.ndarray) -> np.ndarray:
    if geom is None:
        return np.zeros_like(lon_abs, dtype=np.float32)
    if _contains_xy is not None:
        out = _contains_xy(geom, lon_abs, lat_abs)
        return out.astype(np.float32)
    # Slow fallback.
    from shapely.geometry import Point

    flat = np.array([geom.contains(Point(float(x), float(y))) for x, y in zip(lon_abs.ravel(), lat_abs.ravel())])
    return flat.reshape(lon_abs.shape).astype(np.float32)


def iso_fraction_mask(
    raster_ds: rasterio.io.DatasetReader, center_lat: float, center_lon: float, target_country_id: int
) -> np.ndarray:
    if target_country_id is None:
        return np.zeros((WINDOW_N, WINDOW_N), dtype=np.float32)
    try:
        row, col = raster_ds.index(center_lon, center_lat)
    except Exception:
        return np.zeros((WINDOW_N, WINDOW_N), dtype=np.float32)

    patch = raster_ds.read(
        1,
        window=Window(col - HALF_PATCH, row - HALF_PATCH, PATCH_N, PATCH_N),
        boundless=True,
        fill_value=0,
    )
    if patch.shape != (PATCH_N, PATCH_N):
        return np.zeros((WINDOW_N, WINDOW_N), dtype=np.float32)

    target = (patch == target_country_id).astype(np.float32)
    out = target.reshape(WINDOW_N, AGG_FACTOR, WINDOW_N, AGG_FACTOR).sum(axis=(1, 3), dtype=np.float32) / float(
        AGG_FACTOR * AGG_FACTOR
    )
    # raster rows are north->south; window_lat is south->north
    return out[::-1, :].astype(np.float32)


def load_country_lookup(path: Path) -> Dict[str, int]:
    df = pd.read_csv(path)
    if "iso3_final" not in df.columns or "country_id" not in df.columns:
        raise RuntimeError(f"Unexpected country lookup columns: {list(df.columns)}")
    out: Dict[str, int] = {}
    for _, r in df.iterrows():
        iso = str(r["iso3_final"]).upper().strip()
        cid = int(r["country_id"])
        out[iso] = cid
    return out


def build_group_meta(ds: Dataset) -> Dict[str, GroupMeta]:
    out: Dict[str, GroupMeta] = {}
    for gname, grp in ds.groups.items():
        if "time" not in grp.variables:
            continue
        time_vals = np.asarray(grp.variables["time"][:], dtype=np.float64)
        units = grp.variables["time"].units
        dts = num2date(time_vals, units=units, calendar="standard")
        dates = [cfdate_to_date(x) for x in dts]
        counts = dict(Counter(dates))
        storm_name = "_".join(gname.split("_")[1:-1]) if "_" in gname else gname
        out[gname] = GroupMeta(
            name=gname,
            storm_name=storm_name,
            name_tokens=normalize_storm_tokens(storm_name),
            start_date=min(dates),
            end_date=max(dates),
            dates=dates,
            day_counts=counts,
        )
    return out


def ensure_emdat_vars(ds: Dataset, overwrite: bool):
    for gname, grp in ds.groups.items():
        if "time" not in grp.dimensions:
            continue

        def _create_or_get(name: str, dtype: str):
            if name in grp.variables:
                if not overwrite:
                    raise RuntimeError(
                        f"{gname}.{name} already exists. Use --overwrite to reset and rerun phase 2 allocation."
                    )
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
                    fill_value=np.float32(FLOAT_FILL),
                    chunksizes=(1, WINDOW_N, WINDOW_N),
                )
                var.missing_value = np.float32(FLOAT_FILL)
            return var

        v_loss = _create_or_get("emdat_loss_allocated_usd", "f4")
        v_cnt = _create_or_get("emdat_match_count", "i2")
        v_wgt = _create_or_get("emdat_loss_weight_norm", "f4")

        # Always reset for deterministic reruns.
        v_loss[:] = np.float32(0.0)
        v_cnt[:] = np.int16(0)
        v_wgt[:] = np.float32(0.0)


def read_emdat_records(emdat_file: Path, year: int, pad_days: int) -> List[dict]:
    df = pd.read_excel(emdat_file, sheet_name="EM-DAT Data")
    df = df[df["Start Year"] == year].copy()
    records: List[dict] = []
    for _, r in df.iterrows():
        disno = str(r.get("DisNo.", "")).strip()
        iso = str(r.get("ISO", "")).strip().upper()
        event_name = str(r.get("Event Name", "")).strip()
        sy = int(r.get("Start Year"))
        sm = int(r.get("Start Month"))
        sd_raw = r.get("Start Day")
        ey = int(r.get("End Year"))
        em = int(r.get("End Month"))
        ed_raw = r.get("End Day")

        date_imputed = False
        if pd.isna(sd_raw):
            sd = 1
            date_imputed = True
        else:
            sd = int(sd_raw)
        if pd.isna(ed_raw):
            ed = end_of_month(ey, em)
            date_imputed = True
        else:
            ed = int(ed_raw)

        d_start = date(sy, sm, sd)
        d_end = date(ey, em, ed)
        pad_start = d_start - timedelta(days=pad_days)
        pad_end = d_end + timedelta(days=pad_days)

        adj = r.get("Total Damage, Adjusted ('000 US$)")
        raw = r.get("Total Damage ('000 US$)")
        if pd.notna(adj):
            loss_usd = float(adj) * 1000.0
            loss_source = "adjusted"
        elif pd.notna(raw):
            loss_usd = float(raw) * 1000.0
            loss_source = "raw_fallback"
        else:
            loss_usd = None
            loss_source = "missing"

        records.append(
            {
                "DisNo": disno,
                "ISO": iso,
                "EventName": event_name,
                "StartDate": d_start,
                "EndDate": d_end,
                "PadStart": pad_start,
                "PadEnd": pad_end,
                "DateImputed": date_imputed,
                "LossUSD": loss_usd,
                "LossSource": loss_source,
                "AdminUnits": r.get("Admin Units"),
                "GADMAdminUnits": r.get("GADM Admin Units"),
            }
        )
    return records


def select_groups_hybrid_abc(record: dict, group_meta: Dict[str, GroupMeta]) -> Tuple[str, List[str], List[Tuple[str, float]]]:
    d0 = record["PadStart"]
    d1 = record["PadEnd"]
    candidates: List[Tuple[str, float, int]] = []
    for gname, meta in group_meta.items():
        if meta.end_date < d0 or meta.start_date > d1:
            continue
        score = storm_name_score(record["EventName"], meta.storm_name)
        overlap_days = (min(meta.end_date, d1) - max(meta.start_date, d0)).days + 1
        candidates.append((gname, float(score), int(max(overlap_days, 0))))

    if not candidates:
        return "D", [], []

    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    scored = [(g, s) for g, s, _ in candidates]

    top = candidates[0]
    second_score = candidates[1][1] if len(candidates) > 1 else 0.0
    if top[1] >= 0.8 and (len(candidates) == 1 or (top[1] - second_score) >= 0.15):
        return "A", [top[0]], scored

    if top[1] >= 0.6:
        cutoff = max(0.6, top[1] - 0.10)
        selected = [g for g, s, _ in candidates if s >= cutoff]
        if len(selected) == 1:
            return "A", selected, scored
        return "B", selected, scored

    # Grade C: mainly spatiotemporal matching.
    selected = [g for g, _, _ in candidates]
    return "C", selected, scored


def resolve_spatial_strategy(
    record: dict, gadm: GADMIndex, iso_to_country_id: Dict[str, int]
) -> Tuple[SpatialStrategy, Optional[str]]:
    # Priority 1: GADM Admin Units
    geom = gadm.geom_from_gadm_units(record["GADMAdminUnits"])
    if geom is not None:
        return SpatialStrategy(source="gadm_gid", detail="GADM Admin Units", geom=geom), None

    # Priority 2: Admin Units names
    geom = gadm.geom_from_admin_units(record["ISO"], record["AdminUnits"])
    if geom is not None:
        return SpatialStrategy(source="admin_name", detail="Admin Units", geom=geom), None

    # Special ISO fallback for HKG/MAC
    if record["ISO"] in {"HKG", "MAC"}:
        geom = gadm.special_iso_geom(record["ISO"])
        if geom is not None:
            return SpatialStrategy(source="gadm_gid", detail=f"special_{record['ISO']}", geom=geom), None

    # Priority 3: ISO country fraction
    iso = record["ISO"]
    cid = iso_to_country_id.get(iso)
    if cid is not None:
        return SpatialStrategy(source="iso_country", detail=f"country_id={cid}", country_id=cid), None

    return SpatialStrategy(source="none", detail="no_spatial_strategy"), f"No spatial strategy for ISO={iso}"


def add_float_slice(var, t_idx: int, add_arr: np.ndarray):
    cur = var[t_idx, :, :]
    if isinstance(cur, np.ma.MaskedArray):
        cur = cur.filled(0.0)
    cur = np.asarray(cur, dtype=np.float32)
    cur[~np.isfinite(cur)] = 0.0
    cur += add_arr.astype(np.float32)
    var[t_idx, :, :] = cur


def add_int_slice(var, t_idx: int, add_mask: np.ndarray):
    cur = var[t_idx, :, :]
    if isinstance(cur, np.ma.MaskedArray):
        cur = cur.filled(0)
    cur = np.asarray(cur, dtype=np.int16)
    cur += add_mask.astype(np.int16)
    var[t_idx, :, :] = cur


def allocate_year(
    year: int,
    args: argparse.Namespace,
    gadm: GADMIndex,
    iso_to_country_id: Dict[str, int],
    logger,
) -> bool:
    nc_path = args.input_nc_dir / f"typhoon_{year}_ocean_litpop_poplight_emdat.nc"
    if not nc_path.exists():
        logger.error("[%s] missing NC output from phase1: %s", year, nc_path)
        return False

    args.audit_dir.mkdir(parents=True, exist_ok=True)
    match_csv = args.audit_dir / f"emdat_record_match_{year}.csv"
    summary_csv = args.audit_dir / f"emdat_record_allocation_summary_{year}.csv"

    records = read_emdat_records(args.emdat_file, year, args.pad_days)
    logger.info("[%s] EM-DAT records: %s", year, len(records))

    match_rows: List[dict] = []
    summary_rows: List[dict] = []

    with rasterio.open(args.country_raster) as country_ds, Dataset(nc_path, "r+") as ds:
        ensure_emdat_vars(ds, overwrite=bool(args.overwrite))
        group_meta = build_group_meta(ds)
        if not group_meta:
            logger.error("[%s] no groups found in NC", year)
            return False

        for rec in records:
            disno = rec["DisNo"]
            iso = rec["ISO"]
            event_name = rec["EventName"]
            loss = rec["LossUSD"]
            loss_source = rec["LossSource"]

            grade, selected_groups, scored_candidates = select_groups_hybrid_abc(rec, group_meta)
            spatial, spatial_err = resolve_spatial_strategy(rec, gadm, iso_to_country_id)

            candidate_text = ";".join(f"{g}:{s:.3f}" for g, s in scored_candidates)
            selected_text = ";".join(selected_groups)
            pre_status = "matched" if selected_groups else "unmatched"
            if spatial.source == "none":
                pre_status = "unmatched"
            if loss is None:
                pre_status = "missing_loss"

            match_rows.append(
                {
                    "DisNo": disno,
                    "ISO": iso,
                    "EventName": event_name,
                    "StartDate": rec["StartDate"].isoformat(),
                    "EndDate": rec["EndDate"].isoformat(),
                    "PadStart": rec["PadStart"].isoformat(),
                    "PadEnd": rec["PadEnd"].isoformat(),
                    "date_imputed_flag": int(rec["DateImputed"]),
                    "loss_value_source": loss_source,
                    "L_usd": "" if loss is None else float(loss),
                    "candidate_groups": candidate_text,
                    "selected_groups": selected_text,
                    "match_grade": grade,
                    "spatial_source": spatial.source,
                    "spatial_detail": spatial.detail,
                    "status_prealloc": pre_status,
                    "note": spatial_err or "",
                }
            )

            if loss is None:
                summary_rows.append(
                    {
                        "DisNo": disno,
                        "ISO": iso,
                        "EventName": event_name,
                        "L_usd": "",
                        "allocated_usd": 0.0,
                        "sum_weight": 0.0,
                        "conservation_error": "",
                        "status": "missing_loss",
                        "match_grade": grade,
                        "spatial_source": spatial.source,
                        "matched_groups": selected_text,
                        "loss_value_source": loss_source,
                    }
                )
                continue

            if not selected_groups or spatial.source == "none":
                summary_rows.append(
                    {
                        "DisNo": disno,
                        "ISO": iso,
                        "EventName": event_name,
                        "L_usd": float(loss),
                        "allocated_usd": 0.0,
                        "sum_weight": 0.0,
                        "conservation_error": 1.0,
                        "status": "unmatched",
                        "match_grade": grade,
                        "spatial_source": spatial.source,
                        "matched_groups": selected_text,
                        "loss_value_source": loss_source,
                    }
                )
                continue

            weights: List[Tuple[str, int, np.ndarray, float]] = []
            sum_p = 0.0
            wlat2d = None
            wlon2d = None

            for gname in selected_groups:
                meta = group_meta[gname]
                grp = ds.groups[gname]
                idxs = meta.overlap_indices(rec["PadStart"], rec["PadEnd"])
                if not idxs:
                    continue

                if wlat2d is None:
                    wlat = np.asarray(grp.variables["window_lat"][:], dtype=np.float32)
                    wlon = np.asarray(grp.variables["window_lon"][:], dtype=np.float32)
                    wlat2d = wlat[:, None]
                    wlon2d = wlon[None, :]

                for t_idx in idxs:
                    c_lat = float(grp.variables["center_lat"][t_idx])
                    c_lon = float(grp.variables["center_lon"][t_idx])
                    lat_abs = c_lat + wlat2d
                    lon_abs = c_lon + wlon2d

                    if spatial.geom is not None:
                        m = contains_mask(spatial.geom, lon_abs, lat_abs)
                    elif spatial.country_id is not None:
                        m = iso_fraction_mask(country_ds, c_lat, c_lon, int(spatial.country_id))
                    else:
                        m = np.zeros((WINDOW_N, WINDOW_N), dtype=np.float32)

                    hazard = np.asarray(grp.variables["hazard_compound_daily"][t_idx, :, :], dtype=np.float32)
                    litpop = np.asarray(grp.variables["litpop"][t_idx, :, :], dtype=np.float32)
                    litpop = np.where(np.isfinite(litpop), litpop, 0.0)
                    litpop = np.maximum(litpop, 0.0)

                    d = meta.dates[t_idx]
                    n_same_day = max(int(meta.day_counts.get(d, 1)), 1)
                    h_eff = hazard / float(n_same_day)
                    p = m * h_eff * litpop
                    p = np.where(np.isfinite(p), p, 0.0).astype(np.float32)
                    p_sum = float(p.sum(dtype=np.float64))
                    if p_sum > 0:
                        weights.append((gname, t_idx, p, p_sum))
                        sum_p += p_sum

            if sum_p <= 0:
                summary_rows.append(
                    {
                        "DisNo": disno,
                        "ISO": iso,
                        "EventName": event_name,
                        "L_usd": float(loss),
                        "allocated_usd": 0.0,
                        "sum_weight": 0.0,
                        "conservation_error": 1.0,
                        "status": "zero_weight",
                        "match_grade": grade,
                        "spatial_source": spatial.source,
                        "matched_groups": selected_text,
                        "loss_value_source": loss_source,
                    }
                )
                continue

            allocated_total = 0.0
            for gname, t_idx, p, _ in weights:
                grp = ds.groups[gname]
                norm = (p / np.float32(sum_p)).astype(np.float32)
                alloc = (np.float32(loss) * norm).astype(np.float32)
                add_float_slice(grp.variables["emdat_loss_allocated_usd"], t_idx, alloc)
                add_float_slice(grp.variables["emdat_loss_weight_norm"], t_idx, norm)
                add_int_slice(grp.variables["emdat_match_count"], t_idx, (p > 0))
                allocated_total += float(alloc.sum(dtype=np.float64))

            err = abs(float(loss) - allocated_total) / max(float(loss), 1.0)
            summary_rows.append(
                {
                    "DisNo": disno,
                    "ISO": iso,
                    "EventName": event_name,
                    "L_usd": float(loss),
                    "allocated_usd": allocated_total,
                    "sum_weight": sum_p,
                    "conservation_error": err,
                    "status": "allocated",
                    "match_grade": grade,
                    "spatial_source": spatial.source,
                    "matched_groups": selected_text,
                    "loss_value_source": loss_source,
                }
            )

        ds.emdat_attribution_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ds.emdat_attribution_method_version = "v4_1"
        ds.emdat_pad_days = int(args.pad_days)
        ds.emdat_match_policy = str(args.match_policy)
        ds.emdat_output_root = str(args.input_nc_dir.parent)

    pd.DataFrame(match_rows).to_csv(match_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    st = Counter([r["status"] for r in summary_rows])
    logger.info(
        "[%s] summary: allocated=%s missing_loss=%s unmatched=%s zero_weight=%s",
        year,
        st.get("allocated", 0),
        st.get("missing_loss", 0),
        st.get("unmatched", 0),
        st.get("zero_weight", 0),
    )
    logger.info("[%s] wrote audit: %s | %s", year, match_csv.name, summary_csv.name)
    return True


def main() -> int:
    args = parse_args()
    years = parse_years_expr(args.years)
    logger = setup_logger("emdat_phase09_attribution", LOG_DIR / "09_integrate_emdat_attribution.log", args.log_level)
    logger.info("Phase 2 started | years=%s | nc_dir=%s", years, args.input_nc_dir)

    if not args.gadm_gpkg.exists():
        logger.error("Missing GADM file: %s", args.gadm_gpkg)
        return 1
    if not args.country_raster.exists():
        logger.error("Missing country raster: %s", args.country_raster)
        return 1
    if not args.country_lookup.exists():
        logger.error("Missing country lookup: %s", args.country_lookup)
        return 1
    if not args.emdat_file.exists():
        logger.error("Missing EM-DAT file: %s", args.emdat_file)
        return 1

    gadm = GADMIndex(args.gadm_gpkg)
    iso_to_country_id = load_country_lookup(args.country_lookup)

    ok = True
    for year in years:
        try:
            this_ok = allocate_year(year, args, gadm, iso_to_country_id, logger)
            ok = ok and this_ok
        except Exception as exc:
            logger.exception("[%s] failed: %s", year, exc)
            ok = False

    logger.info("Phase 2 finished | success=%s", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
