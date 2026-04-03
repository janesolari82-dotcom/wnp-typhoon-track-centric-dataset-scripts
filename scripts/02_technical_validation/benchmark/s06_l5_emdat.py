from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from netCDF4 import Dataset, num2date
from rasterio.windows import Window
from shapely.ops import unary_union

from s00_core_common import (
    RunConfig,
    StageResult,
    build_run_config,
    create_results_layout,
    get_stage_paths,
    make_shared_parser,
    open_netcdf_readonly,
    write_csv,
    write_text,
)

try:
    from shapely import contains_xy as _contains_xy
except Exception:
    _contains_xy = None

WINDOW_N = 41
AGG_FACTOR = 10
PATCH_N = WINDOW_N * AGG_FACTOR
HALF_PATCH = PATCH_N // 2

# L5-A hard-fail thresholds
MAX_CONSERVATION_ERROR = 1e-6
UNMATCHED_YEAR_FAIL_COUNT = 2
UNMATCHED_GLOBAL_RATIO_FAIL = 0.02
AVOIDABLE_ZERO_WEIGHT_RATIO_FAIL = 0.02

# L5 diagnosis partition
UNAVOIDABLE_LABELS = {"mask_all_zero", "litpop_all_zero_under_mask"}
AVOIDABLE_LABELS = {"no_overlap_time_indices", "all_nonfinite_or_missing", "hazard_all_zero_under_mask"}


@dataclass
class GroupMeta:
    name: str
    dates: list[date]

    def overlap_indices(self, d0: date, d1: date) -> list[int]:
        out: list[int] = []
        for i, d in enumerate(self.dates):
            if d0 <= d <= d1:
                out.append(i)
        return out


@dataclass
class SpatialStrategy:
    source: str
    detail: str
    geom: Optional[object] = None
    country_id: Optional[int] = None


def normalize_text(x: str) -> str:
    s = "" if x is None else str(x)
    s = s.upper().strip()
    s = re.sub(r"\s+", " ", s)
    return s


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


def end_of_month(year: int, month: int) -> int:
    import calendar

    return calendar.monthrange(int(year), int(month))[1]


def read_emdat_records_for_year(emdat_file: Path, year: int, pad_days: int = 1) -> list[dict[str, object]]:
    df = pd.read_excel(emdat_file, sheet_name="EM-DAT Data")
    df = df[df["Start Year"] == year].copy()
    records: list[dict[str, object]] = []

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

        if pd.isna(sd_raw):
            sd = 1
        else:
            sd = int(sd_raw)

        if pd.isna(ed_raw):
            ed = end_of_month(ey, em)
        else:
            ed = int(ed_raw)

        start_date = date(sy, sm, sd)
        end_date = date(ey, em, ed)
        pad_start = start_date - timedelta(days=pad_days)
        pad_end = end_date + timedelta(days=pad_days)

        records.append(
            {
                "DisNo": disno,
                "ISO": iso,
                "EventName": event_name,
                "StartDate": start_date,
                "EndDate": end_date,
                "PadStart": pad_start,
                "PadEnd": pad_end,
                "AdminUnits": r.get("Admin Units"),
                "GADMAdminUnits": r.get("GADM Admin Units"),
            }
        )

    return records


class GADMIndex:
    def __init__(self, gpkg_path: Path):
        self.gpkg_path = gpkg_path
        self.gid1_to_geom: dict[str, object] = {}
        self.gid2_to_geom: dict[str, object] = {}
        self.adm1_name_index: dict[tuple[str, str], list[object]] = {}
        self.adm2_name_index: dict[tuple[str, str], list[object]] = {}
        self.loaded = False

    @staticmethod
    def _norm_key(x: str) -> str:
        return normalize_text(x)

    @staticmethod
    def _is_valid_geom(geom) -> bool:
        return geom is not None and (not getattr(geom, "is_empty", False))

    def load(self) -> None:
        if self.loaded:
            return
        if not self.gpkg_path.exists():
            raise FileNotFoundError(f"missing GADM gpkg: {self.gpkg_path}")

        gdf1 = gpd.read_file(self.gpkg_path, layer="ADM_1", engine="pyogrio")
        for row in gdf1.itertuples(index=False):
            gid0 = str(getattr(row, "GID_0", "") or "").upper()
            gid1 = str(getattr(row, "GID_1", "") or "").upper()
            name1 = str(getattr(row, "NAME_1", "") or "")
            shp = getattr(row, "geometry", None)
            if self._is_valid_geom(shp) and gid1:
                self.gid1_to_geom[gid1] = shp
                self.adm1_name_index.setdefault((gid0, self._norm_key(name1)), []).append(shp)

        gdf2 = gpd.read_file(self.gpkg_path, layer="ADM_2", engine="pyogrio")
        for row in gdf2.itertuples(index=False):
            gid0 = str(getattr(row, "GID_0", "") or "").upper()
            gid2 = str(getattr(row, "GID_2", "") or "").upper()
            name2 = str(getattr(row, "NAME_2", "") or "")
            shp = getattr(row, "geometry", None)
            if self._is_valid_geom(shp) and gid2:
                self.gid2_to_geom[gid2] = shp
                self.adm2_name_index.setdefault((gid0, self._norm_key(name2)), []).append(shp)

        self.loaded = True

    def geom_from_gadm_units(self, text: str) -> Optional[object]:
        if not text or str(text).lower() == "nan":
            return None
        self.load()
        geoms: list[object] = []
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

        geoms: list[object] = []
        for item in obj:
            if not isinstance(item, dict):
                continue
            n2 = str(item.get("adm2_name") or "").strip()
            n1 = str(item.get("adm1_name") or "").strip()
            if n2:
                geoms.extend(self.adm2_name_index.get((gid0, self._norm_key(n2)), []))
            if n1:
                geoms.extend(self.adm1_name_index.get((gid0, self._norm_key(n1)), []))

        if not geoms:
            return None
        return unary_union(geoms)


def contains_mask(geom, lon_abs: np.ndarray, lat_abs: np.ndarray) -> np.ndarray:
    if geom is None:
        return np.zeros_like(lon_abs, dtype=np.float32)
    if _contains_xy is not None:
        out = _contains_xy(geom, lon_abs, lat_abs)
        return out.astype(np.float32)

    from shapely.geometry import Point

    flat = np.array([geom.contains(Point(float(x), float(y))) for x, y in zip(lon_abs.ravel(), lat_abs.ravel())])
    return flat.reshape(lon_abs.shape).astype(np.float32)


def iso_fraction_mask(raster_ds: rasterio.io.DatasetReader, center_lat: float, center_lon: float, target_country_id: int) -> np.ndarray:
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
    out = target.reshape(WINDOW_N, AGG_FACTOR, WINDOW_N, AGG_FACTOR).sum(axis=(1, 3), dtype=np.float32) / float(AGG_FACTOR * AGG_FACTOR)
    return out[::-1, :].astype(np.float32)


def load_country_lookup(path: Path) -> dict[str, int]:
    df = pd.read_csv(path)
    if "iso3_final" not in df.columns or "country_id" not in df.columns:
        raise RuntimeError(f"unexpected country lookup columns: {list(df.columns)}")

    out: dict[str, int] = {}
    for _, r in df.iterrows():
        iso = str(r["iso3_final"]).upper().strip()
        out[iso] = int(r["country_id"])
    return out


def resolve_spatial_strategy(record: dict[str, object], gadm: GADMIndex, iso_to_country_id: dict[str, int]) -> SpatialStrategy:
    geom = gadm.geom_from_gadm_units(str(record.get("GADMAdminUnits")))
    if geom is not None:
        return SpatialStrategy(source="gadm_gid", detail="GADM Admin Units", geom=geom)

    geom = gadm.geom_from_admin_units(str(record.get("ISO", "")), str(record.get("AdminUnits")))
    if geom is not None:
        return SpatialStrategy(source="admin_name", detail="Admin Units", geom=geom)

    iso = str(record.get("ISO", "")).upper()
    cid = iso_to_country_id.get(iso)
    if cid is not None:
        return SpatialStrategy(source="iso_country", detail=f"country_id={cid}", country_id=cid)

    return SpatialStrategy(source="none", detail="no_spatial_strategy")


def build_group_meta(ds: Dataset) -> dict[str, GroupMeta]:
    out: dict[str, GroupMeta] = {}
    for gname, grp in ds.groups.items():
        if "time" not in grp.variables:
            continue
        vals = np.asarray(grp.variables["time"][:], dtype=np.float64)
        units = grp.variables["time"].units
        dts = num2date(vals, units=units, calendar="standard")
        dates = [date(int(x.year), int(x.month), int(x.day)) for x in dts]
        out[gname] = GroupMeta(name=gname, dates=dates)
    return out


def _diagnose_zero_weight_record(
    row: pd.Series,
    record: dict[str, object] | None,
    group_meta: dict[str, GroupMeta],
    ds: Dataset,
    country_ds: rasterio.io.DatasetReader,
    gadm: GADMIndex,
    iso_lookup: dict[str, int],
) -> tuple[str, dict[str, object]]:
    matched_groups = str(row.get("matched_groups", "") or "")
    groups = [g.strip() for g in matched_groups.split(";") if g.strip()]

    details = {
        "selected_groups": len(groups),
        "overlap_slices": 0,
        "mask_positive_cells": 0,
        "finite_cells_under_mask": 0,
        "max_hazard_under_mask": np.nan,
        "max_litpop_under_mask": np.nan,
    }

    if not groups:
        return "no_overlap_time_indices", details
    if record is None:
        return "all_nonfinite_or_missing", details

    pad_start = record["PadStart"]
    pad_end = record["PadEnd"]

    overlaps: list[tuple[str, int]] = []
    for g in groups:
        meta = group_meta.get(g)
        if meta is None:
            continue
        idxs = meta.overlap_indices(pad_start, pad_end)
        overlaps.extend((g, i) for i in idxs)

    details["overlap_slices"] = len(overlaps)
    if not overlaps:
        return "no_overlap_time_indices", details

    strategy = resolve_spatial_strategy(record, gadm, iso_lookup)
    if strategy.source == "none":
        return "mask_all_zero", details

    group_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    max_hazard = -np.inf
    max_litpop = -np.inf
    mask_positive_cells = 0
    finite_cells = 0

    for g, idx in overlaps:
        grp = ds.groups.get(g)
        if grp is None:
            continue
        if g not in group_cache:
            wlat = np.asarray(grp.variables["window_lat"][:], dtype=np.float32)
            wlon = np.asarray(grp.variables["window_lon"][:], dtype=np.float32)
            group_cache[g] = (wlat[:, None], wlon[None, :])
        wlat2d, wlon2d = group_cache[g]

        c_lat = float(grp.variables["center_lat"][idx])
        c_lon = float(grp.variables["center_lon"][idx])
        lat_abs = c_lat + wlat2d
        lon_abs = c_lon + wlon2d

        if strategy.geom is not None:
            m = contains_mask(strategy.geom, lon_abs, lat_abs)
        elif strategy.country_id is not None:
            m = iso_fraction_mask(country_ds, c_lat, c_lon, int(strategy.country_id))
        else:
            m = np.zeros((WINDOW_N, WINDOW_N), dtype=np.float32)

        mask = m > 0
        if not np.any(mask):
            continue

        mask_positive_cells += int(np.count_nonzero(mask))

        hazard = np.asarray(grp.variables["hazard_compound_daily"][idx, :, :], dtype=np.float32)
        litpop = np.asarray(grp.variables["litpop"][idx, :, :], dtype=np.float32)

        finite = mask & np.isfinite(hazard) & np.isfinite(litpop)
        if np.any(finite):
            finite_cells += int(np.count_nonzero(finite))

        hz_masked = hazard[mask & np.isfinite(hazard)]
        lp_masked = litpop[mask & np.isfinite(litpop)]

        if hz_masked.size:
            max_hazard = max(max_hazard, float(np.max(hz_masked)))
        if lp_masked.size:
            max_litpop = max(max_litpop, float(np.max(lp_masked)))

    details["mask_positive_cells"] = int(mask_positive_cells)
    details["finite_cells_under_mask"] = int(finite_cells)
    details["max_hazard_under_mask"] = float(max_hazard) if np.isfinite(max_hazard) else np.nan
    details["max_litpop_under_mask"] = float(max_litpop) if np.isfinite(max_litpop) else np.nan

    if mask_positive_cells == 0:
        return "mask_all_zero", details
    if finite_cells == 0:
        return "all_nonfinite_or_missing", details
    if not np.isfinite(max_hazard) or max_hazard <= 0:
        return "hazard_all_zero_under_mask", details
    if not np.isfinite(max_litpop) or max_litpop <= 0:
        return "litpop_all_zero_under_mask", details

    return "all_nonfinite_or_missing", details


def classify_diagnosis_label(label: str) -> tuple[str, str, int]:
    key = str(label or "").strip()
    if key in UNAVOIDABLE_LABELS:
        return "unavoidable", "L5-B", 0
    if key in AVOIDABLE_LABELS:
        return "avoidable", "L5-A", 1
    # Unknown labels are conservatively treated as avoidable (hard-fail track).
    return "avoidable", "L5-A", 1


def run_l5(cfg: RunConfig) -> StageResult:
    layout = create_results_layout(cfg)
    out_dir = layout["L5"]

    match_quality_rows: list[dict[str, object]] = []
    allocation_rows: list[dict[str, object]] = []
    conservation_rows: list[dict[str, object]] = []
    zero_diag_rows: list[dict[str, object]] = []

    repro_root = cfg.reproduction_root
    emdat_file = cfg.raw_using_root / "disaster_records" / "emdat_typhoon_wnp_2000_2024.xlsx"
    gadm_path = cfg.raw_using_root / "gadm" / "gadm_410-levels.gpkg"
    country_raster = repro_root / "02_processed_data" / "gadm_processed_v1" / "05_raster" / "country_id_30as_global.tif"
    country_lookup = repro_root / "02_processed_data" / "gadm_processed_v1" / "05_raster" / "country_id_lookup.csv"

    gadm = GADMIndex(gadm_path)
    iso_lookup = load_country_lookup(country_lookup)

    # Stage-level aggregation (global across selected years)
    global_with_loss_count = 0
    global_unmatched_with_loss_count = 0
    global_zero_weight_count = 0
    global_zero_weight_avoidable_count = 0
    global_zero_weight_unavoidable_count = 0

    year_hard_fail_count = 0
    year_warn_count = 0
    missing_input_years: list[int] = []

    for year in cfg.years:
        paths = get_stage_paths(cfg, year)
        match_path = paths["audit_match"]
        summary_path = paths["audit_summary"]
        final_nc = paths["final"]

        missing_inputs: list[str] = []
        if not match_path.exists():
            missing_inputs.append("audit_match")
        if not summary_path.exists():
            missing_inputs.append("audit_summary")
        if not final_nc.exists():
            missing_inputs.append("final_nc")
        if missing_inputs:
            missing_input_years.append(int(year))
            year_hard_fail_count += 1
            conservation_rows.append(
                {
                    "year": year,
                    "allocated_count": 0,
                    "max_conservation_error": np.nan,
                    "mean_conservation_error": np.nan,
                    "p95_conservation_error": np.nan,
                    "p99_conservation_error": np.nan,
                    "with_loss_count": 0,
                    "unmatched_with_loss_count_year": 0,
                    "with_loss_unmatched_ratio": np.nan,
                    "with_loss_zero_weight_ratio": np.nan,
                    "zero_weight_avoidable_count": 0,
                    "zero_weight_unavoidable_count": 0,
                    "zero_weight_avoidable_ratio": np.nan,
                    "zero_weight_unavoidable_ratio": np.nan,
                    "status": "FAIL",
                    "year_fail_reasons": ";".join(f"missing_{x}" for x in missing_inputs),
                }
            )
            continue

        match_df = pd.read_csv(match_path)
        summary_df = pd.read_csv(summary_path)

        total_match = len(match_df)
        for col, metric_name in [("match_grade", "match_grade"), ("spatial_source", "spatial_source")]:
            vc = match_df[col].astype(str).value_counts(dropna=False)
            for category, count in vc.items():
                match_quality_rows.append(
                    {
                        "year": year,
                        "metric": metric_name,
                        "category": str(category),
                        "count": int(count),
                        "ratio": float(count / total_match) if total_match else np.nan,
                    }
                )

        # allocation status distribution
        for scope_name, sdf in [
            ("all_records", summary_df),
            (
                "with_loss",
                summary_df[pd.to_numeric(summary_df["L_usd"], errors="coerce").fillna(0.0) > 0.0],
            ),
        ]:
            total = len(sdf)
            vc = sdf["status"].astype(str).value_counts(dropna=False)
            for status, count in vc.items():
                allocation_rows.append(
                    {
                        "year": year,
                        "scope": scope_name,
                        "status": str(status),
                        "count": int(count),
                        "ratio": float(count / total) if total else np.nan,
                    }
                )

        # conservation checks
        alloc = summary_df[summary_df["status"].astype(str) == "allocated"].copy()
        alloc["conservation_error"] = pd.to_numeric(alloc.get("conservation_error"), errors="coerce")
        errs = alloc["conservation_error"].dropna().to_numpy(dtype=float)
        max_err = float(np.max(errs)) if errs.size else np.nan
        mean_err = float(np.mean(errs)) if errs.size else np.nan
        p95_err = float(np.quantile(errs, 0.95)) if errs.size else np.nan
        p99_err = float(np.quantile(errs, 0.99)) if errs.size else np.nan

        with_loss = summary_df[pd.to_numeric(summary_df["L_usd"], errors="coerce").fillna(0.0) > 0.0].copy()
        denom_loss = len(with_loss)
        unmatched_with_loss_count_year = int((with_loss["status"].astype(str) == "unmatched").sum())
        zero_weight_count_year = int((with_loss["status"].astype(str) == "zero_weight").sum())
        unmatched_ratio = float(unmatched_with_loss_count_year / denom_loss) if denom_loss else np.nan
        zero_weight_ratio = float(zero_weight_count_year / denom_loss) if denom_loss else np.nan

        # zero_weight diagnosis + classification
        zero_df = with_loss[with_loss["status"].astype(str) == "zero_weight"].copy()
        zero_weight_avoidable_count_year = 0
        zero_weight_unavoidable_count_year = 0

        if not zero_df.empty:
            records = read_emdat_records_for_year(emdat_file, year, pad_days=1)
            rec_map = {str(r["DisNo"]): r for r in records}

            with rasterio.open(country_raster) as country_ds, open_netcdf_readonly(final_nc, cfg) as ds:
                group_meta = build_group_meta(ds)

                for _, zr in zero_df.iterrows():
                    disno = str(zr.get("DisNo", ""))
                    rec = rec_map.get(disno)
                    label, details = _diagnose_zero_weight_record(
                        zr,
                        rec,
                        group_meta,
                        ds,
                        country_ds,
                        gadm,
                        iso_lookup,
                    )
                    diagnosis_class, l5_track, is_avoidable = classify_diagnosis_label(label)
                    if is_avoidable:
                        zero_weight_avoidable_count_year += 1
                    else:
                        zero_weight_unavoidable_count_year += 1

                    zero_diag_rows.append(
                        {
                            "year": year,
                            "DisNo": disno,
                            "ISO": str(zr.get("ISO", "")),
                            "EventName": str(zr.get("EventName", "")),
                            "matched_groups": str(zr.get("matched_groups", "")),
                            "spatial_source": str(zr.get("spatial_source", "")),
                            "diagnosis_label": label,
                            "diagnosis_class": diagnosis_class,
                            "l5_track": l5_track,
                            "is_avoidable": int(is_avoidable),
                            "selected_groups": int(details.get("selected_groups", 0)),
                            "overlap_slices": int(details.get("overlap_slices", 0)),
                            "mask_positive_cells": int(details.get("mask_positive_cells", 0)),
                            "finite_cells_under_mask": int(details.get("finite_cells_under_mask", 0)),
                            "max_hazard_under_mask": details.get("max_hazard_under_mask", np.nan),
                            "max_litpop_under_mask": details.get("max_litpop_under_mask", np.nan),
                        }
                    )

        zero_weight_avoidable_ratio_year = float(zero_weight_avoidable_count_year / denom_loss) if denom_loss else np.nan
        zero_weight_unavoidable_ratio_year = float(zero_weight_unavoidable_count_year / denom_loss) if denom_loss else np.nan

        # L5-A hard fail (year scope)
        year_fail_reasons: list[str] = []
        if np.isfinite(max_err) and max_err > MAX_CONSERVATION_ERROR:
            year_fail_reasons.append("max_conservation_error")
        if unmatched_with_loss_count_year >= UNMATCHED_YEAR_FAIL_COUNT:
            year_fail_reasons.append("unmatched_with_loss_count_year")

        if year_fail_reasons:
            year_status = "FAIL"
            year_hard_fail_count += 1
        elif np.isfinite(zero_weight_unavoidable_ratio_year) and zero_weight_unavoidable_ratio_year > 0.0:
            # L5-B warn-only track
            year_status = "WARN"
            year_warn_count += 1
        else:
            year_status = "PASS"

        conservation_rows.append(
            {
                "year": year,
                "allocated_count": int(len(alloc)),
                "max_conservation_error": max_err,
                "mean_conservation_error": mean_err,
                "p95_conservation_error": p95_err,
                "p99_conservation_error": p99_err,
                "with_loss_count": int(denom_loss),
                "unmatched_with_loss_count_year": int(unmatched_with_loss_count_year),
                "with_loss_unmatched_ratio": unmatched_ratio,
                "with_loss_zero_weight_ratio": zero_weight_ratio,
                "zero_weight_avoidable_count": int(zero_weight_avoidable_count_year),
                "zero_weight_unavoidable_count": int(zero_weight_unavoidable_count_year),
                "zero_weight_avoidable_ratio": zero_weight_avoidable_ratio_year,
                "zero_weight_unavoidable_ratio": zero_weight_unavoidable_ratio_year,
                "status": year_status,
                "year_fail_reasons": ";".join(year_fail_reasons),
            }
        )

        global_with_loss_count += int(denom_loss)
        global_unmatched_with_loss_count += int(unmatched_with_loss_count_year)
        global_zero_weight_count += int(zero_weight_count_year)
        global_zero_weight_avoidable_count += int(zero_weight_avoidable_count_year)
        global_zero_weight_unavoidable_count += int(zero_weight_unavoidable_count_year)

    match_quality_df = pd.DataFrame(match_quality_rows)
    allocation_df = pd.DataFrame(allocation_rows)
    conservation_df = pd.DataFrame(conservation_rows)
    zero_diag_df = pd.DataFrame(zero_diag_rows)

    match_quality_path = out_dir / "match_quality_by_year.csv"
    allocation_status_path = out_dir / "allocation_status_by_year.csv"
    conservation_path = out_dir / "conservation_check.csv"
    zero_diag_path = out_dir / "zero_weight_diagnosis.csv"

    write_csv(match_quality_df, match_quality_path, cfg)
    write_csv(allocation_df, allocation_status_path, cfg)
    write_csv(conservation_df, conservation_path, cfg)
    write_csv(zero_diag_df, zero_diag_path, cfg)

    unmatched_with_loss_ratio_global = (
        float(global_unmatched_with_loss_count / global_with_loss_count) if global_with_loss_count else np.nan
    )
    zero_weight_avoidable_ratio_global = (
        float(global_zero_weight_avoidable_count / global_with_loss_count) if global_with_loss_count else np.nan
    )
    zero_weight_unavoidable_ratio_global = (
        float(global_zero_weight_unavoidable_count / global_with_loss_count) if global_with_loss_count else np.nan
    )

    global_fail_reasons: list[str] = []
    if missing_input_years:
        global_fail_reasons.append("missing_required_inputs")
    if year_hard_fail_count > 0:
        global_fail_reasons.append("year_level_hard_fail")
    if np.isfinite(unmatched_with_loss_ratio_global) and unmatched_with_loss_ratio_global > UNMATCHED_GLOBAL_RATIO_FAIL:
        global_fail_reasons.append("unmatched_with_loss_ratio_global")
    if np.isfinite(zero_weight_avoidable_ratio_global) and zero_weight_avoidable_ratio_global > AVOIDABLE_ZERO_WEIGHT_RATIO_FAIL:
        global_fail_reasons.append("zero_weight_avoidable_ratio_global")

    if global_fail_reasons:
        stage_status = "FAIL"
    elif np.isfinite(zero_weight_unavoidable_ratio_global) and zero_weight_unavoidable_ratio_global > 0.0:
        stage_status = "WARN"
    else:
        stage_status = "PASS"

    fail_count = int(year_hard_fail_count)
    if np.isfinite(unmatched_with_loss_ratio_global) and unmatched_with_loss_ratio_global > UNMATCHED_GLOBAL_RATIO_FAIL:
        fail_count += 1
    if np.isfinite(zero_weight_avoidable_ratio_global) and zero_weight_avoidable_ratio_global > AVOIDABLE_ZERO_WEIGHT_RATIO_FAIL:
        fail_count += 1
    warn_count = int(year_warn_count)
    if stage_status == "WARN" and warn_count == 0:
        warn_count = 1

    def _fmt_ratio(x: float) -> str:
        if not np.isfinite(x):
            return "nan"
        return f"{x:.4f}"

    def _fmt_err(x: float) -> str:
        if not np.isfinite(x):
            return "nan"
        return f"{x:.8e}"

    summary_lines: list[str] = []
    summary_lines.append("# L5 EMDAT Validation Summary")
    summary_lines.append("")
    summary_lines.append("Thresholds:")
    summary_lines.append("- L5-A hard fail:")
    summary_lines.append(f"  - max(conservation_error) <= {MAX_CONSERVATION_ERROR:.0e}")
    summary_lines.append(f"  - unmatched_with_loss_count_year < {UNMATCHED_YEAR_FAIL_COUNT}")
    summary_lines.append(f"  - unmatched_with_loss_ratio_global <= {UNMATCHED_GLOBAL_RATIO_FAIL:.0%}")
    summary_lines.append(f"  - zero_weight_avoidable_ratio <= {AVOIDABLE_ZERO_WEIGHT_RATIO_FAIL:.0%}")
    summary_lines.append("- L5-B warn only:")
    summary_lines.append("  - zero_weight_unavoidable_ratio > 0 => WARN (not FAIL)")
    summary_lines.append("")
    summary_lines.append("| year | max_err | unmatched_count | unmatched_ratio | zero_weight_ratio | avoidable_ratio | unavoidable_ratio | status |")
    summary_lines.append("|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in conservation_df.sort_values("year").itertuples(index=False):
        summary_lines.append(
            f"| {int(r.year)} | {_fmt_err(float(r.max_conservation_error))} | "
            f"{int(getattr(r, 'unmatched_with_loss_count_year', 0))} | "
            f"{_fmt_ratio(float(r.with_loss_unmatched_ratio))} | "
            f"{_fmt_ratio(float(r.with_loss_zero_weight_ratio))} | "
            f"{_fmt_ratio(float(getattr(r, 'zero_weight_avoidable_ratio', np.nan)))} | "
            f"{_fmt_ratio(float(getattr(r, 'zero_weight_unavoidable_ratio', np.nan)))} | "
            f"{r.status} |"
        )
    summary_lines.append("")
    summary_lines.append("## Global Metrics")
    summary_lines.append("")
    summary_lines.append(f"- with_loss_count_global: {global_with_loss_count}")
    summary_lines.append(f"- unmatched_with_loss_count_global: {global_unmatched_with_loss_count}")
    summary_lines.append(f"- zero_weight_count_global: {global_zero_weight_count}")
    summary_lines.append(f"- zero_weight_avoidable_count_global: {global_zero_weight_avoidable_count}")
    summary_lines.append(f"- zero_weight_unavoidable_count_global: {global_zero_weight_unavoidable_count}")
    summary_lines.append(f"- unmatched_with_loss_ratio_global: {_fmt_ratio(unmatched_with_loss_ratio_global)}")
    summary_lines.append(f"- zero_weight_avoidable_ratio_global: {_fmt_ratio(zero_weight_avoidable_ratio_global)}")
    summary_lines.append(f"- zero_weight_unavoidable_ratio_global: {_fmt_ratio(zero_weight_unavoidable_ratio_global)}")
    summary_lines.append("")
    summary_lines.append("## Final Decision")
    summary_lines.append("")
    summary_lines.append(f"- stage_status: **{stage_status}**")
    if global_fail_reasons:
        summary_lines.append(f"- fail_reasons: {', '.join(global_fail_reasons)}")
    else:
        summary_lines.append("- fail_reasons: none")
    if missing_input_years:
        summary_lines.append(f"- missing_input_years: {','.join(str(x) for x in missing_input_years)}")
    else:
        summary_lines.append("- missing_input_years: none")

    summary_md = out_dir / "emdat_summary.md"
    write_text("\n".join(summary_lines) + "\n", summary_md, cfg)

    return StageResult(
        level="L5",
        status=stage_status,
        fail_count=fail_count,
        warn_count=warn_count,
        metrics={
            "conservation_rows": int(len(conservation_df)),
            "zero_weight_diagnosed": int(len(zero_diag_df)),
            "with_loss_count_global": int(global_with_loss_count),
            "unmatched_with_loss_ratio_global": unmatched_with_loss_ratio_global,
            "zero_weight_avoidable_ratio_global": zero_weight_avoidable_ratio_global,
            "zero_weight_unavoidable_ratio_global": zero_weight_unavoidable_ratio_global,
        },
        artifacts=[
            str(match_quality_path),
            str(allocation_status_path),
            str(conservation_path),
            str(zero_diag_path),
            str(summary_md),
        ],
        notes=[f"global_fail_reasons={','.join(global_fail_reasons)}" if global_fail_reasons else "global_fail_reasons=none"],
    )


def main() -> int:
    parser = make_shared_parser("L5 EM-DAT validation")
    args = parser.parse_args()
    cfg = build_run_config(args)
    result = run_l5(cfg)
    print(f"L5 status={result.status} fail={result.fail_count} warn={result.warn_count}")
    if cfg.strict and result.status == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
