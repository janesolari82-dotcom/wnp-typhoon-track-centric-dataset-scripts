from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
from shapely.geometry import LineString, Point

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = THIS_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from benchmark.s00_core_common import (  # noqa: E402
    RunConfig,
    StageResult,
    assert_write_allowed,
    build_run_config,
    combine_level_status,
    ensure_dir,
    file_hash_sample,
    format_stage_table,
    make_note_lines,
    make_shared_parser,
    open_netcdf_readonly,
    read_only_snapshot,
    stage_should_fail,
    status_from_thresholds,
    write_csv,
    write_json,
    write_run_manifest,
    write_text,
)

CORE_EXT_START = 2000
GDIS_EXT_END = 2018
IMERG_EXT_END = 2020

LEVEL_DIRS = {
    "L0": "00_integrity",
    "L1": "01_schema",
    "L2": "02_track_physics",
    "L3": "03_alignment",
    "L4": "04_hazard_formula",
    "L5": "05_emdat",
    "L6": "06_external_validation",
    "L7": "07_case_validation",
}

STATUS_CODE = {"PASS": 0, "WARN": 1, "FAIL": 2}
TRACK_VARS = ["center_lat", "center_lon", "wind_speed", "central_pressure", "radius_max_wind"]
FILL_VALUE = -9999.0


@dataclass
class IMERGMeta:
    lon0: float
    lat0: float
    dlon: float
    dlat: float
    lon_n: int
    lat_n: int
    path: Path


def make_shared_parser_v2(description: str) -> argparse.ArgumentParser:
    parser = make_shared_parser(description)
    parser.add_argument("--imerg-samples-per-year", type=int, default=300)
    parser.add_argument("--imerg-topq", type=str, default="0.10,0.20")
    parser.add_argument("--case-count", type=int, default=5)
    return parser


def build_run_config_v2(args: argparse.Namespace) -> RunConfig:
    if getattr(args, "results_root", None) is None:
        args.results_root = Path(args.project_root).resolve() / "reproduction_v6" / "05_benchmark_results" / "results_v2"
    return build_run_config(args)


def create_results_layout_v2(cfg: RunConfig) -> dict[str, Path]:
    root = ensure_dir(cfg.results_root, cfg)
    dirs: dict[str, Path] = {"root": root}
    for level, dirname in LEVEL_DIRS.items():
        stage_root = ensure_dir(root / dirname, cfg)
        dirs[level] = stage_root
        dirs[f"{level}_csv"] = ensure_dir(stage_root / "csv", cfg)
        dirs[f"{level}_png"] = ensure_dir(stage_root / "png", cfg)
        dirs[f"{level}_md"] = ensure_dir(stage_root / "md", cfg)
    dirs["reports"] = ensure_dir(root / "reports", cfg)
    return dirs


def stage_dirs(cfg: RunConfig, level: str) -> dict[str, Path]:
    layout = create_results_layout_v2(cfg)
    return {
        "root": layout[level],
        "csv": layout[f"{level}_csv"],
        "png": layout[f"{level}_png"],
        "md": layout[f"{level}_md"],
    }


def append_artifacts(result: StageResult, artifacts: Iterable[str], notes: Iterable[str] | None = None) -> StageResult:
    for item in artifacts:
        if item not in result.artifacts:
            result.artifacts.append(str(item))
    if notes:
        for note in notes:
            if note not in result.notes:
                result.notes.append(str(note))
    return result


def save_figure(fig, path: Path, cfg: RunConfig) -> None:
    assert_write_allowed(path, cfg.write_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_plot_data(df: pd.DataFrame, path: Path, cfg: RunConfig) -> None:
    write_csv(df, path, cfg)


def parse_date_safe(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    try:
        out = pd.to_datetime(text, errors="coerce")
        if pd.isna(out):
            return None
        return out.date()
    except Exception:
        return None


def parse_disno_base(disno: str) -> str:
    text = str(disno or "").strip().upper()
    match = re.search(r"(\d{4}-\d{4})", text)
    return match.group(1) if match else ""


def split_groups(value: str) -> list[str]:
    text = str(value or "")
    return [item.strip() for item in re.split(r"[;,]", text) if item.strip()]


def normalize_cause(text: str) -> str:
    value = str(text or "").strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def safe_ratio(num: float, den: float) -> float:
    if den is None or den == 0:
        return float("nan")
    return float(num) / float(den)


def quantile_safe(values: Sequence[float], q: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanquantile(arr, q))


def corr_safe(x: Sequence[float], y: Sequence[float]) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if np.count_nonzero(mask) < 3:
        return float("nan")
    xv = xa[mask]
    yv = ya[mask]
    if np.nanstd(xv) == 0 or np.nanstd(yv) == 0:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def rmse_safe(x: Sequence[float], y: Sequence[float]) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if np.count_nonzero(mask) == 0:
        return float("nan")
    diff = xa[mask] - ya[mask]
    return float(np.sqrt(np.mean(diff * diff)))


def bias_safe(x: Sequence[float], y: Sequence[float]) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if np.count_nonzero(mask) == 0:
        return float("nan")
    return float(np.mean(xa[mask] - ya[mask]))


def valid_mask_default(arr: np.ndarray) -> np.ndarray:
    return np.isfinite(arr) & (arr != -999.0) & (arr != -9999.0) & (arr != -9999.9)


def wrap_lon_180(lon: np.ndarray | float) -> np.ndarray | float:
    return ((np.asarray(lon) + 180.0) % 360.0) - 180.0


def haversine_km(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    radius = 6371.0088
    lon1r = np.radians(lon1)
    lat1r = np.radians(lat1)
    lon2r = np.radians(lon2)
    lat2r = np.radians(lat2)
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return radius * c


def build_track_geometry(points: pd.DataFrame):
    if points.empty:
        return None
    coords_df = points[["lon", "lat"]].dropna()
    if coords_df.empty:
        return None
    coords = [tuple(value) for value in coords_df.to_numpy(dtype=float)]
    unique_coords: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for xy in coords:
        if xy not in seen:
            seen.add(xy)
            unique_coords.append(xy)
    if len(unique_coords) == 1:
        return Point(unique_coords[0])
    return LineString(unique_coords)


def grade_priority(match_grade: str) -> int:
    grade = str(match_grade or "").upper().strip()
    if grade == "A":
        return 0
    if grade == "B":
        return 1
    if grade == "C":
        return 2
    if grade == "D":
        return 3
    return 4


def finalize_stage_status(has_fail: bool, has_warn: bool) -> str:
    if has_fail:
        return "FAIL"
    if has_warn:
        return "WARN"
    return "PASS"


def split_external_windows(years: Sequence[int]) -> tuple[list[int], list[int], list[int]]:
    core = [int(y) for y in years if CORE_EXT_START <= int(y) <= GDIS_EXT_END]
    imerg_only = [int(y) for y in years if GDIS_EXT_END < int(y) <= IMERG_EXT_END]
    not_eval = [int(y) for y in years if int(y) > IMERG_EXT_END or int(y) < CORE_EXT_START]
    return sorted(core), sorted(imerg_only), sorted(not_eval)


def imerg_date_from_filename(path: Path) -> date | None:
    match = re.search(r"3IMERG\.(\d{8})-S", path.name)
    if not match:
        return None
    text = match.group(1)
    return date(int(text[:4]), int(text[4:6]), int(text[6:8]))


def build_imerg_index(root: Path) -> dict[date, Path]:
    out: dict[date, Path] = {}
    for path in sorted(root.glob("*.nc4")):
        dt = imerg_date_from_filename(path)
        if dt is not None:
            out[dt] = path
    return out


def load_imerg_meta(index: dict[date, Path]) -> IMERGMeta:
    if not index:
        raise FileNotFoundError("IMERG index is empty")
    first = index[sorted(index.keys())[0]]
    with Dataset(first, "r") as ds:
        lon = np.asarray(ds.variables["lon"][:], dtype=float)
        lat = np.asarray(ds.variables["lat"][:], dtype=float)
    return IMERGMeta(
        lon0=float(lon[0]),
        lat0=float(lat[0]),
        dlon=float(lon[1] - lon[0]),
        dlat=float(lat[1] - lat[0]),
        lon_n=int(lon.size),
        lat_n=int(lat.size),
        path=first,
    )


def _prepare_grid_indices(lat_grid: np.ndarray, lon_grid: np.ndarray, meta: IMERGMeta):
    lon_wrapped = wrap_lon_180(lon_grid).astype(float)
    lat_clipped = np.clip(np.asarray(lat_grid, dtype=float), meta.lat0, meta.lat0 + meta.dlat * (meta.lat_n - 1))
    fx = (lon_wrapped - meta.lon0) / meta.dlon
    fy = (lat_clipped - meta.lat0) / meta.dlat
    i0 = np.floor(fx).astype(int) % meta.lon_n
    j0 = np.floor(fy).astype(int)
    j0 = np.clip(j0, 0, meta.lat_n - 2)
    i1 = (i0 + 1) % meta.lon_n
    j1 = np.clip(j0 + 1, 0, meta.lat_n - 1)
    wx = fx - np.floor(fx)
    wy = fy - np.floor(fy)
    return i0, i1, j0, j1, wx.astype(float), wy.astype(float)


def _read_imerg_subset(ds: Dataset, var_name: str, i0: np.ndarray, i1: np.ndarray, j0: np.ndarray, j1: np.ndarray):
    lon_needed = np.unique(np.concatenate([i0.ravel(), i1.ravel()]))
    lat_min = int(min(j0.min(), j1.min()))
    lat_max = int(max(j0.max(), j1.max()))
    contiguous = lon_needed.size > 0 and (int(lon_needed.max()) - int(lon_needed.min()) + 1 <= lon_needed.size + 2)
    var = ds.variables[var_name]
    if contiguous:
        lon_min = int(lon_needed.min())
        lon_max = int(lon_needed.max())
        arr = np.asarray(var[0, lon_min : lon_max + 1, lat_min : lat_max + 1], dtype=float).T
        return arr, lon_min, lat_min, False
    arr = np.asarray(var[0, :, lat_min : lat_max + 1], dtype=float).T
    return arr, 0, lat_min, True


def _apply_fill_mask(arr: np.ndarray, fill_value) -> np.ndarray:
    out = np.asarray(arr, dtype=float).copy()
    out[~np.isfinite(out)] = np.nan
    if fill_value is not None:
        out[out == float(fill_value)] = np.nan
    return out


def _bilinear_sample_from_subset(
    arr: np.ndarray,
    i0: np.ndarray,
    i1: np.ndarray,
    j0: np.ndarray,
    j1: np.ndarray,
    wx: np.ndarray,
    wy: np.ndarray,
    lon_offset: int,
    lat_offset: int,
) -> np.ndarray:
    if lon_offset == 0 and arr.shape[1] > 1000:
        li0 = i0
        li1 = i1
    else:
        li0 = i0 - lon_offset
        li1 = i1 - lon_offset
    lj0 = j0 - lat_offset
    lj1 = j1 - lat_offset

    v00 = arr[lj0, li0]
    v10 = arr[lj0, li1]
    v01 = arr[lj1, li0]
    v11 = arr[lj1, li1]

    w00 = (1.0 - wx) * (1.0 - wy)
    w10 = wx * (1.0 - wy)
    w01 = (1.0 - wx) * wy
    w11 = wx * wy

    vals = np.stack([v00, v10, v01, v11], axis=0)
    weights = np.stack([w00, w10, w01, w11], axis=0)
    valid = np.isfinite(vals)
    weight_sum = np.sum(np.where(valid, weights, 0.0), axis=0)
    num = np.sum(np.where(valid, vals * weights, 0.0), axis=0)
    out = np.full_like(v00, np.nan, dtype=float)
    np.divide(num, weight_sum, out=out, where=weight_sum > 0)
    return out


def _nearest_sample_from_subset(
    arr: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    lon_offset: int,
    lat_offset: int,
) -> np.ndarray:
    ii = np.rint(fx).astype(int)
    jj = np.rint(fy).astype(int)
    if lon_offset == 0 and arr.shape[1] > 1000:
        li = ii % arr.shape[1]
    else:
        li = ii - lon_offset
    lj = jj - lat_offset
    lj = np.clip(lj, 0, arr.shape[0] - 1)
    li = np.clip(li, 0, arr.shape[1] - 1)
    return arr[lj, li]


def sample_imerg_fields(file_path_or_dataset, lat_grid: np.ndarray, lon_grid: np.ndarray, meta: IMERGMeta) -> dict[str, np.ndarray]:
    i0, i1, j0, j1, wx, wy = _prepare_grid_indices(lat_grid, lon_grid, meta)
    fx = (wrap_lon_180(lon_grid).astype(float) - meta.lon0) / meta.dlon
    fy = (np.clip(np.asarray(lat_grid, dtype=float), meta.lat0, meta.lat0 + meta.dlat * (meta.lat_n - 1)) - meta.lat0) / meta.dlat
    fields: dict[str, np.ndarray] = {}
    close_after = False
    if isinstance(file_path_or_dataset, (str, Path)):
        ds = Dataset(file_path_or_dataset, "r")
        close_after = True
    else:
        ds = file_path_or_dataset
    try:
        for name in ["precipitation", "MWprecipitation", "randomError", "precipitation_cnt", "precipitation_cnt_cond"]:
            arr, lon_offset, lat_offset, _ = _read_imerg_subset(ds, name, i0, i1, j0, j1)
            fill_value = getattr(ds.variables[name], "_FillValue", None)
            arr = _apply_fill_mask(arr, fill_value)
            if name in {"precipitation", "MWprecipitation", "randomError"}:
                sampled = _bilinear_sample_from_subset(arr, i0, i1, j0, j1, wx, wy, lon_offset, lat_offset)
            else:
                sampled = _nearest_sample_from_subset(arr, fx, fy, lon_offset, lat_offset)
            fields[name] = sampled.astype(float)
    finally:
        if close_after:
            ds.close()
    return fields


def project_window_abs_coords(grp, t_idx: int) -> tuple[np.ndarray, np.ndarray]:
    center_lat = float(grp.variables["center_lat"][t_idx])
    center_lon = float(grp.variables["center_lon"][t_idx])
    window_lat = np.asarray(grp.variables["window_lat"][:], dtype=float)
    window_lon = np.asarray(grp.variables["window_lon"][:], dtype=float)
    lat_abs = center_lat + np.broadcast_to(window_lat[:, None], (window_lat.size, window_lon.size))
    lon_abs = center_lon + np.broadcast_to(window_lon[None, :], (window_lat.size, window_lon.size))
    return lat_abs, wrap_lon_180(lon_abs).astype(float)


def event_points_from_group(grp, idx: int, hazard_q: float = 0.9) -> pd.DataFrame:
    var_names = set(grp.variables.keys())
    hazard_name = "hazard_compound_daily" if "hazard_compound_daily" in var_names else "hazard_compound"
    rain_name = "hazard_rain_daily" if "hazard_rain_daily" in var_names else "hazard_rain"
    loss_name = "emdat_loss_allocated_usd" if "emdat_loss_allocated_usd" in var_names else None

    lat_abs, lon_abs = project_window_abs_coords(grp, idx)
    hazard = np.asarray(grp.variables[hazard_name][idx, :, :], dtype=float)
    rain = np.asarray(grp.variables[rain_name][idx, :, :], dtype=float)
    if loss_name is not None:
        loss = np.asarray(grp.variables[loss_name][idx, :, :], dtype=float)
    else:
        loss = np.zeros_like(hazard, dtype=float)

    finite = np.isfinite(hazard)
    if np.any(finite):
        qv = float(np.nanquantile(hazard[finite], hazard_q))
        mask = finite & ((hazard >= qv) | (loss > 0))
    else:
        mask = loss > 0

    if not np.any(mask):
        if np.any(np.isfinite(hazard)):
            arg = int(np.nanargmax(hazard))
            rr, cc = np.unravel_index(arg, hazard.shape)
            mask = np.zeros_like(hazard, dtype=bool)
            mask[rr, cc] = True
        else:
            mask = np.zeros_like(hazard, dtype=bool)
            mask[hazard.shape[0] // 2, hazard.shape[1] // 2] = True

    rr, cc = np.where(mask)
    return pd.DataFrame(
        {
            "lat": lat_abs[rr, cc].astype(float),
            "lon": lon_abs[rr, cc].astype(float),
            "hazard_compound": hazard[rr, cc].astype(float),
            "hazard_rain": rain[rr, cc].astype(float),
            "loss_alloc": loss[rr, cc].astype(float),
        }
    )


def extract_event_points(
    ds,
    selected_groups: Sequence[str],
    pad_start: date | None,
    pad_end: date | None,
    seed: int,
    max_points: int = 1500,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for group_name in selected_groups:
        grp = ds.groups.get(group_name)
        if grp is None or "time" not in grp.variables:
            continue
        time_values = np.asarray(grp.variables["time"][:], dtype=np.float64)
        time_units = str(grp.variables["time"].units)
        dts = num2date(time_values, units=time_units, calendar="standard")
        for idx, dt in enumerate(dts):
            current_date = date(int(dt.year), int(dt.month), int(dt.day))
            if pad_start and current_date < pad_start:
                continue
            if pad_end and current_date > pad_end:
                continue
            frame = event_points_from_group(grp, idx)
            if frame.empty:
                continue
            frame["time"] = current_date
            frame["group"] = group_name
            rows.append(frame)
    if not rows:
        return pd.DataFrame(columns=["time", "lat", "lon", "hazard_compound", "hazard_rain", "loss_alloc", "group"])
    out = pd.concat(rows, ignore_index=True)
    if len(out) > max_points:
        out = out.sample(n=max_points, random_state=int(seed))
    return out.reset_index(drop=True)


def load_project_events_from_audit(cfg: RunConfig, years: Sequence[int]) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for year in years:
        paths = {
            "final": cfg.reproduction_root / "04_final_output" / "emdat_attribution_integration" / f"typhoon_{year}_ocean_litpop_poplight_emdat.nc",
            "audit_match": cfg.reproduction_root / "04_final_output" / "emdat_attribution_integration" / "audit" / f"emdat_record_match_{year}.csv",
            "audit_summary": cfg.reproduction_root / "04_final_output" / "emdat_attribution_integration" / "audit" / f"emdat_record_allocation_summary_{year}.csv",
        }
        if not paths["audit_summary"].exists():
            missing.append({"year": str(year), "missing": "audit_summary", "path": str(paths["audit_summary"])})
            continue
        if not paths["audit_match"].exists():
            missing.append({"year": str(year), "missing": "audit_match", "path": str(paths["audit_match"])})
            continue
        if not paths["final"].exists():
            missing.append({"year": str(year), "missing": "final_nc", "path": str(paths["final"])})
            continue

        summary_df = pd.read_csv(paths["audit_summary"])
        match_df = pd.read_csv(paths["audit_match"])
        req_s = {"DisNo", "ISO", "L_usd", "status", "match_grade"}
        req_m = {"DisNo", "ISO", "PadStart", "PadEnd", "selected_groups"}
        if not req_s.issubset(summary_df.columns):
            raise RuntimeError(f"summary_{year} missing columns: {sorted(req_s - set(summary_df.columns))}")
        if not req_m.issubset(match_df.columns):
            raise RuntimeError(f"match_{year} missing columns: {sorted(req_m - set(match_df.columns))}")

        agg = (
            match_df.groupby(["DisNo", "ISO"], as_index=False)
            .agg(
                {
                    "PadStart": "first",
                    "PadEnd": "first",
                    "selected_groups": lambda values: ";".join(
                        sorted({str(value).strip() for value in values if str(value).strip() and str(value).lower() != "nan"})
                    ),
                    "match_grade": "first",
                }
            )
            .copy()
        )
        merged = summary_df.merge(agg, how="left", on=["DisNo", "ISO"], suffixes=("_summary", "_match"))
        if "match_grade_summary" in merged.columns:
            merged["match_grade"] = merged["match_grade_summary"].fillna(merged.get("match_grade_match"))
        elif "match_grade" not in merged.columns:
            merged["match_grade"] = ""

        for _, row in merged.iterrows():
            disno = str(row.get("DisNo", "")).strip().upper()
            iso = str(row.get("ISO", "")).strip().upper()
            rows.append(
                {
                    "year": int(year),
                    "DisNo": disno,
                    "disasterno_base": parse_disno_base(disno),
                    "ISO": iso,
                    "L_usd": float(pd.to_numeric(row.get("L_usd"), errors="coerce") or 0.0),
                    "status": str(row.get("status", "")).strip(),
                    "match_grade": str(row.get("match_grade", "")).strip(),
                    "PadStart": parse_date_safe(row.get("PadStart")),
                    "PadEnd": parse_date_safe(row.get("PadEnd")),
                    "selected_groups": str(row.get("selected_groups", "")).strip(),
                    "final_nc": str(paths["final"]),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates(subset=["year", "DisNo", "ISO"], keep="first").reset_index(drop=True)
    return out, missing


def extract_track_series(ds, selected_groups: Sequence[str], pad_start: date | None, pad_end: date | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name in selected_groups:
        grp = ds.groups.get(group_name)
        if grp is None or "time" not in grp.variables:
            continue
        tvals = np.asarray(grp.variables["time"][:], dtype=np.float64)
        tunits = str(grp.variables["time"].units)
        dts = num2date(tvals, units=tunits, calendar="standard")
        hazard_name = "hazard_compound_daily" if "hazard_compound_daily" in grp.variables else "hazard_compound"
        for idx, dt in enumerate(dts):
            current_date = date(int(dt.year), int(dt.month), int(dt.day))
            if pad_start and current_date < pad_start:
                continue
            if pad_end and current_date > pad_end:
                continue
            hazard = np.asarray(grp.variables[hazard_name][idx, :, :], dtype=float)
            peak = float(np.nanmax(hazard)) if np.any(np.isfinite(hazard)) else float("nan")
            rows.append(
                {
                    "time": current_date,
                    "center_lat": float(grp.variables["center_lat"][idx]),
                    "center_lon": float(grp.variables["center_lon"][idx]),
                    "hazard_peak": peak,
                    "group": group_name,
                    "time_idx": int(idx),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["time", "center_lat", "center_lon", "hazard_peak", "group", "time_idx"])
    return pd.DataFrame(rows).sort_values(["time", "group", "time_idx"]).reset_index(drop=True)


def overlap_ratio_quantile(a: np.ndarray, b: np.ndarray, q: float) -> float:
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    valid = np.isfinite(av) & np.isfinite(bv)
    if np.count_nonzero(valid) == 0:
        return float("nan")
    aa = av[valid]
    bb = bv[valid]
    ta = float(np.nanquantile(aa, max(0.0, min(1.0, 1.0 - q))))
    tb = float(np.nanquantile(bb, max(0.0, min(1.0, 1.0 - q))))
    ma = aa >= ta
    mb = bb >= tb
    union = int(np.count_nonzero(ma | mb))
    inter = int(np.count_nonzero(ma & mb))
    return float(inter / union) if union > 0 else float("nan")


def footprint_iou_recall(a: np.ndarray, b: np.ndarray, q: float) -> tuple[float, float]:
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    valid = np.isfinite(av) & np.isfinite(bv)
    if np.count_nonzero(valid) == 0:
        return float("nan"), float("nan")
    aa = av[valid]
    bb = bv[valid]
    ta = float(np.nanquantile(aa, max(0.0, min(1.0, 1.0 - q))))
    tb = float(np.nanquantile(bb, max(0.0, min(1.0, 1.0 - q))))
    ma = aa >= ta
    mb = bb >= tb
    union = int(np.count_nonzero(ma | mb))
    inter = int(np.count_nonzero(ma & mb))
    iou = float(inter / union) if union > 0 else float("nan")
    recall = float(inter / int(np.count_nonzero(ma))) if np.count_nonzero(ma) > 0 else float("nan")
    return iou, recall
