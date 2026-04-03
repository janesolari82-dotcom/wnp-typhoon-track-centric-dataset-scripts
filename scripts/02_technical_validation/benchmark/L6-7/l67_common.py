from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
import numpy as np
import pandas as pd
from netCDF4 import num2date
from shapely.geometry import LineString, Point

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXTERNAL_YEAR_START = 2000
EXTERNAL_YEAR_END = 2018


def split_external_years(years: Sequence[int]) -> tuple[list[int], list[int]]:
    eval_years = [int(y) for y in years if EXTERNAL_YEAR_START <= int(y) <= EXTERNAL_YEAR_END]
    not_eval_years = [int(y) for y in years if int(y) < EXTERNAL_YEAR_START or int(y) > EXTERNAL_YEAR_END]
    return sorted(eval_years), sorted(not_eval_years)


def parse_disno_base(disno: str) -> str:
    s = str(disno or "").strip().upper()
    m = re.search(r"(\d{4}-\d{4})", s)
    return m.group(1) if m else ""


def parse_date_safe(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    try:
        return pd.to_datetime(s, errors="coerce").date()
    except Exception:
        return None


def overlap_days(a0: date | None, a1: date | None, b0: date | None, b1: date | None) -> int:
    if a0 is None or a1 is None or b0 is None or b1 is None:
        return 0
    lo = max(a0, b0)
    hi = min(a1, b1)
    if hi < lo:
        return 0
    return int((hi - lo).days) + 1


def normalize_cause(text: str) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_iso_set(text: str) -> set[str]:
    # GFD "cc" commonly stores ISO3 codes, possibly comma/semicolon separated.
    s = str(text or "").upper()
    tokens = re.findall(r"[A-Z]{3}", s)
    return {t for t in tokens if t}


def split_groups(value: str) -> list[str]:
    s = str(value or "")
    raw = re.split(r"[;,]", s)
    out = [x.strip() for x in raw if x.strip()]
    return out


def haversine_km(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    r = 6371.0088
    lon1r = np.radians(lon1)
    lat1r = np.radians(lat1)
    lon2r = np.radians(lon2)
    lat2r = np.radians(lat2)
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c


def event_points_from_group(
    grp,
    idx: int,
    hazard_q: float = 0.9,
) -> pd.DataFrame:
    var_names = set(grp.variables.keys())
    hazard_name = "hazard_compound_daily" if "hazard_compound_daily" in var_names else "hazard_compound"
    rain_name = "hazard_rain_daily" if "hazard_rain_daily" in var_names else "hazard_rain"
    loss_name = "emdat_loss_allocated_usd" if "emdat_loss_allocated_usd" in var_names else None

    center_lat = float(grp.variables["center_lat"][idx])
    center_lon = float(grp.variables["center_lon"][idx])
    wlat = np.asarray(grp.variables["window_lat"][:], dtype=np.float32)
    wlon = np.asarray(grp.variables["window_lon"][:], dtype=np.float32)
    hazard = np.asarray(grp.variables[hazard_name][idx, :, :], dtype=np.float32)
    rain = np.asarray(grp.variables[rain_name][idx, :, :], dtype=np.float32)
    if loss_name is not None:
        loss = np.asarray(grp.variables[loss_name][idx, :, :], dtype=np.float32)
    else:
        loss = np.zeros_like(hazard, dtype=np.float32)
    lat_abs = center_lat + np.broadcast_to(wlat[:, None], hazard.shape)
    lon_abs = center_lon + np.broadcast_to(wlon[None, :], hazard.shape)

    finite = np.isfinite(hazard)
    if np.any(finite):
        qv = float(np.nanquantile(hazard[finite], hazard_q))
        mask = finite & ((hazard >= qv) | (loss > 0))
    else:
        mask = loss > 0

    if not np.any(mask):
        # Fallback: at least keep one representative point.
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
    for g in selected_groups:
        grp = ds.groups.get(g)
        if grp is None or "time" not in grp.variables:
            continue
        tvals = np.asarray(grp.variables["time"][:], dtype=np.float64)
        tunits = str(grp.variables["time"].units)
        dts = num2date(tvals, units=tunits, calendar="standard")
        for i, d in enumerate(dts):
            di = date(int(d.year), int(d.month), int(d.day))
            if pad_start and di < pad_start:
                continue
            if pad_end and di > pad_end:
                continue
            pdf = event_points_from_group(grp, i)
            if pdf.empty:
                continue
            pdf["time"] = di
            pdf["group"] = g
            rows.append(pdf)

    if not rows:
        return pd.DataFrame(columns=["time", "lat", "lon", "hazard_compound", "hazard_rain", "loss_alloc", "group"])
    out = pd.concat(rows, ignore_index=True)
    if len(out) > max_points:
        out = out.sample(n=max_points, random_state=int(seed))
    return out.reset_index(drop=True)


def build_track_geometry(points: pd.DataFrame):
    if points.empty:
        return None
    pts = points[["lon", "lat"]].dropna()
    if pts.empty:
        return None
    coords = [tuple(x) for x in pts.to_numpy(dtype=float)]
    uniq: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for xy in coords:
        if xy not in seen:
            seen.add(xy)
            uniq.append(xy)
    if len(uniq) == 1:
        return Point(uniq[0])
    return LineString(uniq)


def grade_priority(match_grade: str) -> int:
    g = str(match_grade or "").upper().strip()
    if g == "A":
        return 0
    if g == "B":
        return 1
    if g == "C":
        return 2
    if g == "D":
        return 3
    return 4


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_plot_data(df: pd.DataFrame, path: Path, write_csv_fn, cfg) -> None:
    write_csv_fn(df, path, cfg)


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def finalize_stage_status(has_fail: bool, has_warn: bool) -> str:
    if has_fail:
        return "FAIL"
    if has_warn:
        return "WARN"
    return "PASS"


def safe_ratio(num: float, den: float) -> float:
    if den is None or den == 0:
        return float("nan")
    return float(num) / float(den)
