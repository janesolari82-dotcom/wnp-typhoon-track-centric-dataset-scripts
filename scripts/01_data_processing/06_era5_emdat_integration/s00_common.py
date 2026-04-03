#!/usr/bin/env python3
"""
Common helpers for EM-DAT integration scripts.
"""

from __future__ import annotations

import calendar
import logging
import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPRO_ROOT = PROJECT_ROOT / "reproduction_v6"

INPUT_NC_DIR = REPRO_ROOT / "03_intermediate_nc" / "pop_light_integration"
OUTPUT_ROOT = REPRO_ROOT / "04_final_output"
OUTPUT_NC_DIR = OUTPUT_ROOT / "emdat_attribution_integration"
OUTPUT_AUDIT_DIR = OUTPUT_NC_DIR / "audit"
LOG_DIR = REPRO_ROOT / "06_logs" / "emdat"

EMDAT_FILE = PROJECT_ROOT / "data" / "raw" / "using" / "disaster_records" / "emdat_typhoon_wnp_2000_2024.xlsx"
ERA5_ROOT = PROJECT_ROOT / "data" / "raw" / "using" / "era5"
GADM_GPKG = PROJECT_ROOT / "data" / "raw" / "using" / "gadm" / "gadm_410-levels.gpkg"
COUNTRY_RASTER = REPRO_ROOT / "02_processed_data" / "gadm_processed_v1" / "05_raster" / "country_id_30as_global.tif"
COUNTRY_LOOKUP = REPRO_ROOT / "02_processed_data" / "gadm_processed_v1" / "05_raster" / "country_id_lookup.csv"


def parse_years_expr(expr: str) -> List[int]:
    expr = expr.strip()
    if not expr:
        raise ValueError("years expression is empty")
    if "-" in expr:
        parts = expr.split("-")
        if len(parts) != 2:
            raise ValueError(f"invalid years expression: {expr}")
        y0 = int(parts[0])
        y1 = int(parts[1])
        if y0 > y1:
            raise ValueError(f"invalid years range: {expr}")
        return list(range(y0, y1 + 1))
    if "," in expr:
        years = sorted({int(x.strip()) for x in expr.split(",") if x.strip()})
        if not years:
            raise ValueError(f"invalid years expression: {expr}")
        return years
    return [int(expr)]


def years_to_expr(years: Sequence[int]) -> str:
    ys = sorted(set(int(y) for y in years))
    if not ys:
        return ""
    if len(ys) == 1:
        return str(ys[0])
    contiguous = all((ys[i] + 1 == ys[i + 1]) for i in range(len(ys) - 1))
    if contiguous:
        return f"{ys[0]}-{ys[-1]}"
    return ",".join(str(y) for y in ys)


def setup_logger(name: str, log_file: Path, level: str = "INFO") -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def normalize_text(text: str) -> str:
    s = "" if text is None else str(text)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.upper().strip()
    s = re.sub(r"\s+", " ", s)
    return s


STOPWORDS = {
    "TYPHOON",
    "TROPICAL",
    "STORM",
    "CYCLONE",
    "DEPRESSION",
    "SEVERE",
    "SUPER",
}


def name_tokens(text: str) -> List[str]:
    s = normalize_text(text)
    # Keep slash tokens because EM-DAT often stores aliases as "A/B".
    s = re.sub(r"[\"'()]", " ", s)
    parts = re.split(r"[^A-Z0-9]+", s)
    toks = []
    for p in parts:
        if not p:
            continue
        if p in STOPWORDS:
            continue
        if len(p) == 1 and not p.isdigit():
            continue
        toks.append(p)
    return toks


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb)) / float(len(sa | sb))


def end_of_month(year: int, month: int) -> int:
    return calendar.monthrange(int(year), int(month))[1]


def cfdate_to_date(x) -> date:
    return date(int(x.year), int(x.month), int(x.day))


def minmax_normalize_2d(arr):
    import numpy as np

    out = np.zeros_like(arr, dtype=np.float32)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return out
    v = arr[valid]
    vmin = float(v.min())
    vmax = float(v.max())
    if vmax <= vmin:
        return out
    out[valid] = ((arr[valid] - vmin) / (vmax - vmin)).astype("float32")
    return out
