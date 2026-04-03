#!/usr/bin/env python3
"""Population Log-PCHIP reconstruction pipeline (GPWv4.11, 2000-2024)."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
import yaml
from rasterio.windows import Window
from scipy.interpolate import PchipInterpolator


STEP_ORDER = [
    "inventory",
    "save_plan",
    "log_pchip_reconstruct",
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


class BlockMemoryError(MemoryError):
    """Memory error with block location context."""

    def __init__(self, row_off: int, col_off: int, block_rows: int, block_cols: int) -> None:
        super().__init__(
            f"MemoryError at block row_off={row_off}, col_off={col_off}, "
            f"block_rows={block_rows}, block_cols={block_cols}"
        )
        self.row_off = row_off
        self.col_off = col_off
        self.block_rows = block_rows
        self.block_cols = block_cols


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Population Log-PCHIP reconstruction")
    parser.add_argument(
        "--config",
        type=str,
        default="reproduction_v6/config/population_log_pchip.yaml",
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
    logger = logging.getLogger("population_log_pchip")
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


def write_csv(path: Path, rows: Sequence[Dict], headers: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(headers))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def year_range(start: int, end: int) -> List[int]:
    return list(range(int(start), int(end) + 1))


def resolve_input_files(cfg: Dict) -> Dict[int, Path]:
    input_root = resolve_path(cfg, "input_root")
    pattern = str(cfg["inputs"]["file_pattern"])
    base_years = [int(y) for y in cfg["inputs"]["base_years"]]
    out: Dict[int, Path] = {}
    for year in base_years:
        out[year] = input_root / pattern.format(year=year)
    return out


def to_crs_string(crs_obj) -> str:
    if crs_obj is None:
        return ""
    return str(crs_obj)


def array_window_iter(height: int, width: int, block_rows: int, block_cols: int) -> Iterable[Window]:
    for row_off in range(0, height, block_rows):
        h = min(block_rows, height - row_off)
        for col_off in range(0, width, block_cols):
            w = min(block_cols, width - col_off)
            yield Window(col_off=col_off, row_off=row_off, width=w, height=h)


def build_anomaly(level: str, scope: str, year: str, issue: str, detail: str) -> Dict:
    return {"level": level, "scope": scope, "year": year, "issue": issue, "detail": detail}


def read_raster_meta(path: Path) -> Dict:
    with rasterio.open(path) as ds:
        t = ds.transform
        return {
            "path": str(path),
            "width": int(ds.width),
            "height": int(ds.height),
            "crs": to_crs_string(ds.crs),
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


def almost_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def validate_metas(meta_by_year: Dict[int, Dict], grid: GridSpec, anomalies: List[Dict]) -> None:
    years = sorted(meta_by_year.keys())
    if not years:
        return

    ref_year = years[0]
    ref = meta_by_year[ref_year]

    for year in years[1:]:
        meta = meta_by_year[year]
        if meta["crs"] != ref["crs"]:
            anomalies.append(build_anomaly("error", "metadata", str(year), "crs_mismatch", f"{meta['crs']} vs {ref['crs']}"))
        if meta["width"] != ref["width"] or meta["height"] != ref["height"]:
            anomalies.append(
                build_anomaly(
                    "error",
                    "metadata",
                    str(year),
                    "shape_mismatch",
                    f"{meta['width']}x{meta['height']} vs {ref['width']}x{ref['height']}",
                )
            )
        if meta["nodata"] != ref["nodata"]:
            anomalies.append(
                build_anomaly(
                    "error",
                    "metadata",
                    str(year),
                    "nodata_mismatch",
                    f"{meta['nodata']} vs {ref['nodata']}",
                )
            )
        for i, (v1, v2) in enumerate(zip(meta["transform"], ref["transform"])):
            if not almost_equal(v1, v2, tol=1e-12):
                anomalies.append(
                    build_anomaly(
                        "error",
                        "metadata",
                        str(year),
                        "transform_mismatch",
                        f"idx={i}, {v1} vs {v2}",
                    )
                )
                break

    ref_bounds = ref["bounds"]
    if ref["crs"] != grid.crs:
        anomalies.append(build_anomaly("error", "grid", str(ref_year), "unexpected_crs", ref["crs"]))
    if ref["width"] != grid.width or ref["height"] != grid.height:
        anomalies.append(
            build_anomaly(
                "error",
                "grid",
                str(ref_year),
                "unexpected_shape",
                f"{ref['width']}x{ref['height']} expected {grid.width}x{grid.height}",
            )
        )
    if not almost_equal(ref["res_x"], grid.resolution, tol=1e-12) or not almost_equal(ref["res_y"], grid.resolution, tol=1e-12):
        anomalies.append(
            build_anomaly(
                "error",
                "grid",
                str(ref_year),
                "unexpected_resolution",
                f"{ref['res_x']},{ref['res_y']} expected {grid.resolution}",
            )
        )
    if not almost_equal(ref_bounds["left"], grid.lon_min, tol=1e-9):
        anomalies.append(
            build_anomaly("error", "grid", str(ref_year), "unexpected_left", f"{ref_bounds['left']} expected {grid.lon_min}")
        )
    if not almost_equal(ref_bounds["right"], grid.lon_max, tol=1e-9):
        anomalies.append(
            build_anomaly("error", "grid", str(ref_year), "unexpected_right", f"{ref_bounds['right']} expected {grid.lon_max}")
        )
    if not almost_equal(ref_bounds["bottom"], grid.lat_min, tol=1e-9):
        anomalies.append(
            build_anomaly("error", "grid", str(ref_year), "unexpected_bottom", f"{ref_bounds['bottom']} expected {grid.lat_min}")
        )
    if not almost_equal(ref_bounds["top"], grid.lat_max, tol=1e-9):
        anomalies.append(
            build_anomaly("error", "grid", str(ref_year), "unexpected_top", f"{ref_bounds['top']} expected {grid.lat_max}")
        )


def collect_inventory(cfg: Dict, logger: logging.Logger) -> Tuple[Path, Path, Dict[int, Path], Dict[int, Dict], List[Dict]]:
    output_root = resolve_path(cfg, "output_root")
    inv_dir = output_root / "00_inventory"
    ensure_dir(inv_dir)

    input_files = resolve_input_files(cfg)
    anomalies: List[Dict] = []
    meta_by_year: Dict[int, Dict] = {}
    base_years = [int(y) for y in cfg["inputs"]["base_years"]]

    for year in base_years:
        p = input_files[year]
        if not p.exists():
            anomalies.append(build_anomaly("error", "input", str(year), "missing_input_file", str(p)))
            continue
        try:
            meta_by_year[year] = read_raster_meta(p)
        except Exception as exc:
            anomalies.append(build_anomaly("error", "input", str(year), "unreadable_input_file", str(exc)))

    validate_metas(meta_by_year, grid_spec(cfg), anomalies)

    payload = {
        "created_at": datetime.now().isoformat(),
        "config_path": cfg.get("_config_path", ""),
        "reference_doc": str(resolve_path(cfg, "reference_doc")),
        "methodology_doc": str(resolve_path(cfg, "methodology_doc")),
        "base_years": base_years,
        "input_files": {str(y): str(input_files[y]) for y in base_years},
        "meta_by_year": {str(y): meta_by_year[y] for y in sorted(meta_by_year.keys())},
        "target_grid": {
            "crs": cfg["grid"]["crs"],
            "resolution_deg": cfg["grid"]["resolution_deg"],
            "lon_min": cfg["grid"]["lon_min"],
            "lon_max": cfg["grid"]["lon_max"],
            "lat_min": cfg["grid"]["lat_min"],
            "lat_max": cfg["grid"]["lat_max"],
            "width": cfg["grid"]["width"],
            "height": cfg["grid"]["height"],
            "nodata": cfg["grid"]["nodata"],
        },
        "anomaly_count": len(anomalies),
        "error_count": sum(1 for a in anomalies if a["level"] == "error"),
    }

    source_inventory_path = inv_dir / "source_inventory.json"
    anomalies_path = inv_dir / "input_anomalies.csv"
    write_json(source_inventory_path, payload)
    write_csv(anomalies_path, anomalies, headers=["level", "scope", "year", "issue", "detail"])

    logger.info("Inventory written: %s", source_inventory_path)
    logger.info("Anomalies written: %s (%d rows)", anomalies_path, len(anomalies))
    return source_inventory_path, anomalies_path, input_files, meta_by_year, anomalies


def inventory_inputs(cfg: Dict, logger: logging.Logger) -> Tuple[Path, Path]:
    logger.info("Step 1/4: inventory and consistency checks")
    source_inventory_path, anomalies_path, _, _, anomalies = collect_inventory(cfg, logger)
    error_count = sum(1 for a in anomalies if a["level"] == "error")
    if error_count > 0:
        raise RuntimeError(f"Inventory failed with {error_count} error(s). See {anomalies_path}")
    return source_inventory_path, anomalies_path

def build_plan_markdown(cfg: Dict) -> str:
    base_years = cfg["inputs"]["base_years"]
    ystart = int(cfg["years"]["target_start"])
    yend = int(cfg["years"]["target_end"])
    return f"""# Population Log-PCHIP Execution Plan

## Summary
- Scope: GPWv4.11 global full extent reconstruction.
- Method: `log1p -> PchipInterpolator(extrapolate=True) -> expm1`.
- Target years: {ystart}-{yend}.
- Base years: {base_years}.
- WPP constraint: disabled by design in this stage.

## Paths
- Input root: `{resolve_path(cfg, "input_root")}`
- Output root: `{resolve_path(cfg, "output_root")}`
- Report root: `{resolve_path(cfg, "report_root")}`
- Reference doc: `{resolve_path(cfg, "reference_doc")}`
- Methodology doc: `{resolve_path(cfg, "methodology_doc")}`

## Steps
1. `inventory`: validate file existence/readability and metadata consistency.
2. `save_plan`: write this plan.
3. `log_pchip_reconstruct`: block-wise yearly reconstruction for 2000-2024.
4. `quality_report`: write QC summary, knot fidelity, extrapolation stability report.

## Output Artifacts
- `00_inventory/source_inventory.json`
- `00_inventory/input_anomalies.csv`
- `00_inventory/population_log_pchip_run.log`
- `01_population_annual/population_{{year}}_30as.tif`
- `01_population_annual/population_manifest.json`
- `01_population_annual/population_yearly_summary.csv`
- `reproduction_v6/07_reproduction_report/population/population_qc_metrics.csv`
- `reproduction_v6/07_reproduction_report/population/population_knot_fidelity.csv`
- `reproduction_v6/07_reproduction_report/population/population_log_pchip_report.md`
"""


def save_plan_file(cfg: Dict, logger: logging.Logger) -> Path:
    logger.info("Step 2/4: write plan markdown")
    plan_output = resolve_path(cfg, "plan_output")
    ensure_dir(plan_output.parent)
    plan_output.write_text(build_plan_markdown(cfg), encoding="utf-8")
    logger.info("Plan written: %s", plan_output)
    return plan_output


def sanitize_population(arr: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    out = arr.astype(np.float64, copy=False)
    if nodata is not None:
        out = np.where(out == nodata, 0.0, out)
    out = np.where(np.isfinite(out), out, 0.0)
    out = np.where(out < 0.0, 0.0, out)
    return out


def init_stats(years: Sequence[int]) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for y in years:
        out[int(y)] = {
            "min": float("inf"),
            "max": float("-inf"),
            "sum": 0.0,
            "sumsq": 0.0,
            "count": 0.0,
            "zero_count": 0.0,
            "finite_issue_count": 0.0,
        }
    return out


def update_stats(st: Dict[str, float], arr: np.ndarray) -> None:
    vals = arr.astype(np.float64, copy=False)
    finite = np.isfinite(vals)
    if not np.all(finite):
        st["finite_issue_count"] += float(vals.size - np.count_nonzero(finite))
    vals = vals[finite]
    if vals.size == 0:
        return

    vmin = float(vals.min())
    vmax = float(vals.max())
    st["min"] = min(st["min"], vmin)
    st["max"] = max(st["max"], vmax)
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
        "count": count,
        "finite_issue_count": int(st["finite_issue_count"]),
    }


def raster_stats(path: Path, nodata: Optional[float]) -> Dict:
    with rasterio.open(path) as ds:
        st = init_stats([0])[0]
        for _, window in ds.block_windows(1):
            arr = ds.read(1, window=window)
            vals = arr.astype(np.float64, copy=False)
            finite = np.isfinite(vals)
            if nodata is not None:
                finite &= vals != nodata
            if not np.any(finite):
                continue
            vv = vals[finite]
            st["min"] = min(st["min"], float(vv.min()))
            st["max"] = max(st["max"], float(vv.max()))
            st["sum"] += float(vv.sum())
            st["sumsq"] += float(np.square(vv).sum())
            st["count"] += float(vv.size)
            st["zero_count"] += float(np.count_nonzero(vv <= 0.0))
            st["finite_issue_count"] += float(vals.size - np.count_nonzero(np.isfinite(vals)))
    return finalize_stats(0, st)


def _profile_for_output(reference_ds: rasterio.io.DatasetReader, cfg: Dict) -> Dict:
    runtime = cfg["runtime"]
    grid = grid_spec(cfg)
    profile = reference_ds.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype=grid.dtype,
        nodata=grid.nodata,
        compress=str(runtime["compression"]),
        tiled=bool(runtime["tiled"]),
        blockxsize=int(runtime["blockxsize"]),
        blockysize=int(runtime["blockysize"]),
        bigtiff=str(runtime["bigtiff"]),
    )
    return profile


def _attempt_reconstruction(
    cfg: Dict,
    input_files: Dict[int, Path],
    years_write: List[int],
    tmp_output_files: Dict[int, Path],
    block_rows: int,
    block_cols: int,
) -> Dict[int, Dict]:
    base_years = [int(y) for y in cfg["inputs"]["base_years"]]
    target_stats = init_stats(years_write)
    current_window = Window(col_off=0, row_off=0, width=0, height=0)

    with ExitStack() as stack:
        input_ds = {year: stack.enter_context(rasterio.open(input_files[year])) for year in base_years}
        profile = _profile_for_output(input_ds[base_years[0]], cfg)
        output_ds = {year: stack.enter_context(rasterio.open(tmp_output_files[year], "w", **profile)) for year in years_write}

        height = input_ds[base_years[0]].height
        width = input_ds[base_years[0]].width
        nodata_map = {year: input_ds[year].nodata for year in base_years}

        try:
            for current_window in array_window_iter(height, width, block_rows=block_rows, block_cols=block_cols):
                cube = np.empty((len(base_years), int(current_window.height), int(current_window.width)), dtype=np.float64)
                for i, year in enumerate(base_years):
                    src = input_ds[year].read(1, window=current_window)
                    cube[i] = sanitize_population(src, nodata_map[year])

                log_cube = np.log1p(cube)
                interp = PchipInterpolator(base_years, log_cube, axis=0, extrapolate=True)
                log_pred = interp(years_write)
                pop_pred = np.expm1(log_pred)
                pop_pred = np.where(np.isfinite(pop_pred), pop_pred, 0.0)
                pop_pred = np.clip(pop_pred, 0.0, None)

                for i, year in enumerate(years_write):
                    arr = pop_pred[i].astype(np.float32, copy=False)
                    output_ds[year].write(arr, 1, window=current_window)
                    update_stats(target_stats[year], arr)
        except MemoryError as exc:
            raise BlockMemoryError(
                row_off=int(current_window.row_off),
                col_off=int(current_window.col_off),
                block_rows=block_rows,
                block_cols=block_cols,
            ) from exc

    return {year: finalize_stats(year, target_stats[year]) for year in years_write}


def _block_size_candidates(block_rows: int, block_cols: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    seen = set()
    r, c = int(block_rows), int(block_cols)
    while True:
        key = (r, c)
        if key not in seen:
            out.append(key)
            seen.add(key)
        if r <= 32 and c <= 32:
            break
        r = max(32, r // 2)
        c = max(32, c // 2)
        if (r, c) in seen:
            break
    return out


def step_log_pchip_reconstruct(cfg: Dict, logger: logging.Logger, overwrite: bool) -> Tuple[Path, Path]:
    logger.info("Step 3/4: block-wise Log-PCHIP reconstruction")
    _, _, input_files, _, anomalies = collect_inventory(cfg, logger)
    error_count = sum(1 for a in anomalies if a["level"] == "error")
    if error_count > 0:
        raise RuntimeError(f"Cannot reconstruct due to inventory errors ({error_count}).")

    out_root = resolve_path(cfg, "output_root") / "01_population_annual"
    ensure_dir(out_root)

    ycfg = cfg["years"]
    target_years = year_range(int(ycfg["target_start"]), int(ycfg["target_end"]))
    runtime = cfg["runtime"]
    block_rows = int(runtime["block_rows"])
    block_cols = int(runtime["block_cols"])
    nodata = float(cfg["grid"]["nodata"])

    output_files = {year: out_root / f"population_{year}_30as.tif" for year in target_years}
    years_skip_existing: List[int] = []
    years_write: List[int] = []
    for year in target_years:
        if output_files[year].exists() and not overwrite:
            years_skip_existing.append(year)
            logger.info("Skip existing output (overwrite=false): %s", output_files[year].name)
        else:
            years_write.append(year)

    stats_by_year: Dict[int, Dict] = {}
    generated_years: List[int] = []
    skipped_years: List[int] = list(years_skip_existing)
    used_block_size: Optional[Tuple[int, int]] = None

    if years_write:
        candidates = _block_size_candidates(block_rows, block_cols)
        success = False
        last_error: Optional[Exception] = None

        for cand_rows, cand_cols in candidates:
            logger.info("Reconstruction attempt with block size rows=%d, cols=%d", cand_rows, cand_cols)
            tmp_output_files = {year: output_files[year].with_suffix(".tif.tmp") for year in years_write}

            for year in years_write:
                tmp = tmp_output_files[year]
                if tmp.exists():
                    tmp.unlink()

            try:
                stats_partial = _attempt_reconstruction(
                    cfg=cfg,
                    input_files=input_files,
                    years_write=years_write,
                    tmp_output_files=tmp_output_files,
                    block_rows=cand_rows,
                    block_cols=cand_cols,
                )
                for year in years_write:
                    tmp_output_files[year].replace(output_files[year])
                    generated_years.append(year)
                stats_by_year.update(stats_partial)
                success = True
                used_block_size = (cand_rows, cand_cols)
                break
            except BlockMemoryError as exc:
                last_error = exc
                logger.warning("%s", exc)
                logger.warning("Retrying with smaller block size.")
                for year in years_write:
                    tmp = tmp_output_files[year]
                    if tmp.exists():
                        tmp.unlink()
                continue
            except Exception as exc:
                last_error = exc
                for year in years_write:
                    tmp = tmp_output_files[year]
                    if tmp.exists():
                        tmp.unlink()
                raise

        if not success:
            if last_error is not None:
                raise RuntimeError(f"All block-size attempts failed: {last_error}") from last_error
            raise RuntimeError("All block-size attempts failed for unknown reason.")

    for year in target_years:
        if year not in stats_by_year:
            st = raster_stats(output_files[year], nodata=nodata)
            st["year"] = year
            stats_by_year[year] = st

    summary_rows = [stats_by_year[year] for year in target_years]
    summary_csv = out_root / "population_yearly_summary.csv"
    write_csv(
        summary_csv,
        summary_rows,
        headers=["year", "min", "max", "mean", "std", "sum", "zero_frac", "count", "finite_issue_count"],
    )

    manifest_payload = {
        "created_at": datetime.now().isoformat(),
        "config_path": cfg.get("_config_path", ""),
        "algorithm": "log1p + PchipInterpolator(extrapolate=True) + expm1",
        "base_years": [int(y) for y in cfg["inputs"]["base_years"]],
        "target_years": target_years,
        "input_files": {str(y): str(input_files[y]) for y in sorted(input_files.keys())},
        "output_files": {str(y): str(output_files[y]) for y in target_years},
        "generated_years": sorted(generated_years),
        "skipped_existing_years": sorted(skipped_years),
        "overwrite": bool(overwrite),
        "runtime_block_size_requested": {"rows": block_rows, "cols": block_cols},
        "runtime_block_size_used": None if used_block_size is None else {"rows": used_block_size[0], "cols": used_block_size[1]},
        "summary_csv": str(summary_csv),
    }
    manifest_json = out_root / "population_manifest.json"
    write_json(manifest_json, manifest_payload)

    logger.info("Summary written: %s", summary_csv)
    logger.info("Manifest written: %s", manifest_json)
    return manifest_json, summary_csv

def load_yearly_summary(summary_csv: Path) -> Dict[int, Dict]:
    out: Dict[int, Dict] = {}
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            year = int(row["year"])
            out[year] = {
                "year": year,
                "min": float(row["min"]),
                "max": float(row["max"]),
                "mean": float(row["mean"]),
                "std": float(row["std"]),
                "sum": float(row["sum"]),
                "zero_frac": float(row["zero_frac"]),
                "count": int(float(row["count"])),
                "finite_issue_count": int(float(row.get("finite_issue_count", 0))),
            }
    return out


def compute_knot_fidelity(
    cfg: Dict,
    logger: logging.Logger,
    input_files: Dict[int, Path],
    output_dir: Path,
) -> List[Dict]:
    knot_years = [int(y) for y in cfg["inputs"]["base_years"]]
    runtime = cfg["runtime"]
    block_rows = int(runtime["block_rows"])
    block_cols = int(runtime["block_cols"])
    rows: List[Dict] = []

    for year in knot_years:
        in_path = input_files[year]
        out_path = output_dir / f"population_{year}_30as.tif"
        if not out_path.exists():
            raise FileNotFoundError(f"Missing reconstructed knot year output: {out_path}")

        mae_num = 0.0
        max_abs = 0.0
        count = 0
        with rasterio.open(in_path) as src, rasterio.open(out_path) as pred:
            for window in array_window_iter(src.height, src.width, block_rows=block_rows, block_cols=block_cols):
                a = sanitize_population(src.read(1, window=window), src.nodata)
                b = pred.read(1, window=window).astype(np.float64, copy=False)
                b = np.where(np.isfinite(b), b, 0.0)
                b = np.where(b < 0.0, 0.0, b)
                diff = np.abs(b - a)
                mae_num += float(diff.sum())
                max_abs = max(max_abs, float(diff.max()))
                count += int(diff.size)

        mae = mae_num / max(count, 1)
        row = {"year": year, "mae": mae, "max_abs_error": max_abs, "count": count}
        logger.info("Knot fidelity %d: MAE=%.6f, MaxAbs=%.6f", year, mae, max_abs)
        rows.append(row)

    return rows


def step_quality_report(cfg: Dict, logger: logging.Logger) -> Tuple[Path, Path, Path]:
    logger.info("Step 4/4: quality metrics and report")
    out_dir = resolve_path(cfg, "output_root") / "01_population_annual"
    summary_csv = out_dir / "population_yearly_summary.csv"
    manifest_json = out_dir / "population_manifest.json"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing yearly summary: {summary_csv}")
    if not manifest_json.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_json}")

    report_root = resolve_path(cfg, "report_root")
    ensure_dir(report_root)
    summary = load_yearly_summary(summary_csv)
    target_years = sorted(summary.keys())

    qc_rows: List[Dict] = []
    for year in target_years:
        row = summary[year]
        qc_rows.append(
            {
                "year": year,
                "min": row["min"],
                "max": row["max"],
                "mean": row["mean"],
                "std": row["std"],
                "sum": row["sum"],
                "zero_frac": row["zero_frac"],
                "count": row["count"],
                "finite_issue_count": row["finite_issue_count"],
                "nonnegative_pass": bool(row["min"] >= -1e-9),
                "finite_pass": bool(row["finite_issue_count"] == 0),
            }
        )

    qc_csv = report_root / "population_qc_metrics.csv"
    write_csv(
        qc_csv,
        qc_rows,
        headers=[
            "year",
            "min",
            "max",
            "mean",
            "std",
            "sum",
            "zero_frac",
            "count",
            "finite_issue_count",
            "nonnegative_pass",
            "finite_pass",
        ],
    )

    input_files = resolve_input_files(cfg)
    knot_rows = compute_knot_fidelity(cfg, logger, input_files=input_files, output_dir=out_dir)
    knot_csv = report_root / "population_knot_fidelity.csv"
    write_csv(knot_csv, knot_rows, headers=["year", "mae", "max_abs_error", "count"])

    sum2020 = summary[2020]["sum"] if 2020 in summary else float("nan")
    sum2024 = summary[2024]["sum"] if 2024 in summary else float("nan")
    growth_2020_2024 = (sum2024 - sum2020) / max(sum2020, 1e-12)
    nonnegative_2021_2024 = all(summary[y]["min"] >= -1e-9 for y in [2021, 2022, 2023, 2024] if y in summary)
    finite_2021_2024 = all(summary[y]["finite_issue_count"] == 0 for y in [2021, 2022, 2023, 2024] if y in summary)

    report_md = report_root / "population_log_pchip_report.md"
    report_text = "\n".join(
        [
            "# Population Log-PCHIP Report",
            "",
            f"- created_at: {datetime.now().isoformat()}",
            f"- years_covered: {min(target_years)}-{max(target_years)}" if target_years else "- years_covered: n/a",
            f"- global_sum_2020: {sum2020}",
            f"- global_sum_2024: {sum2024}",
            f"- global_sum_growth_2020_2024: {growth_2020_2024}",
            f"- nonnegative_pass_2021_2024: {nonnegative_2021_2024}",
            f"- finite_pass_2021_2024: {finite_2021_2024}",
            "",
            "## Files",
            f"- yearly_summary: {summary_csv}",
            f"- manifest: {manifest_json}",
            f"- qc_metrics: {qc_csv}",
            f"- knot_fidelity: {knot_csv}",
            "",
            "## Knot Fidelity",
            *[
                f"- {row['year']}: MAE={row['mae']}, MaxAbsError={row['max_abs_error']}"
                for row in sorted(knot_rows, key=lambda x: int(x["year"]))
            ],
        ]
    )
    report_md.write_text(report_text, encoding="utf-8")

    logger.info("QC metrics: %s", qc_csv)
    logger.info("Knot fidelity: %s", knot_csv)
    logger.info("Report markdown: %s", report_md)
    return qc_csv, knot_csv, report_md


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = project_root() / cfg_path

    cfg = load_config(cfg_path)
    cfg["runtime"]["overwrite"] = bool(args.overwrite or cfg["runtime"].get("overwrite", False))
    overwrite = bool(cfg["runtime"]["overwrite"])

    log_file = resolve_path(cfg, "output_root") / "00_inventory" / "population_log_pchip_run.log"
    logger = setup_logger(log_file)
    logger.info("=" * 72)
    logger.info("Population Log-PCHIP pipeline started")
    logger.info("Config: %s", cfg_path)
    logger.info("Overwrite: %s", overwrite)
    logger.info("=" * 72)

    steps = parse_steps(args.steps)
    logger.info("Steps: %s", steps)

    for step in steps:
        if step == "inventory":
            inventory_inputs(cfg, logger)
        elif step == "save_plan":
            save_plan_file(cfg, logger)
        elif step == "log_pchip_reconstruct":
            step_log_pchip_reconstruct(cfg, logger, overwrite=overwrite)
        elif step == "quality_report":
            step_quality_report(cfg, logger)

    logger.info("=" * 72)
    logger.info("Population Log-PCHIP pipeline completed")
    logger.info("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
