from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from netCDF4 import Dataset

DEFAULT_YEAR_EXPR = "2000-2024"
DEFAULT_SEED = 20260304
DEFAULT_L3_SAMPLES_PER_YEAR = 300
DEFAULT_WORKERS = 1

LEVELS = ["L0", "L1", "L2", "L3", "L4", "L5"]
STATUS_RANK = {"PASS": 0, "WARN": 1, "FAIL": 2}


@dataclass
class StageResult:
    level: str
    status: str
    fail_count: int = 0
    warn_count: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunConfig:
    project_root: Path
    years: list[int]
    results_root: Path
    l3_samples_per_year: int
    seed: int
    workers: int
    strict: bool
    read_roots: list[Path]
    write_root: Path

    @property
    def reproduction_root(self) -> Path:
        return self.project_root / "reproduction_v6"

    @property
    def raw_using_root(self) -> Path:
        return self.project_root / "data" / "raw" / "using"

    @property
    def processed_root(self) -> Path:
        return self.reproduction_root / "02_processed_data"


def infer_project_root() -> Path:
    # all_scripts/02_technical_validation/benchmark/*.py -> benchmark / 02_technical_validation / all_scripts / reproduction_v6 / project
    return Path(__file__).resolve().parents[4]


def parse_years_expr(expr: str) -> list[int]:
    text = (expr or "").strip()
    if not text:
        raise ValueError("years expression is empty")
    if "-" in text:
        parts = text.split("-")
        if len(parts) != 2:
            raise ValueError(f"invalid years expression: {text}")
        start = int(parts[0])
        end = int(parts[1])
        if start > end:
            raise ValueError(f"invalid years range: {text}")
        return list(range(start, end + 1))
    if "," in text:
        years = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
        if not years:
            raise ValueError(f"invalid years expression: {text}")
        return years
    return [int(text)]


def years_to_expr(years: Sequence[int]) -> str:
    ys = sorted(set(int(y) for y in years))
    if not ys:
        return ""
    if len(ys) == 1:
        return str(ys[0])
    contiguous = all(ys[i] + 1 == ys[i + 1] for i in range(len(ys) - 1))
    if contiguous:
        return f"{ys[0]}-{ys[-1]}"
    return ",".join(str(y) for y in ys)


def make_shared_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--project-root", type=Path, default=infer_project_root())
    p.add_argument("--years", type=str, default=DEFAULT_YEAR_EXPR)
    p.add_argument("--results-root", type=Path, default=None)
    p.add_argument("--l3-samples-per-year", type=int, default=DEFAULT_L3_SAMPLES_PER_YEAR)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--strict", action="store_true")
    return p


def build_run_config(args: argparse.Namespace) -> RunConfig:
    project_root = Path(args.project_root).resolve()
    years = parse_years_expr(args.years)
    if not years:
        raise ValueError("no years selected")
    if getattr(args, "results_root", None) is None:
        results_root = (project_root / "reproduction_v6" / "05_benchmark_results" / "results").resolve()
    else:
        results_root = Path(args.results_root).resolve()

    read_roots = [
        (project_root / "reproduction_v6" / "03_intermediate_nc").resolve(),
        (project_root / "reproduction_v6" / "04_final_output").resolve(),
        (project_root / "reproduction_v6" / "02_processed_data").resolve(),
        (project_root / "data" / "raw" / "using").resolve(),
    ]

    return RunConfig(
        project_root=project_root,
        years=years,
        results_root=results_root,
        l3_samples_per_year=max(int(args.l3_samples_per_year), 1),
        seed=int(args.seed),
        workers=max(int(args.workers), 1),
        strict=bool(args.strict),
        read_roots=read_roots,
        write_root=results_root,
    )


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def assert_read_allowed(path: Path, read_roots: Sequence[Path]) -> None:
    rp = path.resolve()
    if not any(is_within(rp, rr) for rr in read_roots):
        roots = ", ".join(str(rr) for rr in read_roots)
        raise PermissionError(f"read path outside whitelist: {rp} | roots={roots}")


def assert_write_allowed(path: Path, write_root: Path) -> None:
    wp = path.resolve()
    if not is_within(wp, write_root):
        raise PermissionError(f"write path outside whitelist: {wp} | root={write_root}")


def open_netcdf_readonly(path: Path, cfg: RunConfig) -> Dataset:
    assert_read_allowed(path, cfg.read_roots)
    return Dataset(path, mode="r")


def ensure_dir(path: Path, cfg: RunConfig) -> Path:
    assert_write_allowed(path, cfg.write_root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(df: pd.DataFrame, path: Path, cfg: RunConfig) -> None:
    assert_write_allowed(path, cfg.write_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_text(text: str, path: Path, cfg: RunConfig) -> None:
    assert_write_allowed(path, cfg.write_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(data: dict[str, Any], path: Path, cfg: RunConfig) -> None:
    assert_write_allowed(path, cfg.write_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def create_results_layout(cfg: RunConfig) -> dict[str, Path]:
    root = cfg.results_root
    dirs = {
        "root": root,
        "L0": root / "00_integrity",
        "L1": root / "01_schema",
        "L2": root / "02_track_physics",
        "L3": root / "03_alignment",
        "L4": root / "04_hazard_formula",
        "L5": root / "05_emdat",
        "reports": root / "reports",
    }
    for d in dirs.values():
        if isinstance(d, Path):
            ensure_dir(d, cfg)
    return dirs


def get_stage_paths(cfg: RunConfig, year: int) -> dict[str, Path]:
    r = cfg.reproduction_root
    final_dir = r / "04_final_output" / "emdat_attribution_integration"
    audit_dir = final_dir / "audit"
    return {
        "base": r / "03_intermediate_nc" / "rubbish" / "typhoon_base" / f"typhoon_{year}_base.nc",
        "ocean": r / "03_intermediate_nc" / "ocean_integration" / f"typhoon_{year}_ocean.nc",
        "litpop": r / "03_intermediate_nc" / "lightpop_integration" / f"typhoon_{year}_ocean_litpop.nc",
        "poplight": r / "03_intermediate_nc" / "pop_light_integration" / f"typhoon_{year}_ocean_litpop_poplight.nc",
        "final": final_dir / f"typhoon_{year}_ocean_litpop_poplight_emdat.nc",
        "audit_match": audit_dir / f"emdat_record_match_{year}.csv",
        "audit_summary": audit_dir / f"emdat_record_allocation_summary_{year}.csv",
    }


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    pkgs = [
        "numpy",
        "pandas",
        "netCDF4",
        "rasterio",
        "geopandas",
        "scipy",
        "xarray",
        "shapely",
        "pyogrio",
        "fiona",
    ]
    for name in pkgs:
        try:
            mod = __import__(name)
            versions[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception as exc:
            versions[name] = f"UNAVAILABLE:{type(exc).__name__}"
    return versions


def write_run_manifest(cfg: RunConfig, selected_levels: Sequence[str], path: Path) -> None:
    data = {
        "generated_at": now_iso(),
        "project_root": str(cfg.project_root),
        "results_root": str(cfg.results_root),
        "years": cfg.years,
        "years_expr": years_to_expr(cfg.years),
        "selected_levels": list(selected_levels),
        "seed": cfg.seed,
        "workers": cfg.workers,
        "python": sys.version,
        "package_versions": package_versions(),
        "constraints": {
            "nc_open_mode": "r",
            "read_whitelist": [str(p) for p in cfg.read_roots],
            "write_root": str(cfg.write_root),
        },
    }
    write_json(data, path, cfg)


def read_only_snapshot(cfg: RunConfig) -> list[dict[str, Any]]:
    targets: list[Path] = []
    r = cfg.reproduction_root
    for y in cfg.years:
        p = get_stage_paths(cfg, y)
        targets.extend([p["base"], p["ocean"], p["litpop"], p["poplight"], p["final"]])
    snap: list[dict[str, Any]] = []

    sample_indices: set[int] = set()
    if targets:
        sample_n = min(10, len(targets))
        sample_indices.update(int(i) for i in np.linspace(0, len(targets) - 1, num=sample_n))

    for idx, p in enumerate(targets):
        row: dict[str, Any] = {"path": str(p), "exists": p.exists()}
        if p.exists():
            st = p.stat()
            row["size"] = int(st.st_size)
            row["mtime"] = float(st.st_mtime)
            row["hash_sampled"] = int(idx in sample_indices)
            if idx in sample_indices:
                row["hash_sample"] = file_hash_sample(p)
        snap.append(row)
    return snap


def file_hash_sample(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute a deterministic sample hash from file head/middle/tail chunks.
    This avoids full-file hashing on very large NetCDF while still detecting edits.
    """
    st = path.stat()
    size = int(st.st_size)
    h = hashlib.sha256()
    h.update(str(size).encode("utf-8"))
    with path.open("rb") as f:
        head = f.read(chunk_size)
        h.update(head)
        if size > chunk_size * 2:
            mid_pos = max((size // 2) - (chunk_size // 2), 0)
            f.seek(mid_pos)
            h.update(f.read(chunk_size))
        if size > chunk_size:
            tail_pos = max(size - chunk_size, 0)
            f.seek(tail_pos)
            h.update(f.read(chunk_size))
    return h.hexdigest()


def quantile_safe(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.nanquantile(values, q))


def mean_safe(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.nanmean(values))


def std_safe(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.nanstd(values))


def corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.allclose(np.nanstd(a), 0.0) or np.allclose(np.nanstd(b), 0.0):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def valid_mask_default(arr: np.ndarray) -> np.ndarray:
    return np.isfinite(arr) & (arr != -999) & (arr != -9999)


def worst_status(statuses: Iterable[str]) -> str:
    ranked = sorted(statuses, key=lambda s: STATUS_RANK.get(s, 99), reverse=True)
    return ranked[0] if ranked else "PASS"


def format_stage_table(stage_results: Sequence[StageResult]) -> str:
    lines = ["| Level | Status | Fails | Warns |", "|---|---:|---:|---:|"]
    for s in stage_results:
        lines.append(f"| {s.level} | {s.status} | {s.fail_count} | {s.warn_count} |")
    return "\n".join(lines)


def stage_should_fail(result: StageResult) -> bool:
    return result.status == "FAIL" or result.fail_count > 0


def combine_level_status(results: Sequence[StageResult]) -> str:
    if any(r.status == "FAIL" for r in results):
        return "FAIL"
    if any(r.status == "WARN" for r in results):
        return "WARN"
    return "PASS"


def pct(n: float, d: float) -> float:
    if d <= 0:
        return float("nan")
    return float(n) / float(d)


def status_from_thresholds(fail_count: int, warn_count: int) -> str:
    if fail_count > 0:
        return "FAIL"
    if warn_count > 0:
        return "WARN"
    return "PASS"


def level_dir(cfg: RunConfig, level: str) -> Path:
    layout = create_results_layout(cfg)
    return layout[level]


def make_note_lines(notes: Sequence[str]) -> str:
    if not notes:
        return "- none"
    return "\n".join(f"- {x}" for x in notes)
