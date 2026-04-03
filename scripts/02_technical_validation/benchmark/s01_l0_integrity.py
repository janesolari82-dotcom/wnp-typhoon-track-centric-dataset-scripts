from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from s00_core_common import (
    RunConfig,
    StageResult,
    build_run_config,
    create_results_layout,
    get_stage_paths,
    make_shared_parser,
    open_netcdf_readonly,
    status_from_thresholds,
    write_csv,
    write_text,
)

NC_STAGES = ["base", "ocean", "litpop", "poplight", "final"]
CSV_STAGES = ["audit_match", "audit_summary"]


def _check_nc(path: Path, cfg: RunConfig) -> tuple[bool, int, str]:
    try:
        with open_netcdf_readonly(path, cfg) as ds:
            group_count = len(ds.groups)
            if group_count <= 0:
                return False, group_count, "group_count<=0"
            return True, group_count, ""
    except Exception as exc:
        return False, 0, f"{type(exc).__name__}: {exc}"


def _check_csv(path: Path) -> tuple[bool, int, str]:
    try:
        df = pd.read_csv(path)
        return True, int(len(df)), ""
    except Exception as exc:
        return False, 0, f"{type(exc).__name__}: {exc}"


def run_l0(cfg: RunConfig) -> StageResult:
    layout = create_results_layout(cfg)
    out_dir = layout["L0"]
    inventory_path = out_dir / "file_inventory.csv"
    summary_path = out_dir / "integrity_summary.md"

    rows: list[dict[str, object]] = []
    fail_count = 0

    for year in cfg.years:
        p = get_stage_paths(cfg, year)

        for stage in NC_STAGES:
            f = p[stage]
            row = {
                "year": year,
                "stage": stage,
                "kind": "nc",
                "path": str(f),
                "exists": bool(f.exists()),
                "readable": False,
                "group_count": 0,
                "row_count": "",
                "error": "",
                "status": "PASS",
            }
            if not f.exists():
                row["error"] = "missing_file"
                row["status"] = "FAIL"
                fail_count += 1
            else:
                ok, group_count, err = _check_nc(f, cfg)
                row["readable"] = bool(ok)
                row["group_count"] = int(group_count)
                row["error"] = err
                if not ok:
                    row["status"] = "FAIL"
                    fail_count += 1
            rows.append(row)

        for stage in CSV_STAGES:
            f = p[stage]
            row = {
                "year": year,
                "stage": stage,
                "kind": "csv",
                "path": str(f),
                "exists": bool(f.exists()),
                "readable": False,
                "group_count": "",
                "row_count": 0,
                "error": "",
                "status": "PASS",
            }
            if not f.exists():
                row["error"] = "missing_file"
                row["status"] = "FAIL"
                fail_count += 1
            else:
                ok, row_count, err = _check_csv(f)
                row["readable"] = bool(ok)
                row["row_count"] = int(row_count)
                row["error"] = err
                if not ok:
                    row["status"] = "FAIL"
                    fail_count += 1
            rows.append(row)

    df = pd.DataFrame(rows)
    write_csv(df, inventory_path, cfg)

    summary = []
    summary.append("# L0 Integrity Summary")
    summary.append("")
    summary.append(f"- years: {cfg.years[0]}-{cfg.years[-1]}")
    summary.append(f"- records_checked: {len(df)}")
    summary.append(f"- fail_count: {fail_count}")
    summary.append("")

    by_stage = (
        df.groupby(["stage", "status"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["stage", "status"])
    )

    summary.append("## Stage Status Counts")
    summary.append("")
    summary.append("| stage | status | count |")
    summary.append("|---|---|---:|")
    for r in by_stage.itertuples(index=False):
        summary.append(f"| {r.stage} | {r.status} | {int(r.count)} |")
    summary.append("")

    failed = df[df["status"] == "FAIL"].copy()
    summary.append("## Failed Items")
    summary.append("")
    if failed.empty:
        summary.append("- none")
    else:
        for r in failed.itertuples(index=False):
            summary.append(f"- year={r.year} stage={r.stage} path={r.path} error={r.error}")

    write_text("\n".join(summary) + "\n", summary_path, cfg)

    status = status_from_thresholds(fail_count=fail_count, warn_count=0)
    return StageResult(
        level="L0",
        status=status,
        fail_count=fail_count,
        warn_count=0,
        metrics={
            "records_checked": int(len(df)),
            "failed_records": int(fail_count),
        },
        artifacts=[str(inventory_path), str(summary_path)],
        notes=[],
    )


def main() -> int:
    parser = make_shared_parser("L0 integrity check")
    args = parser.parse_args()
    cfg = build_run_config(args)
    result = run_l0(cfg)
    print(f"L0 status={result.status} fail={result.fail_count} warn={result.warn_count}")
    if cfg.strict and result.status == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
