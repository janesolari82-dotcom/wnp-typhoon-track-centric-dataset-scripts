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

PAIR_ORDER = [("base", "ocean"), ("ocean", "litpop"), ("litpop", "poplight"), ("poplight", "final")]

REQUIRED_ROOT_ATTRS = {"Conventions", "creation_date", "year", "num_typhoons"}
FIXED_WINDOW_DIMS = {"window_time": 15, "window_lat": 41, "window_lon": 41}

REQUIRED_VARS_BY_STAGE = {
    "base": ["time", "center_lat", "center_lon", "wind_speed", "central_pressure", "radius_max_wind"],
    "ocean": [
        "window_time",
        "window_lat",
        "window_lon",
        "ocean_thetao",
        "ocean_zos",
        "ocean_so",
        "ocean_uo",
        "ocean_vo",
        "ocean_mlotst",
    ],
    "litpop": ["litpop"],
    "poplight": ["nightlight_intensity", "population_count"],
    "final": [
        "era5_u10_daily_max",
        "era5_v10_daily_max",
        "era5_wind_speed_daily_proxy",
        "era5_tp_daily_sum_mm",
        "hazard_wind_power_daily",
        "hazard_rain_daily",
        "hazard_compound_daily",
        "emdat_loss_allocated_usd",
        "emdat_match_count",
        "emdat_loss_weight_norm",
    ],
}


def _issue(
    year: int,
    check_type: str,
    severity: str,
    stage_src: str,
    stage_dst: str,
    group: str,
    var: str,
    expected: str,
    actual: str,
    message: str,
) -> dict[str, object]:
    return {
        "year": year,
        "check_type": check_type,
        "severity": severity,
        "stage_src": stage_src,
        "stage_dst": stage_dst,
        "group": group,
        "var": var,
        "expected": expected,
        "actual": actual,
        "message": message,
    }


def _check_stage_basics(year: int, stage: str, path: Path, cfg: RunConfig) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    try:
        with open_netcdf_readonly(path, cfg) as ds:
            root_attrs = set(ds.ncattrs())
            missing_attrs = sorted(REQUIRED_ROOT_ATTRS - root_attrs)
            for attr in missing_attrs:
                issues.append(
                    _issue(
                        year,
                        "root_attr",
                        "FAIL",
                        stage,
                        stage,
                        "",
                        attr,
                        "present",
                        "missing",
                        f"missing required root attr: {attr}",
                    )
                )

            if stage == "final":
                emdat_attrs = [x for x in root_attrs if x.startswith("emdat_")]
                if not emdat_attrs:
                    issues.append(
                        _issue(
                            year,
                            "root_attr",
                            "FAIL",
                            stage,
                            stage,
                            "",
                            "emdat_*",
                            "exists",
                            "missing",
                            "final stage missing emdat_* root attrs",
                        )
                    )

            for gname, grp in ds.groups.items():
                req = REQUIRED_VARS_BY_STAGE.get("base", []) + REQUIRED_VARS_BY_STAGE.get(stage, [])
                for var in req:
                    if var not in grp.variables:
                        issues.append(
                            _issue(
                                year,
                                "required_var",
                                "FAIL",
                                stage,
                                stage,
                                gname,
                                var,
                                "present",
                                "missing",
                                f"required variable missing in stage={stage}",
                            )
                        )

                if stage in {"ocean", "litpop", "poplight", "final"}:
                    for dname, dlen in FIXED_WINDOW_DIMS.items():
                        dim = grp.dimensions.get(dname)
                        actual = "missing" if dim is None else str(len(dim))
                        if dim is None or len(dim) != dlen:
                            issues.append(
                                _issue(
                                    year,
                                    "fixed_dim",
                                    "FAIL",
                                    stage,
                                    stage,
                                    gname,
                                    dname,
                                    str(dlen),
                                    actual,
                                    f"unexpected fixed dimension in stage={stage}",
                                )
                            )
    except Exception as exc:
        issues.append(
            _issue(
                year,
                "open_file",
                "FAIL",
                stage,
                stage,
                "",
                "",
                "readable",
                "unreadable",
                f"{type(exc).__name__}: {exc}",
            )
        )
    return issues


def _check_pair(year: int, src_stage: str, dst_stage: str, src_path: Path, dst_path: Path, cfg: RunConfig) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []

    try:
        with open_netcdf_readonly(src_path, cfg) as src, open_netcdf_readonly(dst_path, cfg) as dst:
            src_groups = set(src.groups.keys())
            dst_groups = set(dst.groups.keys())

            for g in sorted(src_groups - dst_groups):
                issues.append(
                    _issue(
                        year,
                        "group_set",
                        "FAIL",
                        src_stage,
                        dst_stage,
                        g,
                        "",
                        "exists_in_dst",
                        "missing",
                        "group missing in destination stage",
                    )
                )
            for g in sorted(dst_groups - src_groups):
                issues.append(
                    _issue(
                        year,
                        "group_set",
                        "FAIL",
                        src_stage,
                        dst_stage,
                        g,
                        "",
                        "not_extra",
                        "extra",
                        "extra group exists in destination stage",
                    )
                )

            for g in sorted(src_groups & dst_groups):
                sg = src.groups[g]
                dg = dst.groups[g]
                for var_name, svar in sg.variables.items():
                    if var_name not in dg.variables:
                        issues.append(
                            _issue(
                                year,
                                "var_presence",
                                "FAIL",
                                src_stage,
                                dst_stage,
                                g,
                                var_name,
                                "exists_in_dst",
                                "missing",
                                "original variable missing in destination stage",
                            )
                        )
                        continue
                    dvar = dg.variables[var_name]
                    s_dims = tuple(str(x) for x in svar.dimensions)
                    d_dims = tuple(str(x) for x in dvar.dimensions)
                    if s_dims != d_dims:
                        issues.append(
                            _issue(
                                year,
                                "var_dims",
                                "FAIL",
                                src_stage,
                                dst_stage,
                                g,
                                var_name,
                                str(s_dims),
                                str(d_dims),
                                "dimension drift on original variable",
                            )
                        )
                    s_dtype = str(svar.dtype)
                    d_dtype = str(dvar.dtype)
                    if s_dtype != d_dtype:
                        issues.append(
                            _issue(
                                year,
                                "var_dtype",
                                "FAIL",
                                src_stage,
                                dst_stage,
                                g,
                                var_name,
                                s_dtype,
                                d_dtype,
                                "dtype drift on original variable",
                            )
                        )

    except Exception as exc:
        issues.append(
            _issue(
                year,
                "open_pair",
                "FAIL",
                src_stage,
                dst_stage,
                "",
                "",
                "readable",
                "unreadable",
                f"{type(exc).__name__}: {exc}",
            )
        )

    return issues


def run_l1(cfg: RunConfig) -> StageResult:
    layout = create_results_layout(cfg)
    out_dir = layout["L1"]
    summary_path = out_dir / "schema_validation_summary.md"

    all_issues: list[dict[str, object]] = []
    total_fail = 0
    total_warn = 0

    for year in cfg.years:
        p = get_stage_paths(cfg, year)
        year_issues: list[dict[str, object]] = []

        for stage in ["base", "ocean", "litpop", "poplight", "final"]:
            year_issues.extend(_check_stage_basics(year, stage, p[stage], cfg))

        for src_stage, dst_stage in PAIR_ORDER:
            year_issues.extend(_check_pair(year, src_stage, dst_stage, p[src_stage], p[dst_stage], cfg))

        if not year_issues:
            year_issues.append(
                _issue(
                    year,
                    "schema",
                    "PASS",
                    "all",
                    "all",
                    "",
                    "",
                    "",
                    "",
                    "no schema drift detected",
                )
            )

        ydf = pd.DataFrame(year_issues)
        write_csv(ydf, out_dir / f"schema_diff_{year}.csv", cfg)
        all_issues.extend(year_issues)

        total_fail += int((ydf["severity"] == "FAIL").sum())
        total_warn += int((ydf["severity"] == "WARN").sum())

    adf = pd.DataFrame(all_issues)
    by_year = (
        adf.groupby(["year", "severity"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["year", "severity"])
    )

    lines: list[str] = []
    lines.append("# L1 Schema Validation Summary")
    lines.append("")
    lines.append(f"- years: {cfg.years[0]}-{cfg.years[-1]}")
    lines.append(f"- total_fail: {total_fail}")
    lines.append(f"- total_warn: {total_warn}")
    lines.append("")
    lines.append("## By Year")
    lines.append("")
    lines.append("| year | severity | count |")
    lines.append("|---:|---|---:|")
    if by_year.empty:
        lines.append("| - | - | 0 |")
    else:
        for r in by_year.itertuples(index=False):
            lines.append(f"| {int(r.year)} | {r.severity} | {int(r.count)} |")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- FAIL when original var dims/dtype drift, group mismatch, fixed dims mismatch, or required attrs/vars missing.")

    write_text("\n".join(lines) + "\n", summary_path, cfg)

    status = status_from_thresholds(fail_count=total_fail, warn_count=total_warn)
    return StageResult(
        level="L1",
        status=status,
        fail_count=total_fail,
        warn_count=total_warn,
        metrics={
            "issue_records": int(len(adf)),
            "total_fail": int(total_fail),
            "total_warn": int(total_warn),
        },
        artifacts=[str(summary_path)] + [str(out_dir / f"schema_diff_{y}.csv") for y in cfg.years],
        notes=[],
    )


def main() -> int:
    parser = make_shared_parser("L1 schema and metadata validation")
    args = parser.parse_args()
    cfg = build_run_config(args)
    result = run_l1(cfg)
    print(f"L1 status={result.status} fail={result.fail_count} warn={result.warn_count}")
    if cfg.strict and result.status == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
