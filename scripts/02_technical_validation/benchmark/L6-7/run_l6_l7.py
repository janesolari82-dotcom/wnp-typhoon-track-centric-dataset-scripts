from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from core_common import (  # noqa: E402
    STATUS_RANK,
    build_run_config,
    make_shared_parser,
    write_json,
    write_text,
)
from l6_external_validation import run_l6_external_validation  # noqa: E402
from l7_case_validation import run_l7_case_validation  # noqa: E402


def _compose_overall_status(statuses: list[str]) -> str:
    if not statuses:
        return "WARN"
    best = max(statuses, key=lambda s: STATUS_RANK.get(str(s), 1))
    return str(best)


def main() -> int:
    parser = make_shared_parser("Run L6-L7 external/case validation (independent entry)")
    parser.add_argument("--case-count", type=int, default=5, help="L7 case count, clipped to [3,5]")
    parser.add_argument(
        "--geometry-sample-size",
        type=int,
        default=200,
        help="L6 GDIS geometry sample size",
    )
    args = parser.parse_args()
    cfg = build_run_config(args)

    l6_result, l6_ctx = run_l6_external_validation(cfg, geometry_sample_size=int(max(1, args.geometry_sample_size)))
    l7_result = run_l7_case_validation(cfg, l6_ctx, case_count=int(args.case_count))

    overall = _compose_overall_status([l6_result.status, l7_result.status])
    reports_dir = cfg.results_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "l6_l7_report.md"
    status_path = reports_dir / "l6_l7_status.json"

    lines = [
        "# L6-L7 Validation Report",
        "",
        "## Run Config",
        "",
        f"- project_root: {cfg.project_root}",
        f"- results_root: {cfg.results_root}",
        f"- years: {cfg.years}",
        f"- seed: {cfg.seed}",
        f"- workers: {cfg.workers}",
        f"- case_count: {int(args.case_count)}",
        f"- geometry_sample_size: {int(args.geometry_sample_size)}",
        "",
        "## Stage Status",
        "",
        f"- L6: **{l6_result.status}** (fail={l6_result.fail_count}, warn={l6_result.warn_count})",
        f"- L7: **{l7_result.status}** (fail={l7_result.fail_count}, warn={l7_result.warn_count})",
        f"- overall: **{overall}**",
        "",
        "## Artifacts",
        "",
        "- L6 root: `results/06_external_validation/`",
        "- L7 root: `results/07_case_validation/`",
        "- this report does not overwrite existing L0-L5 final report.",
        "",
    ]
    write_text("\n".join(lines) + "\n", report_path, cfg)

    write_json(
        {
            "overall_status": overall,
            "l6": l6_result.to_dict(),
            "l7": l7_result.to_dict(),
            "report_path": str(report_path),
        },
        status_path,
        cfg,
    )

    print(
        f"L6={l6_result.status}(fail={l6_result.fail_count},warn={l6_result.warn_count}) "
        f"L7={l7_result.status}(fail={l7_result.fail_count},warn={l7_result.warn_count}) "
        f"overall={overall}"
    )

    if cfg.strict and overall == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

