from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from benchmark.s00_core_common import write_json, write_run_manifest, write_text

from s00_common_v2 import build_run_config_v2, create_results_layout_v2, make_shared_parser_v2
from s07_l6_external_gdis_imerg_v2 import run_l6_v2
from s08_l7_imerg_case_validation_v2 import run_l7_v2


def parse_args():
    p = make_shared_parser_v2("Run benchmark_v2 external validation pipeline (L6-L7)")
    return p.parse_args()


def _compose_overall_status(results) -> str:
    if any(result.status == "FAIL" for result in results):
        return "FAIL"
    if any(result.status == "WARN" for result in results):
        return "WARN"
    return "PASS"


def main() -> int:
    args = parse_args()
    cfg = build_run_config_v2(args)
    layout = create_results_layout_v2(cfg)
    reports_dir = layout["reports"]
    write_run_manifest(cfg, ["L6", "L7"], reports_dir / "run_manifest_external.json")

    imerg_topq = tuple(float(x.strip()) for x in str(args.imerg_topq).split(",") if x.strip())
    if len(imerg_topq) < 2:
        imerg_topq = (0.10, 0.20)

    l6_result, l6_ctx = run_l6_v2(cfg, imerg_samples_per_year=int(args.imerg_samples_per_year), imerg_topq=imerg_topq)
    l7_result = run_l7_v2(cfg, l6_ctx=l6_ctx, case_count=int(args.case_count), imerg_topq=imerg_topq[0])
    results = [l6_result, l7_result]
    overall_status = _compose_overall_status(results)

    lines = [
        "# External Validation Report (L6-L7, v2)",
        "",
        f"- years: {cfg.years[0]}-{cfg.years[-1]}",
        f"- results_root: {cfg.results_root}",
        f"- imerg_samples_per_year: {int(args.imerg_samples_per_year)}",
        f"- case_count: {int(args.case_count)}",
        f"- overall_status: **{overall_status}**",
        "",
        "## Stage Results",
        "",
    ]
    for res in results:
        lines.extend(
            [
                f"### {res.level}",
                f"- status: **{res.status}**",
                f"- fail_count: {int(res.fail_count)}",
                f"- warn_count: {int(res.warn_count)}",
                "",
            ]
        )
    report_path = reports_dir / "l6_l7_validation_report.md"
    write_text("\n".join(lines) + "\n", report_path, cfg)
    status_path = reports_dir / "external_validation_status.json"
    write_json(
        {
            "overall_status": overall_status,
            "levels": [x.to_dict() for x in results],
            "artifacts": {
                "report": str(report_path),
                "run_manifest": str(reports_dir / "run_manifest_external.json"),
            },
        },
        status_path,
        cfg,
    )
    print(f"overall_status={overall_status}")
    print(f"status_json={status_path}")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
