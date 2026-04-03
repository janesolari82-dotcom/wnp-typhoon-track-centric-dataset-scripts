from __future__ import annotations

import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from benchmark.s00_core_common import StageResult, combine_level_status, write_json, write_run_manifest, write_text

from s00_common_v2 import build_run_config_v2, create_results_layout_v2, format_stage_table, make_shared_parser_v2
from s01_l0_release_integrity_v2 import run_l0_v2
from s02_l1_schema_metadata_v2 import run_l1_v2
from s03_l2_track_realism_v2 import run_l2_v2
from s04_l3_alignment_coverage_v2 import run_l3_v2
from s05_l4_hazard_sanity_v2 import run_l4_v2
from s06_l5_attribution_quality_v2 import run_l5_v2
from s07_l6_external_gdis_imerg_v2 import run_l6_v2
from s08_l7_imerg_case_validation_v2 import run_l7_v2


def parse_args():
    p = make_shared_parser_v2("Run benchmark_v2 full L0-L7 validation pipeline")
    p.add_argument("--rerun-existing", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = build_run_config_v2(args)
    layout = create_results_layout_v2(cfg)
    reports_dir = layout["reports"]
    write_run_manifest(cfg, ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7"], reports_dir / "run_manifest.json")

    imerg_topq = tuple(float(x.strip()) for x in str(args.imerg_topq).split(",") if x.strip())
    if len(imerg_topq) < 2:
        imerg_topq = (0.10, 0.20)

    core_status_path = reports_dir / "core_validation_status.json"
    ext_status_path = reports_dir / "external_validation_status.json"

    results = []
    if (not args.rerun_existing) and core_status_path.exists() and ext_status_path.exists():
        core_obj = json.loads(core_status_path.read_text(encoding="utf-8"))
        ext_obj = json.loads(ext_status_path.read_text(encoding="utf-8"))
        results.extend(StageResult(**item) for item in core_obj.get("levels", []))
        results.extend(StageResult(**item) for item in ext_obj.get("levels", []))
    else:
        results = [
            run_l0_v2(cfg),
            run_l1_v2(cfg),
            run_l2_v2(cfg),
            run_l3_v2(cfg),
            run_l4_v2(cfg),
            run_l5_v2(cfg),
        ]
        l6_result, l6_ctx = run_l6_v2(cfg, imerg_samples_per_year=int(args.imerg_samples_per_year), imerg_topq=imerg_topq)
        results.append(l6_result)
        results.append(run_l7_v2(cfg, l6_ctx=l6_ctx, case_count=int(args.case_count), imerg_topq=imerg_topq[0]))

    overall_status = combine_level_status(results)
    lines = [
        "# Final Validation Report (L0-L7, v2)",
        "",
        f"- years: {cfg.years[0]}-{cfg.years[-1]}",
        f"- results_root: {cfg.results_root}",
        f"- imerg_samples_per_year: {int(args.imerg_samples_per_year)}",
        f"- case_count: {int(args.case_count)}",
        f"- overall_status: **{overall_status}**",
        "",
        "## Level Status",
        "",
        format_stage_table(results),
        "",
        "## Year Windows",
        "",
        "- Core validated years: 2000-2018",
        "- Extended rainfall-only years: 2019-2020",
        "- Not-Evaluable years: 2021-2024",
        "",
        "## Artifact Roots",
        "",
        f"- 00_integrity: `{layout['L0']}`",
        f"- 01_schema: `{layout['L1']}`",
        f"- 02_track_physics: `{layout['L2']}`",
        f"- 03_alignment: `{layout['L3']}`",
        f"- 04_hazard_formula: `{layout['L4']}`",
        f"- 05_emdat: `{layout['L5']}`",
        f"- 06_external_validation: `{layout['L6']}`",
        f"- 07_case_validation: `{layout['L7']}`",
        "",
    ]
    report_path = reports_dir / "final_validation_report.md"
    write_text("\n".join(lines) + "\n", report_path, cfg)
    status_path = reports_dir / "final_validation_status.json"
    write_json(
        {
            "overall_status": overall_status,
            "levels": [x.to_dict() for x in results],
            "artifacts": {
                "run_manifest": str(reports_dir / "run_manifest.json"),
                "final_report": str(report_path),
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
