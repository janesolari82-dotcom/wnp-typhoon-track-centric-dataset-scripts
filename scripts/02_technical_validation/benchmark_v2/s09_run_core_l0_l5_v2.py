from __future__ import annotations

import traceback
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from benchmark.s00_core_common import combine_level_status, make_note_lines, read_only_snapshot, stage_should_fail, write_json, write_run_manifest, write_text

from s00_common_v2 import build_run_config_v2, create_results_layout_v2, format_stage_table, make_shared_parser_v2
from s01_l0_release_integrity_v2 import run_l0_v2
from s02_l1_schema_metadata_v2 import run_l1_v2
from s03_l2_track_realism_v2 import run_l2_v2
from s04_l3_alignment_coverage_v2 import run_l3_v2
from s05_l4_hazard_sanity_v2 import run_l4_v2
from s06_l5_attribution_quality_v2 import run_l5_v2
from benchmark.s00_core_common import StageResult


RUNNERS = {
    "L0": run_l0_v2,
    "L1": run_l1_v2,
    "L2": run_l2_v2,
    "L3": run_l3_v2,
    "L4": run_l4_v2,
    "L5": run_l5_v2,
}


def parse_args():
    p = make_shared_parser_v2("Run benchmark_v2 core validation pipeline (L0-L5)")
    p.add_argument("--levels", type=str, default="L0,L1,L2,L3,L4,L5")
    return p.parse_args()


def _parse_levels(level_text: str) -> list[str]:
    raw = [x.strip().upper() for x in str(level_text).split(",") if x.strip()]
    ordered = [lv for lv in ["L0", "L1", "L2", "L3", "L4", "L5"] if lv in raw]
    if not ordered:
        raise ValueError(f"no valid levels in --levels={level_text}")
    return ordered


def _release_decision(stage_results: list[StageResult]) -> str:
    by_level = {x.level: x for x in stage_results}
    key_levels = ["L0", "L1", "L4", "L5"]
    key_fail = any(by_level.get(k) and by_level[k].status == "FAIL" for k in key_levels)
    if key_fail:
        return "No Go"
    any_warn = any(x.status == "WARN" for x in stage_results)
    if any_warn:
        return "Conditional Go"
    return "Go"


def main() -> int:
    args = parse_args()
    cfg = build_run_config_v2(args)
    layout = create_results_layout_v2(cfg)
    reports_dir = layout["reports"]
    selected_levels = _parse_levels(args.levels)
    write_run_manifest(cfg, selected_levels, reports_dir / "run_manifest.json")

    pre_snapshot = read_only_snapshot(cfg)
    stage_results: list[StageResult] = []
    pipeline_failed = False
    for lv in selected_levels:
        runner = RUNNERS[lv]
        try:
            stage_results.append(runner(cfg))
        except Exception as exc:
            pipeline_failed = True
            stage_results.append(
                StageResult(
                    level=lv,
                    status="FAIL",
                    fail_count=1,
                    warn_count=0,
                    metrics={"exception": str(exc)},
                    artifacts=[],
                    notes=[traceback.format_exc()],
                )
            )

    post_snapshot = read_only_snapshot(cfg)
    changed = []
    by_pre = {x["path"]: x for x in pre_snapshot}
    by_post = {x["path"]: x for x in post_snapshot}
    for path, before in by_pre.items():
        after = by_post.get(path)
        if after is None or before.get("exists") != after.get("exists") or before.get("size") != after.get("size") or before.get("mtime") != after.get("mtime") or (before.get("hash_sampled") and after.get("hash_sampled") and before.get("hash_sample") != after.get("hash_sample")):
            changed.append({"path": path, "before": before, "after": after})
    if changed:
        pipeline_failed = True

    overall_status = combine_level_status(stage_results)
    if changed and overall_status != "FAIL":
        overall_status = "FAIL"
    decision = _release_decision(stage_results)

    lines = [
        "# Core Validation Report (L0-L5, v2)",
        "",
        f"- years: {cfg.years[0]}-{cfg.years[-1]}",
        f"- results_root: {cfg.results_root}",
        f"- imerg_samples_per_year: {int(args.imerg_samples_per_year)}",
        f"- seed: {cfg.seed}",
        f"- workers: {cfg.workers}",
        "",
        "## Level Status",
        "",
        format_stage_table(stage_results),
        "",
        "## Read-only Guard",
        "",
        "- PASS: no changes detected in tracked input NC files." if not changed else "- FAIL: input files changed during validation run.",
        "",
        "## Release Decision",
        "",
        f"- decision: **{decision}**",
        "",
        "## Notes",
        "",
    ]
    for res in stage_results:
        if res.notes:
            lines.append(f"### {res.level}")
            lines.append(make_note_lines(res.notes))
            lines.append("")

    final_report_path = reports_dir / "core_validation_report.md"
    write_text("\n".join(lines) + "\n", final_report_path, cfg)
    status_path = reports_dir / "core_validation_status.json"
    write_json(
        {
            "overall_status": overall_status,
            "release_decision": decision,
            "readonly_changes": changed,
            "levels": [x.to_dict() for x in stage_results],
            "artifacts": {
                "run_manifest": str(reports_dir / "run_manifest.json"),
                "final_report": str(final_report_path),
            },
        },
        status_path,
        cfg,
    )
    print(f"overall_status={overall_status}")
    print(f"release_decision={decision}")
    print(f"status_json={status_path}")
    print(f"final_report={final_report_path}")
    has_fail = any(stage_should_fail(x) for x in stage_results)
    if cfg.strict and (has_fail or pipeline_failed):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
