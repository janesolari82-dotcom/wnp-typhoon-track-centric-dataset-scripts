#!/usr/bin/env python3
"""
Phase 4: workflow runner
  1) trial year (default 2013)
  2) validate trial
  3) if pass, run full years and validate
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

from s00_common import LOG_DIR, parse_years_expr, setup_logger, years_to_expr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run EM-DAT integration workflow with validation gate.")
    p.add_argument("--trial-years", type=str, default="2013")
    p.add_argument("--full-years", type=str, default="2000-2024")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--skip-trial-in-full", dest="skip_trial_in_full", action="store_true")
    g.add_argument("--include-trial-in-full", dest="skip_trial_in_full", action="store_false")
    p.set_defaults(skip_trial_in_full=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--pad-days", type=int, default=1)
    p.add_argument("--log-level", type=str, default="INFO")
    p.add_argument("--trial-only", action="store_true")
    return p.parse_args()


def run_cmd(
    cmd: List[str],
    logger,
    soft_success_markers: List[str] | None = None,
    soft_success_returncodes: List[int] | None = None,
) -> bool:
    logger.info("RUN: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        logger.info(proc.stdout.strip())
    if proc.stderr:
        logger.info(proc.stderr.strip())
    if proc.returncode != 0 and soft_success_markers and soft_success_returncodes:
        out_text = f"{proc.stdout or ''}\n{proc.stderr or ''}"
        if proc.returncode in set(int(x) for x in soft_success_returncodes):
            if all(m in out_text for m in soft_success_markers):
                logger.warning("Command returned soft-fail rc=%s but success markers found. Continue.", proc.returncode)
                return True
    if proc.returncode != 0:
        logger.error("FAILED rc=%s", proc.returncode)
        return False
    return True


def main() -> int:
    args = parse_args()
    logger = setup_logger("emdat_phase11_runner", LOG_DIR / "11_run_emdat_workflow_v4.log", args.log_level)

    this_py = sys.executable
    script_dir = Path(__file__).resolve().parent
    s08 = script_dir / "s01_integrate_era5_daily_windows.py"
    s09 = script_dir / "s02_integrate_emdat_attribution.py"
    s10 = script_dir / "s03_validate_emdat_workflow_v4.py"

    trial_years = parse_years_expr(args.trial_years)
    full_years = parse_years_expr(args.full_years)
    if args.skip_trial_in_full:
        full_years = [y for y in full_years if y not in set(trial_years)]

    trial_expr = years_to_expr(trial_years)
    full_expr = years_to_expr(full_years)

    logger.info("Workflow start | trial=%s full=%s", trial_expr, full_expr)

    # Trial stage
    if not run_cmd(
        [
            this_py,
            str(s08),
            "--years",
            trial_expr,
            "--log-level",
            args.log_level,
            *([] if not args.overwrite else ["--overwrite"]),
        ],
        logger,
    ):
        return 1
    if not run_cmd(
        [
            this_py,
            str(s09),
            "--years",
            trial_expr,
            "--pad-days",
            str(args.pad_days),
            "--match-policy",
            "hybrid_abc",
            "--log-level",
            args.log_level,
            *([] if not args.overwrite else ["--overwrite"]),
        ],
        logger,
        soft_success_markers=["Phase 2 finished | success=True"],
        soft_success_returncodes=[3221225477, -1073741819],
    ):
        return 1
    if not run_cmd(
        [
            this_py,
            str(s10),
            "--years",
            trial_expr,
            "--log-level",
            args.log_level,
        ],
        logger,
    ):
        logger.error("Trial validation failed, stop before full run.")
        return 1

    if args.trial_only:
        logger.info("Trial-only mode complete.")
        return 0

    # Full stage
    if full_years:
        if not run_cmd(
            [
                this_py,
                str(s08),
                "--years",
                full_expr,
                "--log-level",
                args.log_level,
                *([] if not args.overwrite else ["--overwrite"]),
            ],
            logger,
        ):
            return 1
        if not run_cmd(
            [
                this_py,
                str(s09),
                "--years",
                full_expr,
                "--pad-days",
                str(args.pad_days),
                "--match-policy",
                "hybrid_abc",
                "--log-level",
                args.log_level,
                *([] if not args.overwrite else ["--overwrite"]),
            ],
            logger,
            soft_success_markers=["Phase 2 finished | success=True"],
            soft_success_returncodes=[3221225477, -1073741819],
        ):
            return 1
        if not run_cmd(
            [
                this_py,
                str(s10),
                "--years",
                full_expr,
                "--log-level",
                args.log_level,
            ],
            logger,
        ):
            return 1

    logger.info("Workflow complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
