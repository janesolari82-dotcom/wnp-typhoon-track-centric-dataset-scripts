from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from benchmark.s00_core_common import RunConfig, StageResult, open_netcdf_readonly, write_csv, write_json, write_text  # noqa: E402
from s00_common_v2 import (  # noqa: E402
    extract_track_series,
    finalize_stage_status,
    grade_priority,
    project_window_abs_coords,
    quantile_safe,
    sample_imerg_fields,
    save_figure,
    split_groups,
    stage_dirs,
    write_plot_data,
)
from s07_l6_external_gdis_imerg_v2 import _event_day_grids  # noqa: E402


def _safe_case_id(disno: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(disno or "").strip())


def _case_status(
    peak_lag_days: float,
    iou_top10pct: float,
    recall_top10pct: float,
    imerg_project_rain_corr: float,
    event_total_rain_bias_ratio: float,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    has_fail = False
    has_warn = False

    if np.isfinite(iou_top10pct) and np.isfinite(recall_top10pct) and iou_top10pct < 0.05 and recall_top10pct < 0.10:
        has_fail = True
        reasons.append("spatial_displacement")
    if np.isfinite(imerg_project_rain_corr) and imerg_project_rain_corr <= 0 and np.isfinite(event_total_rain_bias_ratio) and event_total_rain_bias_ratio > 0.50:
        has_fail = True
        reasons.append("hazard_proxy_mismatch")

    if not has_fail:
        if np.isfinite(peak_lag_days) and abs(float(peak_lag_days)) > 3:
            has_warn = True
            reasons.append("temporal_mismatch")
        if np.isfinite(iou_top10pct) and 0.05 <= iou_top10pct < 0.15:
            has_warn = True
            reasons.append("spatial_displacement")
        if np.isfinite(imerg_project_rain_corr) and 0 < imerg_project_rain_corr < 0.30:
            has_warn = True
            reasons.append("hazard_proxy_mismatch")

    status = finalize_stage_status(has_fail=has_fail, has_warn=has_warn)
    if not reasons:
        reasons.append("none")
    return status, sorted(set(reasons))


def _topq_mask(arr: np.ndarray, q: float = 0.10) -> np.ndarray:
    vals = np.asarray(arr, dtype=float)
    valid = np.isfinite(vals)
    if np.count_nonzero(valid) == 0:
        return np.zeros_like(vals, dtype=bool)
    thr = float(np.nanquantile(vals[valid], 1.0 - q))
    return valid & (vals >= thr)


def _overlay_metrics(project_arr: np.ndarray, imerg_arr: np.ndarray, q: float = 0.10) -> tuple[float, float]:
    mp = _topq_mask(project_arr, q=q)
    mi = _topq_mask(imerg_arr, q=q)
    union = int(np.count_nonzero(mp | mi))
    inter = int(np.count_nonzero(mp & mi))
    iou = float(inter / union) if union > 0 else float("nan")
    recall = float(inter / int(np.count_nonzero(mp))) if np.count_nonzero(mp) > 0 else float("nan")
    return iou, recall


def _event_project_var(ds, selected_groups: list[str], pad_start, pad_end, var_name: str) -> np.ndarray | None:
    from netCDF4 import num2date
    from datetime import date

    rows = []
    for group_name in selected_groups:
        grp = ds.groups.get(group_name)
        if grp is None or "time" not in grp.variables or var_name not in grp.variables:
            continue
        tvals = np.asarray(grp.variables["time"][:], dtype=np.float64)
        tunits = str(grp.variables["time"].units)
        dts = num2date(tvals, units=tunits, calendar="standard")
        for idx, dt in enumerate(dts):
            current_date = date(int(dt.year), int(dt.month), int(dt.day))
            if pad_start and current_date < pad_start:
                continue
            if pad_end and current_date > pad_end:
                continue
            rows.append(np.asarray(grp.variables[var_name][idx, :, :], dtype=float))
    if not rows:
        return None
    return np.nanmean(np.stack(rows, axis=0), axis=0)


def _plot_grid(grid: np.ndarray, title: str, out_png: Path, cfg: RunConfig, cmap: str = "viridis") -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.imshow(grid, origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("window_lon index")
    ax.set_ylabel("window_lat index")
    fig.colorbar(im, ax=ax, shrink=0.8)
    save_figure(fig, out_png, cfg)


def run_l7_v2(
    cfg: RunConfig,
    l6_ctx: dict[str, Any],
    case_count: int = 5,
    imerg_topq: float = 0.10,
) -> StageResult:
    out = stage_dirs(cfg, "L7")
    target_case_count = int(max(3, min(5, int(case_count))))

    events = l6_ctx["project_events_imerg"].copy()
    event_metrics = l6_ctx["imerg_event_metrics"].copy()
    imerg_index = l6_ctx["imerg_index"]
    imerg_meta = l6_ctx["imerg_meta"]

    candidates = events.merge(
        event_metrics[
            [
                "year",
                "DisNo",
                "ISO",
                "event_total_project_rain",
                "event_total_imerg_rain",
                "event_total_rain_bias",
                "imerg_event_peak_day_lag",
                "imerg_event_topq_footprint_iou",
                "imerg_event_topq_footprint_recall",
                "imerg_event_total_rainfall_corr",
            ]
        ],
        how="inner",
        on=["year", "DisNo", "ISO"],
    )
    if not candidates.empty:
        candidates = candidates[candidates["year"] <= 2020].copy()
        candidates = candidates[pd.to_numeric(candidates["L_usd"], errors="coerce").fillna(0.0) > 0].copy()
        candidates = candidates[candidates["selected_groups"].astype(str).str.strip() != ""].copy()
        candidates["grade_priority"] = candidates["match_grade"].map(grade_priority)
        candidates = candidates.sort_values(["L_usd", "grade_priority", "DisNo"], ascending=[False, True, True]).reset_index(drop=True)
        candidates["candidate_rank"] = np.arange(1, len(candidates) + 1)
        candidates["selected"] = 0
        selected = candidates.head(target_case_count).copy()
        candidates.loc[selected.index, "selected"] = 1
    else:
        selected = pd.DataFrame()
        candidates = pd.DataFrame()

    write_csv(candidates, out["root"] / "case_catalog.csv", cfg)

    metrics_rows = []
    case_artifacts: list[str] = []
    status_counts = {"PASS": 0, "WARN": 0, "FAIL": 0}

    for order, (_, row) in enumerate(selected.iterrows(), start=1):
        year = int(row["year"])
        disno = str(row["DisNo"])
        iso = str(row["ISO"])
        case_id = _safe_case_id(disno)
        final_nc = Path(str(row["final_nc"]))
        selected_groups = split_groups(row["selected_groups"])
        pad_start = row.get("PadStart")
        pad_end = row.get("PadEnd")

        with open_netcdf_readonly(final_nc, cfg) as ds:
            track_df = extract_track_series(ds, selected_groups, pad_start, pad_end)
            day_rows, agg_project_rain, agg_imerg_rain = _event_day_grids(ds, selected_groups, pad_start, pad_end, imerg_index, imerg_meta)
            agg_project_compound = _event_project_var(ds, selected_groups, pad_start, pad_end, "hazard_compound_daily")
            agg_loss = _event_project_var(ds, selected_groups, pad_start, pad_end, "emdat_loss_allocated_usd")

            if not track_df.empty:
                peak_idx = pd.to_numeric(track_df["hazard_peak"], errors="coerce").astype(float).idxmax()
                peak_group = str(track_df.loc[peak_idx, "group"])
                peak_time_idx = int(track_df.loc[peak_idx, "time_idx"])
                peak_date = pd.to_datetime(track_df.loc[peak_idx, "time"]).date()
                grp = ds.groups[peak_group]
                hazard_rain_slice = np.asarray(grp.variables["hazard_rain_daily"][peak_time_idx, :, :], dtype=float)
                hazard_compound_slice = np.asarray(grp.variables["hazard_compound_daily"][peak_time_idx, :, :], dtype=float)
                loss_slice = np.asarray(grp.variables["emdat_loss_allocated_usd"][peak_time_idx, :, :], dtype=float)
                lat_grid, lon_grid = project_window_abs_coords(grp, peak_time_idx)
                imerg_slice = sample_imerg_fields(imerg_index[peak_date], lat_grid, lon_grid, imerg_meta)["precipitation"] if peak_date in imerg_index else np.full_like(hazard_rain_slice, np.nan)
                top_project_mask = _topq_mask(hazard_rain_slice, q=imerg_topq)
                top_imerg_mask = _topq_mask(imerg_slice, q=imerg_topq)
                overlay_rows = []
                for rr, cc in zip(*np.where(top_project_mask)):
                    overlay_rows.append({"lon": float(lon_grid[rr, cc]), "lat": float(lat_grid[rr, cc]), "source": "project_topq"})
                for rr, cc in zip(*np.where(top_imerg_mask)):
                    overlay_rows.append({"lon": float(lon_grid[rr, cc]), "lat": float(lat_grid[rr, cc]), "source": "imerg_topq"})
                overlay_df = pd.DataFrame(overlay_rows)
            else:
                peak_date = None
                hazard_rain_slice = agg_project_rain if agg_project_rain is not None else np.full((41, 41), np.nan)
                hazard_compound_slice = agg_project_compound if agg_project_compound is not None else np.full((41, 41), np.nan)
                loss_slice = agg_loss if agg_loss is not None else np.full((41, 41), np.nan)
                imerg_slice = agg_imerg_rain if agg_imerg_rain is not None else np.full((41, 41), np.nan)
                overlay_df = pd.DataFrame(columns=["lon", "lat", "source"])

        daily_df = pd.DataFrame(
            {
                "date": [item["date"] for item in day_rows],
                "project_total_rain": [item["project_total_rain"] for item in day_rows],
                "imerg_total_rain": [item["imerg_total_rain"] for item in day_rows],
            }
        )
        iou_top10pct = float(row["imerg_event_topq_footprint_iou"])
        recall_top10pct = float(row["imerg_event_topq_footprint_recall"])
        imerg_project_rain_corr = float(row["imerg_event_total_rainfall_corr"])
        event_total_rain_bias = float(row["event_total_rain_bias"])
        peak_lag_days = float(row["imerg_event_peak_day_lag"])
        denom = max(abs(float(row["event_total_imerg_rain"])), 1e-6)
        event_total_rain_bias_ratio = abs(event_total_rain_bias) / denom

        case_status, reason_tags = _case_status(
            peak_lag_days=peak_lag_days,
            iou_top10pct=iou_top10pct,
            recall_top10pct=recall_top10pct,
            imerg_project_rain_corr=imerg_project_rain_corr,
            event_total_rain_bias_ratio=event_total_rain_bias_ratio,
        )
        status_counts[case_status] += 1

        track_csv = out["csv"] / f"case_{case_id}_track_timeseries_plot_data.csv"
        overlay_csv = out["csv"] / f"case_{case_id}_overlay_plot_data.csv"
        write_plot_data(track_df, track_csv, cfg)
        write_plot_data(overlay_df, overlay_csv, cfg)

        track_png = out["png"] / f"case_{case_id}_track_timeseries.png"
        fig, ax1 = plt.subplots(figsize=(9, 4.8))
        if not track_df.empty:
            ax1.plot(pd.to_datetime(track_df["time"]), track_df["hazard_peak"], marker="o", color="tab:red", label="project hazard peak")
        ax2 = ax1.twinx()
        if not daily_df.empty:
            ax2.plot(pd.to_datetime(daily_df["date"]), daily_df["imerg_total_rain"], marker="o", color="tab:blue", label="IMERG daily rain")
        ax1.set_title(f"Case {disno}: Track-Time Series Context")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Project hazard peak", color="tab:red")
        ax2.set_ylabel("IMERG daily rain", color="tab:blue")
        ax1.grid(alpha=0.2)
        save_figure(fig, track_png, cfg)

        rain_png = out["png"] / f"case_{case_id}_hazard_rain_map.png"
        _plot_grid(hazard_rain_slice, f"{disno} hazard_rain", rain_png, cfg, cmap="YlGnBu")

        compound_png = out["png"] / f"case_{case_id}_hazard_compound_map.png"
        _plot_grid(hazard_compound_slice, f"{disno} hazard_compound", compound_png, cfg, cmap="YlOrRd")

        imerg_png = out["png"] / f"case_{case_id}_imerg_daily_precip_map.png"
        _plot_grid(imerg_slice, f"{disno} IMERG daily precipitation", imerg_png, cfg, cmap="Blues")

        overlay_png = out["png"] / f"case_{case_id}_project_vs_imerg_overlay.png"
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        if not overlay_df.empty:
            for source, sub in overlay_df.groupby("source"):
                ax.scatter(sub["lon"], sub["lat"], s=24, alpha=0.8, label=source)
        ax.set_title(f"{disno} project vs IMERG overlay")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.2)
        if not overlay_df.empty:
            ax.legend()
        save_figure(fig, overlay_png, cfg)

        loss_png = out["png"] / f"case_{case_id}_loss_allocation_map.png"
        _plot_grid(loss_slice, f"{disno} loss allocation", loss_png, cfg, cmap="magma")

        metrics_rows.append(
            {
                "case_order": order,
                "year": year,
                "DisNo": disno,
                "ISO": iso,
                "L_usd": float(row["L_usd"]),
                "match_grade": str(row["match_grade"]),
                "event_total_project_rain": float(row["event_total_project_rain"]),
                "event_total_imerg_rain": float(row["event_total_imerg_rain"]),
                "event_total_rain_bias": event_total_rain_bias,
                "peak_lag_days": peak_lag_days,
                "iou_top10pct": iou_top10pct,
                "recall_top10pct": recall_top10pct,
                "imerg_project_rain_corr": imerg_project_rain_corr,
                "status": case_status,
                "reason_tags": ";".join(reason_tags),
            }
        )

        report_lines = [
            f"# Case Validation Report: {disno}",
            "",
            f"- year: {year}",
            f"- ISO: {iso}",
            f"- L_usd: {float(row['L_usd']):.2f}",
            f"- match_grade: {str(row['match_grade'])}",
            f"- status: **{case_status}**",
            f"- reason_tags: {';'.join(reason_tags)}",
            "",
            "## Metrics",
            "",
            f"- event_total_project_rain: {float(row['event_total_project_rain']):.4f}",
            f"- event_total_imerg_rain: {float(row['event_total_imerg_rain']):.4f}",
            f"- event_total_rain_bias: {event_total_rain_bias:.4f}",
            f"- peak_lag_days: {peak_lag_days:.1f}",
            f"- iou_top10pct: {iou_top10pct:.4f}" if np.isfinite(iou_top10pct) else "- iou_top10pct: nan",
            f"- recall_top10pct: {recall_top10pct:.4f}" if np.isfinite(recall_top10pct) else "- recall_top10pct: nan",
            f"- imerg_project_rain_corr: {imerg_project_rain_corr:.4f}" if np.isfinite(imerg_project_rain_corr) else "- imerg_project_rain_corr: nan",
            "",
            "## Artifacts",
            "",
            f"- {track_png}",
            f"- {rain_png}",
            f"- {compound_png}",
            f"- {imerg_png}",
            f"- {overlay_png}",
            f"- {loss_png}",
        ]
        report_path = out["md"] / f"case_{case_id}_report.md"
        write_text("\n".join(report_lines) + "\n", report_path, cfg)
        case_artifacts.extend([str(track_csv), str(overlay_csv), str(track_png), str(rain_png), str(compound_png), str(imerg_png), str(overlay_png), str(loss_png), str(report_path)])

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["case_order", "DisNo"]).reset_index(drop=True) if metrics_rows else pd.DataFrame()
    write_csv(metrics_df, out["root"] / "case_metrics.csv", cfg)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.bar(list(status_counts.keys()), [status_counts[k] for k in status_counts], color=["tab:green", "tab:orange", "tab:red"])
    ax.set_title("Case Status Counts")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2, axis="y")
    save_figure(fig, out["png"] / "case_status_counts.png", cfg)
    write_plot_data(pd.DataFrame({"status": list(status_counts.keys()), "count": [status_counts[k] for k in status_counts]}), out["csv"] / "case_status_counts_plot_data.csv", cfg)

    lag_df = metrics_df[["DisNo", "peak_lag_days", "iou_top10pct"]].copy() if not metrics_df.empty else pd.DataFrame(columns=["DisNo", "peak_lag_days", "iou_top10pct"])
    write_plot_data(lag_df, out["csv"] / "case_peak_lag_vs_iou_scatter_plot_data.csv", cfg)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    if not lag_df.empty:
        ax.scatter(lag_df["peak_lag_days"], lag_df["iou_top10pct"], alpha=0.8)
        for _, row in lag_df.iterrows():
            ax.annotate(str(row["DisNo"]), (row["peak_lag_days"], row["iou_top10pct"]), fontsize=7)
    ax.set_title("Peak Lag vs IoU")
    ax.set_xlabel("peak_lag_days")
    ax.set_ylabel("iou_top10pct")
    ax.grid(alpha=0.2)
    save_figure(fig, out["png"] / "case_peak_lag_vs_iou_scatter.png", cfg)

    bias_df = metrics_df[["DisNo", "event_total_rain_bias", "iou_top10pct"]].copy() if not metrics_df.empty else pd.DataFrame(columns=["DisNo", "event_total_rain_bias", "iou_top10pct"])
    write_plot_data(bias_df, out["csv"] / "case_total_rain_bias_vs_overlap_scatter_plot_data.csv", cfg)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    if not bias_df.empty:
        ax.scatter(bias_df["event_total_rain_bias"], bias_df["iou_top10pct"], alpha=0.8)
        for _, row in bias_df.iterrows():
            ax.annotate(str(row["DisNo"]), (row["event_total_rain_bias"], row["iou_top10pct"]), fontsize=7)
    ax.set_title("Total Rain Bias vs Overlap")
    ax.set_xlabel("event_total_rain_bias")
    ax.set_ylabel("iou_top10pct")
    ax.grid(alpha=0.2)
    save_figure(fig, out["png"] / "case_total_rain_bias_vs_overlap_scatter.png", cfg)

    has_fail = bool((metrics_df["status"] == "FAIL").any()) if not metrics_df.empty else False
    has_warn = bool((metrics_df["status"] == "WARN").any() or len(metrics_df) < 3)
    stage_status = finalize_stage_status(has_fail=has_fail, has_warn=has_warn)
    if metrics_df.empty:
        stage_status = "WARN"
    status_payload = {
        "level": "L7",
        "status": stage_status,
        "selected_cases": int(len(metrics_df)),
        "target_case_count": target_case_count,
        "status_counts": status_counts,
        "warn_reason_insufficient_cases": bool(len(metrics_df) < 3),
    }
    write_json(status_payload, out["root"] / "l7_status.json", cfg)

    summary_lines = [
        "# L7 IMERG Case Validation Summary",
        "",
        f"- selected_cases: {len(metrics_df)}",
        f"- target_case_count: {target_case_count}",
        f"- stage_status: **{stage_status}**",
        f"- PASS: {status_counts['PASS']}",
        f"- WARN: {status_counts['WARN']}",
        f"- FAIL: {status_counts['FAIL']}",
        "",
        "## Case Metrics",
        "",
        ("```text\n" + metrics_df.to_string(index=False) + "\n```") if not metrics_df.empty else "_no rows_",
    ]
    write_text("\n".join(summary_lines) + "\n", out["root"] / "case_validation_summary.md", cfg)

    artifacts = [
        out["root"] / "case_catalog.csv",
        out["root"] / "case_metrics.csv",
        out["root"] / "case_validation_summary.md",
        out["root"] / "l7_status.json",
    ] + list(out["png"].glob("*.png")) + list(out["csv"].glob("*plot_data.csv"))
    artifacts += [Path(p) for p in case_artifacts]
    return StageResult(
        level="L7",
        status=stage_status,
        fail_count=int(status_counts["FAIL"]),
        warn_count=int(status_counts["WARN"] + (1 if len(metrics_df) < 3 else 0)),
        metrics={
            "selected_cases": int(len(metrics_df)),
            "pass_cases": int(status_counts["PASS"]),
            "warn_cases": int(status_counts["WARN"]),
            "fail_cases": int(status_counts["FAIL"]),
        },
        artifacts=[str(path) for path in artifacts if Path(path).exists()],
        notes=["selection_rule=L_usd_desc_then_match_grade_then_disno"],
    )
