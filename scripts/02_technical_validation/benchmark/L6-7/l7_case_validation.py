from __future__ import annotations

import re
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from core_common import (  # noqa: E402
    RunConfig,
    StageResult,
    open_netcdf_readonly,
    write_csv,
    write_text,
)
from l67_common import (  # noqa: E402
    build_track_geometry,
    ensure_dir,
    extract_event_points,
    finalize_stage_status,
    grade_priority,
    haversine_km,
    parse_date_safe,
    save_figure,
    split_groups,
)

GDIS_HIT_DISTANCE_KM = 25.0
CASE_TOPN = 10


def _prepare_output_dirs(results_root: Path) -> dict[str, Path]:
    root = results_root / "07_case_validation"
    csv_dir = ensure_dir(root / "csv")
    md_dir = ensure_dir(root / "md")
    png_dir = ensure_dir(root / "png")
    ensure_dir(root)
    return {"root": root, "csv": csv_dir, "md": md_dir, "png": png_dir}


def _safe_case_id(disno: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(disno or "").strip())


def _extract_track_series(ds, selected_groups: list[str], pad_start, pad_end) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for g in selected_groups:
        grp = ds.groups.get(g)
        if grp is None or "time" not in grp.variables:
            continue
        tvals = np.asarray(grp.variables["time"][:], dtype=np.float64)
        tunits = str(grp.variables["time"].units)
        from netCDF4 import num2date

        dts = num2date(tvals, units=tunits, calendar="standard")
        hazard_name = "hazard_compound_daily" if "hazard_compound_daily" in grp.variables else "hazard_compound"
        for i, dt in enumerate(dts):
            di = parse_date_safe(f"{int(dt.year):04d}-{int(dt.month):02d}-{int(dt.day):02d}")
            if pad_start and di and di < pad_start:
                continue
            if pad_end and di and di > pad_end:
                continue
            hz = np.asarray(grp.variables[hazard_name][i, :, :], dtype=np.float32)
            peak = float(np.nanmax(hz)) if np.any(np.isfinite(hz)) else float("nan")
            rows.append(
                {
                    "time": di,
                    "center_lat": float(grp.variables["center_lat"][i]),
                    "center_lon": float(grp.variables["center_lon"][i]),
                    "hazard_peak": peak,
                    "group": g,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["time", "center_lat", "center_lon", "hazard_peak", "group"])
    return pd.DataFrame(rows).sort_values(["time", "group"]).reset_index(drop=True)


def _read_gfd_top_points(tif_path: Path, bbox: tuple[float, float, float, float], top_n: int = CASE_TOPN) -> pd.DataFrame:
    if not tif_path.exists():
        return pd.DataFrame(columns=["lon", "lat", "flooded", "duration", "permwater"])

    minx, miny, maxx, maxy = bbox
    pad = 1.0
    rows: list[dict[str, Any]] = []
    try:
        with rasterio.open(tif_path) as ds:
            win = from_bounds(minx - pad, miny - pad, maxx + pad, maxy + pad, transform=ds.transform)
            b1 = ds.read(1, window=win, boundless=True, fill_value=0)
            b2 = ds.read(2, window=win, boundless=True, fill_value=0)
            b5 = ds.read(5, window=win, boundless=True, fill_value=0)
            wtfm = ds.window_transform(win)

            rr, cc = np.where(b1 > 0)
            if rr.size == 0:
                return pd.DataFrame(columns=["lon", "lat", "flooded", "duration", "permwater"])

            duration = b2[rr, cc].astype(float)
            order = np.argsort(-duration)[: int(max(1, top_n))]
            for idx in order:
                r = int(rr[idx])
                c = int(cc[idx])
                lon, lat = rasterio.transform.xy(wtfm, r, c, offset="center")
                rows.append(
                    {
                        "lon": float(lon),
                        "lat": float(lat),
                        "flooded": float(b1[r, c]),
                        "duration": float(b2[r, c]),
                        "permwater": float(b5[r, c]),
                    }
                )
    except Exception:
        return pd.DataFrame(columns=["lon", "lat", "flooded", "duration", "permwater"])

    return pd.DataFrame(rows)


def _compute_iou_recall(project_top: pd.DataFrame, gfd_top: pd.DataFrame, dist_km: float = GDIS_HIT_DISTANCE_KM) -> tuple[float, float]:
    if project_top.empty or gfd_top.empty:
        return float("nan"), float("nan")
    plon = project_top["lon"].to_numpy(dtype=float)
    plat = project_top["lat"].to_numpy(dtype=float)
    glon = gfd_top["lon"].to_numpy(dtype=float)
    glat = gfd_top["lat"].to_numpy(dtype=float)

    hit_cnt = 0
    for x, y in zip(plon, plat):
        d = haversine_km(np.full_like(glon, x), np.full_like(glat, y), glon, glat)
        if np.any(d <= dist_km):
            hit_cnt += 1

    union = len(project_top) + len(gfd_top) - hit_cnt
    iou = float(hit_cnt / union) if union > 0 else float("nan")
    recall = float(hit_cnt / len(project_top)) if len(project_top) > 0 else float("nan")
    return iou, recall


def _case_status(peak_lag_days: float, iou_top10: float, recall_top10: float, gdis_distance_km: float) -> tuple[str, list[str]]:
    reasons: list[str] = []
    has_fail = False
    has_warn = False

    if np.isfinite(gdis_distance_km) and gdis_distance_km > 500:
        has_fail = True
        reasons.append("gdis_distance_too_large")
    if np.isfinite(iou_top10) and np.isfinite(recall_top10) and iou_top10 < 0.05 and recall_top10 < 0.10:
        has_fail = True
        reasons.append("spatial_overlap_too_low")

    if not has_fail:
        if np.isfinite(peak_lag_days) and abs(float(peak_lag_days)) > 3:
            has_warn = True
            reasons.append("peak_lag_gt_3d")
        if np.isfinite(gdis_distance_km) and gdis_distance_km > 100:
            has_warn = True
            reasons.append("gdis_distance_gt_100km")
        if np.isfinite(iou_top10) and iou_top10 < 0.10:
            has_warn = True
            reasons.append("iou_low")

    status = finalize_stage_status(has_fail=has_fail, has_warn=has_warn)
    if not reasons:
        reasons.append("none")
    return status, reasons


def _plot_case_track(track_df: pd.DataFrame, out_png: Path) -> None:
    fig, axs = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
    t = pd.to_datetime(track_df["time"], errors="coerce")
    axs[0].plot(t, track_df["center_lat"], marker="o", linewidth=1.2)
    axs[0].set_ylabel("Latitude")
    axs[0].grid(alpha=0.25)
    axs[1].plot(t, track_df["center_lon"], marker="o", linewidth=1.2, color="tab:orange")
    axs[1].set_ylabel("Longitude")
    axs[1].set_xlabel("Date")
    axs[1].grid(alpha=0.25)
    fig.suptitle("Typhoon Track Time Series")
    save_figure(fig, out_png)


def _plot_case_hazard(points_df: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    sc = ax.scatter(
        points_df["lon"],
        points_df["lat"],
        c=points_df["hazard_compound"],
        s=15,
        cmap="YlOrRd",
        alpha=0.75,
    )
    ax.set_title("Project Hazard Field (Sampled High-Risk Points)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2)
    fig.colorbar(sc, ax=ax, label="hazard_compound")
    save_figure(fig, out_png)


def _plot_case_gfd_compare(project_top: pd.DataFrame, gfd_top: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    if not gfd_top.empty:
        sc = ax.scatter(
            gfd_top["lon"],
            gfd_top["lat"],
            c=gfd_top["duration"],
            s=42,
            cmap="Blues",
            alpha=0.75,
            label="GFD top flooded points",
        )
        fig.colorbar(sc, ax=ax, label="GFD duration")
    if not project_top.empty:
        ax.scatter(
            project_top["lon"],
            project_top["lat"],
            s=52,
            marker="x",
            color="tab:red",
            label="Project top hazard points",
        )
    ax.set_title("GFD Flooded/Duration vs Project Hazard Top10")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="best")
    save_figure(fig, out_png)


def _plot_case_gdis_overlay(track_geom, gdis_geom, project_top: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    if gdis_geom is not None:
        gpd.GeoSeries([gdis_geom], crs="EPSG:4326").boundary.plot(ax=ax, color="tab:blue", linewidth=1.2, label="GDIS geometry")
    if track_geom is not None:
        gpd.GeoSeries([track_geom], crs="EPSG:4326").plot(ax=ax, color="tab:red", linewidth=1.4, label="Project track")
    if not project_top.empty:
        ax.scatter(project_top["lon"], project_top["lat"], s=28, color="tab:orange", alpha=0.8, label="Project top10")
    ax.set_title("GDIS Geometry Overlay")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="best")
    save_figure(fig, out_png)


def run_l7_case_validation(
    cfg: RunConfig,
    l6_context: dict[str, Any],
    case_count: int = 5,
) -> StageResult:
    out = _prepare_output_dirs(cfg.results_root)

    linkage = l6_context.get("event_linkage")
    if linkage is None or not isinstance(linkage, pd.DataFrame) or linkage.empty:
        write_text("# L7 Case Validation Summary\n\nNo linkage context from L6.\n", out["root"] / "case_validation_summary.md", cfg)
        return StageResult(
            level="L7",
            status="FAIL",
            fail_count=1,
            warn_count=0,
            metrics={"cases_selected": 0},
            artifacts=[str(out["root"] / "case_validation_summary.md")],
            notes=["missing_l6_context"],
        )

    cdf = linkage.copy()
    cdf["L_usd"] = pd.to_numeric(cdf.get("L_usd"), errors="coerce").fillna(0.0)
    cdf = cdf[
        (cdf["L_usd"] > 0)
        & ((cdf.get("gdis_disasterno_matched", 0) == 1) | (cdf.get("gdis_pair_matched", 0) == 1))
        & ((cdf.get("gfd_linked_tc", 0) == 1) | (cdf.get("gfd_linked_hydromet", 0) == 1))
    ].copy()

    if cdf.empty:
        write_csv(pd.DataFrame(columns=["DisNo", "ISO", "year"]), out["root"] / "case_catalog.csv", cfg)
        write_csv(pd.DataFrame(columns=["DisNo", "ISO", "year", "status"]), out["root"] / "case_metrics.csv", cfg)
        write_text(
            "# L7 Case Validation Summary\n\nNo candidates matched the required pool (L_usd>0 + GDIS match + GFD link).\n",
            out["root"] / "case_validation_summary.md",
            cfg,
        )
        return StageResult(
            level="L7",
            status="WARN",
            fail_count=0,
            warn_count=1,
            metrics={"cases_selected": 0},
            artifacts=[
                str(out["root"] / "case_catalog.csv"),
                str(out["root"] / "case_metrics.csv"),
                str(out["root"] / "case_validation_summary.md"),
            ],
            notes=["no_case_candidates"],
        )

    cdf["grade_rank"] = cdf["match_grade"].map(grade_priority)
    cdf = cdf.sort_values(["L_usd", "grade_rank", "DisNo"], ascending=[False, True, True]).reset_index(drop=True)
    n_case = int(max(3, min(5, int(case_count))))
    sel = cdf.head(n_case).copy()
    sel = sel.drop(columns=["grade_rank"])
    write_csv(sel, out["root"] / "case_catalog.csv", cfg)

    gdis_disno_map = l6_context.get("gdis_disno_geom_map", {}) or {}
    gdis_pair_map = l6_context.get("gdis_pair_geom_map", {}) or {}
    gfd_catalog = l6_context.get("gfd_catalog")
    gfd_map = {}
    if isinstance(gfd_catalog, pd.DataFrame) and not gfd_catalog.empty:
        gfd_map = {str(r["gfd_event_id"]): r for _, r in gfd_catalog.iterrows()}

    metrics_rows: list[dict[str, Any]] = []
    status_counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    case_artifacts: list[str] = []

    for year, ydf in sel.groupby("year", dropna=False):
        nc_path = Path(str(ydf["final_nc"].iloc[0]))
        if not nc_path.exists():
            for _, r in ydf.iterrows():
                metrics_rows.append(
                    {
                        "year": int(year),
                        "DisNo": str(r["DisNo"]),
                        "ISO": str(r["ISO"]),
                        "peak_lag_days": np.nan,
                        "iou_top10": np.nan,
                        "recall_top10": np.nan,
                        "gdis_hit": 0,
                        "gdis_distance_km": np.nan,
                        "status": "FAIL",
                        "reason_tags": "missing_final_nc",
                    }
                )
                status_counts["FAIL"] += 1
            continue

        with open_netcdf_readonly(nc_path, cfg) as ds:
            for _, r in ydf.iterrows():
                disno = str(r["DisNo"])
                iso = str(r["ISO"])
                disno_base = str(r.get("disasterno_base", ""))
                case_id = _safe_case_id(disno)
                groups = split_groups(r.get("selected_groups", ""))
                p0 = r.get("PadStart")
                p1 = r.get("PadEnd")

                points = extract_event_points(
                    ds,
                    selected_groups=groups,
                    pad_start=p0,
                    pad_end=p1,
                    seed=cfg.seed + int(year),
                    max_points=2000,
                )
                track_df = _extract_track_series(ds, groups, p0, p1)
                track_geom = build_track_geometry(points)
                gdis_geom = gdis_pair_map.get((disno_base, iso), gdis_disno_map.get(disno_base))

                if track_geom is not None and gdis_geom is not None:
                    try:
                        dist_s = gpd.GeoSeries([track_geom, gdis_geom], crs="EPSG:4326").to_crs(3857)
                        gdis_distance_km = float(dist_s.iloc[0].distance(dist_s.iloc[1]) / 1000.0)
                    except Exception:
                        gdis_distance_km = float("nan")
                else:
                    gdis_distance_km = float("nan")
                gdis_hit = int(np.isfinite(gdis_distance_km) and gdis_distance_km <= GDIS_HIT_DISTANCE_KM)

                gfd_id = str(r.get("gfd_event_id_tc", "") or r.get("gfd_event_id_hydromet", ""))
                grec = gfd_map.get(gfd_id)
                if grec is not None:
                    tif_path = Path(str(grec.get("tif_path", "")))
                    gfd_mid = grec.get("began") + (grec.get("ended") - grec.get("began")) / 2
                else:
                    tif_path = Path("")
                    gfd_mid = None

                if not points.empty:
                    top_project = points.sort_values("hazard_compound", ascending=False).head(CASE_TOPN).copy()
                    bbox = (
                        float(points["lon"].min()),
                        float(points["lat"].min()),
                        float(points["lon"].max()),
                        float(points["lat"].max()),
                    )
                else:
                    top_project = pd.DataFrame(columns=["lon", "lat", "hazard_compound"])
                    bbox = (0.0, 0.0, 0.0, 0.0)

                top_gfd = _read_gfd_top_points(tif_path, bbox, top_n=CASE_TOPN) if tif_path.exists() else pd.DataFrame()
                iou_top10, recall_top10 = _compute_iou_recall(top_project[["lon", "lat"]], top_gfd[["lon", "lat"]] if not top_gfd.empty else pd.DataFrame())

                if not track_df.empty:
                    peak_idx = pd.to_numeric(track_df["hazard_peak"], errors="coerce").astype(float).idxmax()
                    project_peak_date = parse_date_safe(track_df.loc[peak_idx, "time"])
                else:
                    project_peak_date = None
                peak_lag_days = float((project_peak_date - gfd_mid).days) if (project_peak_date is not None and gfd_mid is not None) else float("nan")

                case_status, reason_tags = _case_status(peak_lag_days, iou_top10, recall_top10, gdis_distance_km)
                status_counts[case_status] += 1

                metrics_rows.append(
                    {
                        "year": int(year),
                        "DisNo": disno,
                        "ISO": iso,
                        "peak_lag_days": peak_lag_days,
                        "iou_top10": iou_top10,
                        "recall_top10": recall_top10,
                        "gdis_hit": gdis_hit,
                        "gdis_distance_km": gdis_distance_km,
                        "status": case_status,
                        "reason_tags": ";".join(reason_tags),
                        "gfd_event_id": gfd_id,
                        "match_grade": str(r.get("match_grade", "")),
                        "L_usd": float(r.get("L_usd", 0.0)),
                    }
                )

                track_csv = out["csv"] / f"case_{case_id}_track_timeseries.csv"
                hazard_csv = out["csv"] / f"case_{case_id}_hazard_map_data.csv"
                gfd_csv = out["csv"] / f"case_{case_id}_gfd_compare_data.csv"
                gdis_csv = out["csv"] / f"case_{case_id}_gdis_overlay_data.csv"
                write_csv(track_df, track_csv, cfg)
                write_csv(points, hazard_csv, cfg)
                gfd_compare_df = top_gfd.copy()
                if not top_project.empty:
                    for c in ["hazard_compound"]:
                        if c in top_project.columns:
                            top_project[c] = pd.to_numeric(top_project[c], errors="coerce")
                    top_project["source"] = "project_top10"
                if not gfd_compare_df.empty:
                    gfd_compare_df["source"] = "gfd_top10"
                mix_parts = [df for df in [top_project, gfd_compare_df] if isinstance(df, pd.DataFrame) and not df.empty]
                if mix_parts:
                    mix = pd.concat(mix_parts, ignore_index=True, sort=False)
                else:
                    mix = pd.DataFrame(columns=["lon", "lat", "hazard_compound", "flooded", "duration", "permwater", "source"])
                write_csv(mix, gfd_csv, cfg)
                gdis_df = pd.DataFrame()
                if gdis_geom is not None:
                    try:
                        gdf = gpd.GeoDataFrame({"kind": ["gdis"]}, geometry=[gdis_geom], crs="EPSG:4326")
                        bounds = gdf.total_bounds
                        gdis_df = pd.DataFrame(
                            [{"minx": bounds[0], "miny": bounds[1], "maxx": bounds[2], "maxy": bounds[3], "kind": "gdis_bounds"}]
                        )
                    except Exception:
                        gdis_df = pd.DataFrame()
                write_csv(gdis_df, gdis_csv, cfg)

                track_png = out["png"] / f"case_{case_id}_track_timeseries.png"
                hazard_png = out["png"] / f"case_{case_id}_hazard_map.png"
                gfd_png = out["png"] / f"case_{case_id}_gfd_flood_duration.png"
                gdis_png = out["png"] / f"case_{case_id}_gdis_overlay.png"
                if not track_df.empty:
                    _plot_case_track(track_df, track_png)
                if not points.empty:
                    _plot_case_hazard(points, hazard_png)
                _plot_case_gfd_compare(top_project, top_gfd, gfd_png)
                _plot_case_gdis_overlay(track_geom, gdis_geom, top_project, gdis_png)

                report_lines = [
                    f"# Case Validation Report: {disno}",
                    "",
                    f"- year: {int(year)}",
                    f"- ISO: {iso}",
                    f"- L_usd: {float(r.get('L_usd', 0.0)):.2f}",
                    f"- match_grade: {str(r.get('match_grade', ''))}",
                    f"- gfd_event_id: {gfd_id or 'NA'}",
                    "",
                    "## Metrics",
                    "",
                    f"- peak_lag_days: {peak_lag_days:.1f}" if np.isfinite(peak_lag_days) else "- peak_lag_days: nan",
                    f"- iou_top10: {iou_top10:.4f}" if np.isfinite(iou_top10) else "- iou_top10: nan",
                    f"- recall_top10: {recall_top10:.4f}" if np.isfinite(recall_top10) else "- recall_top10: nan",
                    f"- gdis_hit: {gdis_hit}",
                    f"- gdis_distance_km: {gdis_distance_km:.2f}" if np.isfinite(gdis_distance_km) else "- gdis_distance_km: nan",
                    "",
                    f"## Case Status: **{case_status}**",
                    "",
                    f"- reason_tags: {';'.join(reason_tags)}",
                    "",
                    "## Artifacts",
                    "",
                    f"- track_png: {track_png}",
                    f"- hazard_png: {hazard_png}",
                    f"- gfd_png: {gfd_png}",
                    f"- gdis_png: {gdis_png}",
                ]
                report_path = out["root"] / f"case_{case_id}_report.md"
                write_text("\n".join(report_lines) + "\n", report_path, cfg)
                case_artifacts.extend([str(report_path), str(track_png), str(hazard_png), str(gfd_png), str(gdis_png)])

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["year", "DisNo"]).reset_index(drop=True)
    write_csv(metrics_df, out["root"] / "case_metrics.csv", cfg)

    if metrics_df.empty:
        stage_status = "WARN"
        fail_count = 0
        warn_count = 1
    else:
        has_fail = bool((metrics_df["status"] == "FAIL").any())
        has_warn = bool((metrics_df["status"] == "WARN").any() or len(metrics_df) < 3)
        stage_status = finalize_stage_status(has_fail=has_fail, has_warn=has_warn)
        fail_count = int((metrics_df["status"] == "FAIL").sum())
        warn_count = int((metrics_df["status"] == "WARN").sum())
        if len(metrics_df) < 3:
            warn_count += 1

    summary_lines = [
        "# L7 Case Validation Summary",
        "",
        f"- selected_cases: {len(metrics_df)}",
        f"- target_case_count: {int(max(3, min(5, int(case_count))))}",
        f"- stage_status: **{stage_status}**",
        f"- PASS: {status_counts.get('PASS', 0)}",
        f"- WARN: {status_counts.get('WARN', 0)}",
        f"- FAIL: {status_counts.get('FAIL', 0)}",
        "",
        "## Case Metrics",
        "",
        ("```text\n" + metrics_df.to_string(index=False) + "\n```") if not metrics_df.empty else "_no rows_",
        "",
    ]
    if len(metrics_df) < 3:
        summary_lines.append("- note: fewer than 3 valid cases were available, stage kept at least WARN.")
    write_text("\n".join(summary_lines) + "\n", out["root"] / "case_validation_summary.md", cfg)

    return StageResult(
        level="L7",
        status=stage_status,
        fail_count=fail_count,
        warn_count=warn_count,
        metrics={
            "cases_selected": int(len(metrics_df)),
            "pass_cases": int(status_counts.get("PASS", 0)),
            "warn_cases": int(status_counts.get("WARN", 0)),
            "fail_cases": int(status_counts.get("FAIL", 0)),
        },
        artifacts=[
            str(out["root"] / "case_catalog.csv"),
            str(out["root"] / "case_metrics.csv"),
            str(out["root"] / "case_validation_summary.md"),
        ]
        + case_artifacts,
        notes=["selection_rule=L_usd_desc_then_match_grade_then_disno"],
    )
