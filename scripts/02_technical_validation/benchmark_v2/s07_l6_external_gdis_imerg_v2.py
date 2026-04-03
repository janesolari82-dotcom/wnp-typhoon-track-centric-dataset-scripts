from __future__ import annotations

import importlib.util
import sys
from collections import OrderedDict
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from benchmark.s00_core_common import RunConfig, StageResult, open_netcdf_readonly, write_csv, write_json, write_text  # noqa: E402
from s00_common_v2 import (  # noqa: E402
    CORE_EXT_START,
    GDIS_EXT_END,
    IMERG_EXT_END,
    build_imerg_index,
    corr_safe,
    create_results_layout_v2,
    extract_event_points,
    footprint_iou_recall,
    load_imerg_meta,
    load_project_events_from_audit,
    quantile_safe,
    rmse_safe,
    safe_ratio,
    sample_imerg_fields,
    save_figure,
    split_external_windows,
    split_groups,
    stage_dirs,
    write_plot_data,
)

OLD_L67_DIR = PARENT_DIR / "benchmark" / "L6-7"
if str(OLD_L67_DIR) not in sys.path:
    sys.path.insert(0, str(OLD_L67_DIR))


def _load_legacy_l6_module():
    path = OLD_L67_DIR / "l6_external_validation.py"
    spec = importlib.util.spec_from_file_location("legacy_l6_external_validation_mod", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load legacy module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LEGACY_L6 = _load_legacy_l6_module()

GDIS_DISASTERNO_MATCH_FAIL = 0.60
GDIS_PAIR_MATCH_FAIL = 0.40
GDIS_HIT_WARN = 0.50
GDIS_P95_DISTANCE_WARN_KM = 500.0

IMERG_COVERAGE_FAIL = 0.95
IMERG_COVERAGE_WARN = 1.00
IMERG_TOP10_FAIL = 0.05
IMERG_TOP20_WARN = 0.15
IMERG_CORR_WARN = 0.30


class IMERGDatasetCache:
    def __init__(self, max_open: int = 16) -> None:
        self.max_open = int(max_open)
        self.cache: "OrderedDict[Path, Dataset]" = OrderedDict()

    def get(self, path: Path) -> Dataset:
        path = Path(path)
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        ds = Dataset(path, "r")
        self.cache[path] = ds
        if len(self.cache) > self.max_open:
            _, old = self.cache.popitem(last=False)
            old.close()
        return ds

    def close(self) -> None:
        for ds in self.cache.values():
            try:
                ds.close()
            except Exception:
                pass
        self.cache.clear()


def _prepare_output_dirs(results_root: Path, cfg: RunConfig) -> dict[str, Path]:
    create_results_layout_v2(cfg)
    return stage_dirs(cfg, "L6")


def _compute_gdis_block(cfg: RunConfig, core_years: list[int], out: dict[str, Path]):
    if not core_years:
        empty_match = pd.DataFrame(
            columns=[
                "year",
                "project_event_count",
                "gdis_storm_disasterno_count",
                "gdis_storm_pair_count",
                "gdis_disasterno_match_count",
                "gdis_pair_match_count",
                "gdis_disasterno_match_rate",
                "gdis_pair_match_rate",
            ]
        )
        empty_geom = pd.DataFrame(columns=["year", "DisNo", "ISO", "gdis_key", "distance_km", "gdis_hit", "point_count", "selected_groups_count"])
        return empty_match, empty_geom, {"status": "WARN", "reasons": ["no_core_years"], "metrics": {}, "project_events": pd.DataFrame()}

    project_events, missing_inputs = load_project_events_from_audit(cfg, core_years)
    gdis_gpkg = cfg.raw_using_root / "gdis" / "pend-gdis-1960-2018-disasterlocations.gpkg"
    gdis_storm = LEGACY_L6.load_gdis_storm(gdis_gpkg, core_years)
    gdis_disno_map, gdis_pair_map = LEGACY_L6.build_gdis_geom_maps(gdis_storm)
    match_df = LEGACY_L6.compute_gdis_match_by_year(project_events, gdis_storm)
    geom_df = LEGACY_L6.compute_gdis_geometry_validation(cfg, project_events, gdis_disno_map, gdis_pair_map, sample_size=200)

    write_csv(match_df, out["root"] / "gdis_match_metrics_by_year.csv", cfg)
    write_csv(geom_df, out["root"] / "gdis_geometry_validation.csv", cfg)
    LEGACY_L6._plot_gdis(match_df, geom_df, out, cfg)

    fail_reasons = []
    warn_reasons = []
    all_row = match_df[match_df["year"] == "ALL"]
    if not all_row.empty:
        disno_rate = float(all_row["gdis_disasterno_match_rate"].iloc[0])
        pair_rate = float(all_row["gdis_pair_match_rate"].iloc[0])
        if np.isfinite(disno_rate) and disno_rate < GDIS_DISASTERNO_MATCH_FAIL:
            fail_reasons.append("gdis_disasterno_match_rate_below_threshold")
        if np.isfinite(pair_rate) and pair_rate < GDIS_PAIR_MATCH_FAIL:
            fail_reasons.append("gdis_pair_match_rate_below_threshold")
    else:
        disno_rate = float("nan")
        pair_rate = float("nan")
        fail_reasons.append("missing_gdis_match_metrics")

    if not geom_df.empty:
        hit_rate = float(np.nanmean(pd.to_numeric(geom_df["gdis_hit"], errors="coerce")))
        p95_dist = float(np.nanquantile(pd.to_numeric(geom_df["distance_km"], errors="coerce"), 0.95))
        if np.isfinite(hit_rate) and hit_rate < GDIS_HIT_WARN:
            warn_reasons.append("gdis_geometry_hit_rate_low")
        if np.isfinite(p95_dist) and p95_dist > GDIS_P95_DISTANCE_WARN_KM:
            warn_reasons.append("gdis_geometry_distance_p95_high")
    else:
        hit_rate = float("nan")
        p95_dist = float("nan")
        warn_reasons.append("gdis_geometry_sample_empty")

    if missing_inputs:
        fail_reasons.append("missing_gdis_project_inputs")

    status = "FAIL" if fail_reasons else ("WARN" if warn_reasons else "PASS")
    summary_lines = [
        "# GDIS Validation Summary",
        "",
        f"- external_window: {CORE_EXT_START}-{GDIS_EXT_END}",
        f"- stage_status: **{status}**",
        f"- project_events: {len(project_events)}",
        f"- gdis_disasterno_match_rate: {disno_rate}",
        f"- gdis_pair_match_rate: {pair_rate}",
        f"- gdis_geometry_hit_rate: {hit_rate}",
        f"- gdis_geometry_distance_km_p95: {p95_dist}",
        f"- fail_reasons: {';'.join(fail_reasons) if fail_reasons else 'none'}",
        f"- warn_reasons: {';'.join(warn_reasons) if warn_reasons else 'none'}",
    ]
    write_text("\n".join(summary_lines) + "\n", out["root"] / "gdis_validation_summary.md", cfg)
    return match_df, geom_df, {
        "status": status,
        "fail_reasons": fail_reasons,
        "warn_reasons": warn_reasons,
        "metrics": {
            "gdis_disasterno_match_rate": disno_rate,
            "gdis_pair_match_rate": pair_rate,
            "gdis_geometry_hit_rate": hit_rate,
            "gdis_geometry_distance_km_p95": p95_dist,
        },
        "project_events": project_events,
    }


def _iter_all_track_points(ds) -> list[tuple[str, int, date]]:
    rows = []
    for group_name, grp in ds.groups.items():
        if "time" not in grp.variables:
            continue
        from netCDF4 import num2date

        time_values = np.asarray(grp.variables["time"][:], dtype=np.float64)
        time_units = str(grp.variables["time"].units)
        dts = num2date(time_values, units=time_units, calendar="standard")
        for idx, dt in enumerate(dts):
            rows.append((group_name, idx, date(int(dt.year), int(dt.month), int(dt.day))))
    return rows


def _sample_group_project_imerg(grp, time_idx: int, imerg_path: Path, imerg_meta, topqs: tuple[float, float], ds_cache: IMERGDatasetCache | None = None) -> dict[str, Any]:
    center_lat = float(grp.variables["center_lat"][time_idx])
    center_lon = float(grp.variables["center_lon"][time_idx])
    window_lat = np.asarray(grp.variables["window_lat"][:], dtype=float)
    window_lon = np.asarray(grp.variables["window_lon"][:], dtype=float)
    lat_grid = np.broadcast_to(center_lat + window_lat[:, None], (window_lat.size, window_lon.size))
    lon_grid = np.broadcast_to(center_lon + window_lon[None, :], (window_lat.size, window_lon.size))
    ds_obj = ds_cache.get(imerg_path) if ds_cache is not None else imerg_path
    fields = sample_imerg_fields(ds_obj, lat_grid, lon_grid, imerg_meta)
    project_rain = np.asarray(grp.variables["hazard_rain_daily"][time_idx, :, :], dtype=float)
    imerg_rain = fields["precipitation"]
    valid = np.isfinite(project_rain) & np.isfinite(imerg_rain)
    valid_coverage_ratio = float(np.count_nonzero(valid) / valid.size) if valid.size else float("nan")
    cnt_cond = fields["precipitation_cnt_cond"]
    if np.isfinite(cnt_cond).any():
        cnt_valid = np.isfinite(cnt_cond) & (cnt_cond > 0)
        valid_coverage_ratio = float(np.count_nonzero(cnt_valid) / cnt_valid.size)
    return {
        "project_rain": project_rain,
        "imerg_rain": imerg_rain,
        "valid_coverage_ratio": valid_coverage_ratio,
        "random_error_p50": quantile_safe(fields["randomError"].reshape(-1), 0.50),
        "random_error_p95": quantile_safe(fields["randomError"].reshape(-1), 0.95),
        "imerg_hazard_rain_corr": corr_safe(project_rain.reshape(-1), imerg_rain.reshape(-1)),
        "imerg_hazard_rain_bias": float(np.nanmean(project_rain - imerg_rain)) if np.isfinite(project_rain - imerg_rain).any() else float("nan"),
        "imerg_hazard_rain_rmse": float(np.sqrt(np.nanmean((project_rain - imerg_rain) ** 2))) if np.isfinite(project_rain - imerg_rain).any() else float("nan"),
        "imerg_top10pct_overlap_ratio": common_overlap(project_rain, imerg_rain, topqs[0]),
        "imerg_top20pct_overlap_ratio": common_overlap(project_rain, imerg_rain, topqs[1]),
    }


def common_overlap(project_rain: np.ndarray, imerg_rain: np.ndarray, q: float) -> float:
    valid = np.isfinite(project_rain) & np.isfinite(imerg_rain)
    if np.count_nonzero(valid) == 0:
        return float("nan")
    pa = project_rain[valid]
    pb = imerg_rain[valid]
    ta = float(np.nanquantile(pa, 1.0 - q))
    tb = float(np.nanquantile(pb, 1.0 - q))
    ma = pa >= ta
    mb = pb >= tb
    union = int(np.count_nonzero(ma | mb))
    inter = int(np.count_nonzero(ma & mb))
    return float(inter / union) if union > 0 else float("nan")


def _compute_sample_metrics_for_year(
    cfg: RunConfig,
    year: int,
    imerg_index: dict[date, Path],
    imerg_meta,
    sample_count: int,
    topqs: tuple[float, float],
) -> pd.DataFrame:
    final_path = cfg.reproduction_root / "04_final_output" / "emdat_attribution_integration" / f"typhoon_{year}_ocean_litpop_poplight_emdat.nc"
    if not final_path.exists():
        return pd.DataFrame()
    rng = np.random.default_rng(cfg.seed + year)
    rows = []
    ds_cache = IMERGDatasetCache()
    with open_netcdf_readonly(final_path, cfg) as ds:
        pairs = _iter_all_track_points(ds)
        if len(pairs) > sample_count:
            sel = rng.choice(len(pairs), size=sample_count, replace=False)
            pairs = [pairs[int(i)] for i in sel]
        for group_name, time_idx, current_date in pairs:
            grp = ds.groups[group_name]
            if current_date not in imerg_index:
                rows.append(
                    {
                        "year": year,
                        "group": group_name,
                        "time_idx": time_idx,
                        "date": current_date.isoformat(),
                        "file_available": 0,
                        "valid_coverage_ratio": np.nan,
                        "random_error_p50": np.nan,
                        "random_error_p95": np.nan,
                        "imerg_hazard_rain_corr": np.nan,
                        "imerg_hazard_rain_bias": np.nan,
                        "imerg_hazard_rain_rmse": np.nan,
                        "imerg_top10pct_overlap_ratio": np.nan,
                        "imerg_top20pct_overlap_ratio": np.nan,
                    }
                )
                continue
            metrics = _sample_group_project_imerg(grp, time_idx, imerg_index[current_date], imerg_meta, topqs, ds_cache=ds_cache)
            rows.append(
                {
                    "year": year,
                    "group": group_name,
                    "time_idx": time_idx,
                    "date": current_date.isoformat(),
                    "file_available": 1,
                    **metrics,
                }
            )
    ds_cache.close()
    return pd.DataFrame(rows)


def _compute_imerg_coverage(index: dict[date, Path], years: list[int]) -> pd.DataFrame:
    rows = []
    for year in years:
        leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        expected_days = 366 if leap else 365
        year_files = [path for dt, path in index.items() if dt.year == year]
        rows.append(
            {
                "year": year,
                "expected_days": expected_days,
                "available_days": len(year_files),
                "imerg_daily_file_coverage_rate": safe_ratio(len(year_files), expected_days),
                "readable_days": len(year_files),
                "imerg_variable_readability_rate": 1.0 if year_files else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def _event_day_grids(ds, selected_groups: list[str], pad_start: date | None, pad_end: date | None, imerg_index, imerg_meta):
    from netCDF4 import num2date

    project_by_day: dict[date, list[np.ndarray]] = OrderedDict()
    imerg_by_day: dict[date, list[np.ndarray]] = OrderedDict()

    ds_cache = IMERGDatasetCache()
    for group_name in selected_groups:
        grp = ds.groups.get(group_name)
        if grp is None or "time" not in grp.variables:
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
            if current_date not in imerg_index:
                continue
            project_grid = np.asarray(grp.variables["hazard_rain_daily"][idx, :, :], dtype=float)
            metrics = _sample_group_project_imerg(grp, idx, imerg_index[current_date], imerg_meta, (0.10, 0.20), ds_cache=ds_cache)
            imerg_grid = metrics["imerg_rain"]
            project_by_day.setdefault(current_date, []).append(project_grid)
            imerg_by_day.setdefault(current_date, []).append(imerg_grid)

    day_rows = []
    agg_project = []
    agg_imerg = []
    for d in sorted(project_by_day.keys()):
        pmean = np.nanmean(np.stack(project_by_day[d], axis=0), axis=0)
        imean = np.nanmean(np.stack(imerg_by_day[d], axis=0), axis=0)
        day_rows.append(
            {
                "date": d,
                "project_total_rain": float(np.nansum(pmean)),
                "imerg_total_rain": float(np.nansum(imean)),
                "project_grid": pmean,
                "imerg_grid": imean,
            }
        )
        agg_project.append(pmean)
        agg_imerg.append(imean)
    agg_project_grid = np.nansum(np.stack(agg_project, axis=0), axis=0) if agg_project else None
    agg_imerg_grid = np.nansum(np.stack(agg_imerg, axis=0), axis=0) if agg_imerg else None
    ds_cache.close()
    return day_rows, agg_project_grid, agg_imerg_grid


def _compute_imerg_event_metrics(cfg: RunConfig, project_events: pd.DataFrame, imerg_index, imerg_meta, topq: float) -> pd.DataFrame:
    rows = []
    work_df = project_events.copy()
    if "L_usd" in work_df.columns:
        work_df = work_df[pd.to_numeric(work_df["L_usd"], errors="coerce").fillna(0.0) > 0].copy()
    for _, row in work_df.iterrows():
        year = int(row["year"])
        if year > IMERG_EXT_END:
            continue
        final_nc = Path(str(row["final_nc"]))
        if not final_nc.exists():
            continue
        selected_groups = split_groups(row.get("selected_groups", ""))
        if not selected_groups:
            continue
        pad_start = row.get("PadStart")
        pad_end = row.get("PadEnd")
        with open_netcdf_readonly(final_nc, cfg) as ds:
            day_rows, agg_project, agg_imerg = _event_day_grids(ds, selected_groups, pad_start, pad_end, imerg_index, imerg_meta)
        if not day_rows or agg_project is None or agg_imerg is None:
            continue
        daily_df = pd.DataFrame(
            {
                "date": [item["date"] for item in day_rows],
                "project_total_rain": [item["project_total_rain"] for item in day_rows],
                "imerg_total_rain": [item["imerg_total_rain"] for item in day_rows],
            }
        )
        event_total_project = float(daily_df["project_total_rain"].sum())
        event_total_imerg = float(daily_df["imerg_total_rain"].sum())
        peak_project_date = pd.to_datetime(daily_df.loc[daily_df["project_total_rain"].idxmax(), "date"]).date()
        peak_imerg_date = pd.to_datetime(daily_df.loc[daily_df["imerg_total_rain"].idxmax(), "date"]).date()
        peak_lag_days = float((peak_project_date - peak_imerg_date).days)
        iou, recall = footprint_iou_recall(agg_project.reshape(-1), agg_imerg.reshape(-1), topq)
        rows.append(
            {
                "year": year,
                "DisNo": str(row["DisNo"]),
                "ISO": str(row["ISO"]),
                "L_usd": float(row["L_usd"]),
                "status": str(row["status"]),
                "match_grade": str(row["match_grade"]),
                "selected_groups": str(row["selected_groups"]),
                "event_total_project_rain": event_total_project,
                "event_total_imerg_rain": event_total_imerg,
                "event_total_rain_bias": float(event_total_project - event_total_imerg),
                "imerg_event_peak_day_lag": peak_lag_days,
                "imerg_event_topq_footprint_iou": iou,
                "imerg_event_topq_footprint_recall": recall,
                "daily_points": int(len(daily_df)),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["imerg_event_total_rainfall_corr"] = corr_safe(out["event_total_project_rain"], out["event_total_imerg_rain"])
        return out.sort_values(["year", "DisNo"]).reset_index(drop=True)
    return out


def _plot_imerg(coverage_df: pd.DataFrame, sample_df: pd.DataFrame, event_df: pd.DataFrame, out: dict[str, Path], cfg: RunConfig) -> None:
    write_plot_data(coverage_df, out["csv"] / "imerg_coverage_by_year_plot_data.csv", cfg)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(coverage_df["year"], coverage_df["imerg_daily_file_coverage_rate"], color="tab:blue", alpha=0.7, label="coverage")
    ax.plot(coverage_df["year"], coverage_df["imerg_variable_readability_rate"], color="tab:orange", marker="o", label="readability")
    ax.set_title("IMERG Coverage by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.2)
    ax.legend()
    save_figure(fig, out["png"] / "imerg_coverage_by_year.png", cfg)

    conv_df = (
        sample_df.groupby("year", as_index=False)
        .agg(
            imerg_hazard_rain_corr=("imerg_hazard_rain_corr", "median"),
            imerg_hazard_rain_bias=("imerg_hazard_rain_bias", "mean"),
            imerg_hazard_rain_rmse=("imerg_hazard_rain_rmse", "mean"),
            imerg_top10pct_overlap_ratio=("imerg_top10pct_overlap_ratio", "median"),
            imerg_top20pct_overlap_ratio=("imerg_top20pct_overlap_ratio", "median"),
            random_error_p50=("random_error_p50", "median"),
            random_error_p95=("random_error_p95", "median"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )
    write_plot_data(conv_df, out["csv"] / "imerg_rainfall_convergence_trend_plot_data.csv", cfg)

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9), sharex=True)
    if not conv_df.empty:
        axes[0].plot(conv_df["year"], conv_df["imerg_hazard_rain_corr"], marker="o")
        axes[1].plot(conv_df["year"], conv_df["imerg_hazard_rain_bias"], marker="o")
        axes[2].plot(conv_df["year"], conv_df["imerg_hazard_rain_rmse"], marker="o")
    axes[0].set_ylabel("corr")
    axes[1].set_ylabel("bias")
    axes[2].set_ylabel("rmse")
    axes[2].set_xlabel("Year")
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].set_title("IMERG Rainfall Convergence Trend")
    save_figure(fig, out["png"] / "imerg_rainfall_convergence_trend.png", cfg)

    write_plot_data(sample_df[["year", "imerg_top10pct_overlap_ratio", "imerg_top20pct_overlap_ratio"]], out["csv"] / "imerg_overlap_distribution_plot_data.csv", cfg)
    fig, ax = plt.subplots(figsize=(10, 5))
    data = []
    labels = []
    for year in sorted(sample_df["year"].dropna().astype(int).unique().tolist()):
        vals = pd.to_numeric(sample_df.loc[sample_df["year"] == year, "imerg_top10pct_overlap_ratio"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size:
            data.append(vals)
            labels.append(str(year))
    if data:
        ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title("IMERG Top10% Overlap Distribution")
    ax.set_xlabel("Year")
    ax.set_ylabel("Overlap ratio")
    ax.grid(alpha=0.2)
    save_figure(fig, out["png"] / "imerg_overlap_distribution.png", cfg)

    write_plot_data(event_df[["year", "DisNo", "imerg_event_peak_day_lag"]], out["csv"] / "imerg_event_peak_day_lag_distribution_plot_data.csv", cfg)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    data = []
    labels = []
    for year in sorted(event_df["year"].dropna().astype(int).unique().tolist()):
        vals = pd.to_numeric(event_df.loc[event_df["year"] == year, "imerg_event_peak_day_lag"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size:
            data.append(vals)
            labels.append(str(year))
    if data:
        ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title("IMERG Event Peak-day Lag Distribution")
    ax.set_xlabel("Year")
    ax.set_ylabel("Peak lag (days)")
    ax.grid(alpha=0.2)
    save_figure(fig, out["png"] / "imerg_event_peak_day_lag_distribution.png", cfg)

    err_df = conv_df[["year", "random_error_p50", "random_error_p95"]].copy() if not conv_df.empty else pd.DataFrame(
        columns=["year", "random_error_p50", "random_error_p95"]
    )
    write_plot_data(err_df, out["csv"] / "imerg_random_error_diagnostics_plot_data.csv", cfg)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if not err_df.empty:
        ax.plot(err_df["year"], err_df["random_error_p50"], marker="o", label="p50")
        ax.plot(err_df["year"], err_df["random_error_p95"], marker="o", label="p95")
    ax.set_title("IMERG Random Error Diagnostics")
    ax.set_xlabel("Year")
    ax.set_ylabel("randomError")
    ax.grid(alpha=0.2)
    ax.legend()
    save_figure(fig, out["png"] / "imerg_random_error_diagnostics.png", cfg)

    write_plot_data(event_df[["year", "DisNo", "event_total_project_rain", "event_total_imerg_rain"]], out["csv"] / "imerg_event_total_rainfall_scatter_plot_data.csv", cfg)
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    if not event_df.empty:
        sc = ax.scatter(event_df["event_total_project_rain"], event_df["event_total_imerg_rain"], c=event_df["year"], cmap="viridis", alpha=0.8)
        lo = float(min(event_df["event_total_project_rain"].min(), event_df["event_total_imerg_rain"].min()))
        hi = float(max(event_df["event_total_project_rain"].max(), event_df["event_total_imerg_rain"].max()))
        ax.plot([lo, hi], [lo, hi], color="tab:red", linestyle="--")
        fig.colorbar(sc, ax=ax, label="Year")
    ax.set_title("Project vs IMERG Event Total Rainfall")
    ax.set_xlabel("Project total rain")
    ax.set_ylabel("IMERG total rain")
    ax.grid(alpha=0.2)
    save_figure(fig, out["png"] / "imerg_event_total_rainfall_scatter.png", cfg)


def run_l6_v2(cfg: RunConfig, imerg_samples_per_year: int = 300, imerg_topq: tuple[float, float] = (0.10, 0.20)) -> tuple[StageResult, dict[str, Any]]:
    out = _prepare_output_dirs(cfg.results_root, cfg)
    core_years, imerg_only_years, not_eval_years = split_external_windows(cfg.years)

    gdis_match_df, gdis_geom_df, gdis_status = _compute_gdis_block(cfg, core_years, out)

    imerg_root = cfg.raw_using_root / "gpm imerg" / "data"
    imerg_index = build_imerg_index(imerg_root)
    imerg_meta = load_imerg_meta(imerg_index)
    imerg_eval_years = sorted(core_years + imerg_only_years)
    coverage_df = _compute_imerg_coverage(imerg_index, imerg_eval_years)

    sample_frames = []
    for year in imerg_eval_years:
        sample_frames.append(_compute_sample_metrics_for_year(cfg, year, imerg_index, imerg_meta, imerg_samples_per_year, imerg_topq))
    sample_df = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()

    project_events_imerg, missing_inputs_imerg = load_project_events_from_audit(cfg, imerg_eval_years)
    event_df = _compute_imerg_event_metrics(cfg, project_events_imerg, imerg_index, imerg_meta, topq=imerg_topq[0])

    write_csv(coverage_df, out["root"] / "imerg_coverage_by_year.csv", cfg)
    sample_year_df = (
        sample_df.groupby("year", as_index=False)
        .agg(
            valid_coverage_ratio=("valid_coverage_ratio", "median"),
            imerg_hazard_rain_corr=("imerg_hazard_rain_corr", "median"),
            imerg_hazard_rain_bias=("imerg_hazard_rain_bias", "mean"),
            imerg_hazard_rain_rmse=("imerg_hazard_rain_rmse", "mean"),
            imerg_top10pct_overlap_ratio=("imerg_top10pct_overlap_ratio", "median"),
            imerg_top20pct_overlap_ratio=("imerg_top20pct_overlap_ratio", "median"),
            random_error_p50=("random_error_p50", "median"),
            random_error_p95=("random_error_p95", "median"),
        )
        .sort_values("year")
        .reset_index(drop=True)
        if not sample_df.empty
        else pd.DataFrame()
    )
    write_csv(sample_year_df, out["root"] / "imerg_rainfall_convergence_by_year.csv", cfg)
    write_csv(event_df, out["root"] / "imerg_event_scale_metrics.csv", cfg)
    write_csv(pd.DataFrame({"year": not_eval_years, "reason": ["Not-Evaluable by current external datasets"] * len(not_eval_years)}), out["root"] / "not_evaluable_years_2021_2024.csv", cfg)

    _plot_imerg(coverage_df, sample_df, event_df, out, cfg)

    fail_reasons = []
    warn_reasons = []
    core_cov = coverage_df[coverage_df["year"].isin(core_years)].copy()
    if not core_cov.empty:
        if (core_cov["imerg_daily_file_coverage_rate"] < IMERG_COVERAGE_FAIL).any():
            fail_reasons.append("imerg_daily_file_coverage_rate_low")
        elif (core_cov["imerg_daily_file_coverage_rate"] < IMERG_COVERAGE_WARN).any():
            warn_reasons.append("imerg_daily_file_coverage_partial")
        if (core_cov["imerg_variable_readability_rate"] < 1.0).any():
            fail_reasons.append("imerg_variable_readability_rate_low")
    else:
        fail_reasons.append("imerg_core_coverage_missing")

    if not sample_year_df.empty:
        core_conv = sample_year_df[sample_year_df["year"].isin(core_years)]
        med_corr = quantile_safe(core_conv["imerg_hazard_rain_corr"], 0.5)
        med_top10 = quantile_safe(core_conv["imerg_top10pct_overlap_ratio"], 0.5)
        med_top20 = quantile_safe(core_conv["imerg_top20pct_overlap_ratio"], 0.5)
        if np.isfinite(med_corr) and med_corr <= 0 and np.isfinite(med_top10) and med_top10 < IMERG_TOP10_FAIL:
            fail_reasons.append("imerg_rainfall_convergence_failure")
        if np.isfinite(med_corr) and 0 < med_corr < IMERG_CORR_WARN:
            warn_reasons.append("imerg_rainfall_corr_low")
        if np.isfinite(med_top20) and med_top20 < IMERG_TOP20_WARN:
            warn_reasons.append("imerg_top20_overlap_low")
    else:
        med_corr = med_top10 = med_top20 = float("nan")
        warn_reasons.append("imerg_sample_metrics_empty")

    if not event_df.empty:
        event_corr = float(event_df["imerg_event_total_rainfall_corr"].iloc[0])
        med_peak_lag = quantile_safe(np.abs(event_df["imerg_event_peak_day_lag"]), 0.5)
        if np.isfinite(event_corr) and event_corr <= 0 and np.isfinite(med_peak_lag) and med_peak_lag > 3:
            fail_reasons.append("imerg_event_scale_failure")
        if np.isfinite(event_corr) and 0 < event_corr < IMERG_CORR_WARN:
            warn_reasons.append("imerg_event_total_rainfall_corr_low")
        if np.isfinite(med_peak_lag) and med_peak_lag > 1:
            warn_reasons.append("imerg_event_peak_day_lag_high")
    else:
        event_corr = med_peak_lag = float("nan")
        warn_reasons.append("imerg_event_metrics_empty")

    if missing_inputs_imerg:
        fail_reasons.append("missing_imerg_project_inputs")
    if imerg_only_years:
        warn_reasons.append("extended_rainfall_only_years_present")
    if not_eval_years:
        warn_reasons.append("years_not_evaluable_by_current_external_datasets")

    l6b_status = "FAIL" if fail_reasons else ("WARN" if warn_reasons else "PASS")
    overall_fail = bool(gdis_status["status"] == "FAIL" or l6b_status == "FAIL")
    overall_warn = bool(gdis_status["status"] == "WARN" or l6b_status == "WARN" or not_eval_years or imerg_only_years)
    overall_status = "FAIL" if overall_fail else ("WARN" if overall_warn else "PASS")

    summary_lines = [
        "# L6 External Validation Summary (GDIS + IMERG)",
        "",
        f"- core_external_years: {core_years}",
        f"- extended_rainfall_only_years: {imerg_only_years}",
        f"- not_evaluable_years: {not_eval_years}",
        f"- L6-A status: **{gdis_status['status']}**",
        f"- L6-B status: **{l6b_status}**",
        f"- L6 overall status: **{overall_status}**",
        f"- L6-B median_hazard_rain_corr: {med_corr}",
        f"- L6-B median_top10_overlap: {med_top10}",
        f"- L6-B event_total_rainfall_corr: {event_corr}",
        "",
        f"- L6-A fail_reasons: {';'.join(gdis_status.get('fail_reasons', [])) if gdis_status.get('fail_reasons') else 'none'}",
        f"- L6-A warn_reasons: {';'.join(gdis_status.get('warn_reasons', [])) if gdis_status.get('warn_reasons') else 'none'}",
        f"- L6-B fail_reasons: {';'.join(fail_reasons) if fail_reasons else 'none'}",
        f"- L6-B warn_reasons: {';'.join(warn_reasons) if warn_reasons else 'none'}",
    ]
    write_text("\n".join(summary_lines) + "\n", out["md"] / "l6_validation_summary.md", cfg)

    status_payload = {
        "level": "L6",
        "status": overall_status,
        "L6A_status": gdis_status["status"],
        "L6B_status": l6b_status,
        "core_external_years": core_years,
        "extended_rainfall_only_years": imerg_only_years,
        "not_evaluable_years": not_eval_years,
        "gdis_metrics": gdis_status["metrics"],
        "imerg_metrics": {
            "median_hazard_rain_corr": med_corr,
            "median_top10_overlap": med_top10,
            "median_top20_overlap": med_top20,
            "event_total_rainfall_corr": event_corr,
            "median_peak_day_lag": med_peak_lag,
        },
        "fail_reasons": {"L6A": gdis_status.get("fail_reasons", []), "L6B": fail_reasons},
        "warn_reasons": {"L6A": gdis_status.get("warn_reasons", []), "L6B": warn_reasons},
    }
    write_json(status_payload, out["root"] / "l6_status.json", cfg)

    artifacts = [
        out["root"] / "gdis_match_metrics_by_year.csv",
        out["root"] / "gdis_geometry_validation.csv",
        out["root"] / "imerg_coverage_by_year.csv",
        out["root"] / "imerg_rainfall_convergence_by_year.csv",
        out["root"] / "imerg_event_scale_metrics.csv",
        out["md"] / "l6_validation_summary.md",
        out["root"] / "l6_status.json",
    ] + list(out["png"].glob("*.png")) + list(out["csv"].glob("*plot_data.csv"))
    result = StageResult(
        level="L6",
        status=overall_status,
        fail_count=int(overall_fail),
        warn_count=int(bool(overall_warn)),
        metrics=status_payload["imerg_metrics"] | status_payload["gdis_metrics"],
        artifacts=[str(path) for path in artifacts if Path(path).exists()],
        notes=[
            f"core_external_years={core_years}",
            f"extended_rainfall_only_years={imerg_only_years}",
            f"not_evaluable_years={not_eval_years}",
            f"imerg_samples_per_year={imerg_samples_per_year}",
        ],
    )
    ctx = {
        "project_events_imerg": project_events_imerg,
        "imerg_event_metrics": event_df,
        "imerg_index": imerg_index,
        "imerg_meta": imerg_meta,
    }
    return result, ctx
