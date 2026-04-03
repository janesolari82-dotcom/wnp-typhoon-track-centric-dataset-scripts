from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from shapely.ops import unary_union

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from core_common import (  # noqa: E402
    RunConfig,
    StageResult,
    get_stage_paths,
    open_netcdf_readonly,
    write_csv,
    write_json,
    write_text,
)
from l67_common import (  # noqa: E402
    EXTERNAL_YEAR_END,
    EXTERNAL_YEAR_START,
    build_track_geometry,
    ensure_dir,
    extract_event_points,
    finalize_stage_status,
    normalize_cause,
    overlap_days,
    parse_date_safe,
    parse_disno_base,
    parse_iso_set,
    save_figure,
    safe_ratio,
    split_external_years,
    split_groups,
)

STRICT_CAUSES = {"tropical cyclone"}
HYDROMET_CAUSES = {"heavy rain", "monsoonal rain", "torrential rain", "tropical cyclone"}

GDIS_DISASTERNO_MATCH_FAIL = 0.60
GDIS_PAIR_MATCH_FAIL = 0.40
GDIS_HIT_DISTANCE_KM = 25.0
GDIS_HIT_WARN = 0.50
GDIS_P95_DISTANCE_WARN_KM = 500.0

GFD_STRICT_MIN_SAMPLES_WARN = 10
GFD_LINK_RATE_TC_FAIL = 0.01
GFD_OVERLAP_LOW_FAIL = 0.05


def _require_columns(df: pd.DataFrame, cols: list[str], where: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise RuntimeError(f"missing required columns in {where}: {miss}")


def _prepare_output_dirs(results_root: Path) -> dict[str, Path]:
    root = results_root / "06_external_validation"
    csv_dir = ensure_dir(root / "csv")
    md_dir = ensure_dir(root / "md")
    png_dir = ensure_dir(root / "png")
    ensure_dir(root)
    return {"root": root, "csv": csv_dir, "md": md_dir, "png": png_dir}


def load_project_events(cfg: RunConfig, years: list[int]) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []

    for year in years:
        paths = get_stage_paths(cfg, year)
        summary_path = paths["audit_summary"]
        match_path = paths["audit_match"]
        final_nc = paths["final"]

        if not summary_path.exists():
            missing.append({"year": str(year), "missing": "audit_summary", "path": str(summary_path)})
            continue
        if not match_path.exists():
            missing.append({"year": str(year), "missing": "audit_match", "path": str(match_path)})
            continue
        if not final_nc.exists():
            missing.append({"year": str(year), "missing": "final_nc", "path": str(final_nc)})
            continue

        summary_df = pd.read_csv(summary_path)
        match_df = pd.read_csv(match_path)
        _require_columns(summary_df, ["DisNo", "ISO", "L_usd", "status", "match_grade"], f"summary_{year}")
        _require_columns(match_df, ["DisNo", "ISO", "PadStart", "PadEnd", "selected_groups"], f"match_{year}")

        agg = (
            match_df.groupby(["DisNo", "ISO"], as_index=False)
            .agg(
                {
                    "PadStart": "first",
                    "PadEnd": "first",
                    "selected_groups": lambda x: ";".join(
                        sorted({str(v).strip() for v in x if str(v).strip() and str(v).lower() != "nan"})
                    ),
                    "match_grade": "first",
                }
            )
            .copy()
        )
        merged = summary_df.merge(
            agg,
            how="left",
            on=["DisNo", "ISO"],
            suffixes=("_summary", "_match"),
        )
        if "match_grade_summary" in merged.columns:
            merged["match_grade"] = merged["match_grade_summary"].fillna(merged.get("match_grade_match"))
        else:
            merged["match_grade"] = merged.get("match_grade", pd.Series(dtype=str))

        for _, r in merged.iterrows():
            disno = str(r.get("DisNo", "")).strip().upper()
            iso = str(r.get("ISO", "")).strip().upper()
            rows.append(
                {
                    "year": int(year),
                    "DisNo": disno,
                    "disasterno_base": parse_disno_base(disno),
                    "ISO": iso,
                    "L_usd": float(pd.to_numeric(r.get("L_usd"), errors="coerce") or 0.0),
                    "status": str(r.get("status", "")).strip(),
                    "match_grade": str(r.get("match_grade", "")).strip(),
                    "PadStart": parse_date_safe(r.get("PadStart")),
                    "PadEnd": parse_date_safe(r.get("PadEnd")),
                    "selected_groups": str(r.get("selected_groups", "")).strip(),
                    "final_nc": str(final_nc),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates(subset=["year", "DisNo", "ISO"], keep="first").reset_index(drop=True)
    return out, missing


def load_gdis_storm(gpkg_path: Path, eval_years: list[int]) -> pd.GeoDataFrame:
    gdf = gpd.read_file(gpkg_path, layer="GPKG", engine="pyogrio")
    _require_columns(gdf, ["disastertype", "disasterno", "iso3", "geometry"], "gdis_GPKG")
    gdf = gdf.copy()
    gdf["disastertype_norm"] = gdf["disastertype"].astype(str).str.lower().str.strip()
    gdf = gdf[gdf["disastertype_norm"] == "storm"].copy()
    gdf["disasterno"] = gdf["disasterno"].astype(str).str.upper().str.strip()
    gdf["iso3"] = gdf["iso3"].astype(str).str.upper().str.strip()
    gdf["year"] = pd.to_numeric(gdf["disasterno"].str.slice(0, 4), errors="coerce").astype("Int64")
    gdf = gdf[gdf["year"].isin(eval_years)].copy()
    return gdf


def build_gdis_geom_maps(gdis_storm: gpd.GeoDataFrame) -> tuple[dict[str, Any], dict[tuple[str, str], Any]]:
    disno_map: dict[str, Any] = {}
    pair_map: dict[tuple[str, str], Any] = {}

    for disno, sdf in gdis_storm.groupby("disasterno", dropna=False):
        geoms = [g for g in sdf.geometry.tolist() if g is not None and (not getattr(g, "is_empty", False))]
        if geoms:
            disno_map[str(disno)] = unary_union(geoms)

    for (disno, iso3), sdf in gdis_storm.groupby(["disasterno", "iso3"], dropna=False):
        geoms = [g for g in sdf.geometry.tolist() if g is not None and (not getattr(g, "is_empty", False))]
        if geoms:
            pair_map[(str(disno), str(iso3))] = unary_union(geoms)

    return disno_map, pair_map


def compute_gdis_match_by_year(project_events: pd.DataFrame, gdis_storm: gpd.GeoDataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    gdis_by_year_disno = {
        int(y): set(sdf["disasterno"].dropna().astype(str).tolist()) for y, sdf in gdis_storm.groupby("year", dropna=False)
    }
    gdis_by_year_pair = {
        int(y): set(zip(sdf["disasterno"].astype(str), sdf["iso3"].astype(str)))
        for y, sdf in gdis_storm.groupby("year", dropna=False)
    }

    for year, sdf in project_events.groupby("year", dropna=False):
        y = int(year)
        pdf = sdf.copy()
        total = int(len(pdf))
        disno_set = gdis_by_year_disno.get(y, set())
        pair_set = gdis_by_year_pair.get(y, set())

        disno_match = int(pdf["disasterno_base"].astype(str).isin(disno_set).sum())
        pair_match = int(
            np.sum(
                [
                    (str(a), str(b)) in pair_set
                    for a, b in zip(pdf["disasterno_base"].astype(str).tolist(), pdf["ISO"].astype(str).tolist())
                ]
            )
        )
        rows.append(
            {
                "year": y,
                "project_event_count": total,
                "gdis_storm_disasterno_count": int(len(disno_set)),
                "gdis_storm_pair_count": int(len(pair_set)),
                "gdis_disasterno_match_count": disno_match,
                "gdis_pair_match_count": pair_match,
                "gdis_disasterno_match_rate": safe_ratio(disno_match, total),
                "gdis_pair_match_rate": safe_ratio(pair_match, total),
            }
        )

    out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    if out.empty:
        return out

    total_project = int(out["project_event_count"].sum())
    total_disno_match = int(out["gdis_disasterno_match_count"].sum())
    total_pair_match = int(out["gdis_pair_match_count"].sum())
    out.loc[len(out)] = {
        "year": "ALL",
        "project_event_count": total_project,
        "gdis_storm_disasterno_count": int(gdis_storm["disasterno"].nunique()),
        "gdis_storm_pair_count": int(gdis_storm.drop_duplicates(subset=["disasterno", "iso3"]).shape[0]),
        "gdis_disasterno_match_count": total_disno_match,
        "gdis_pair_match_count": total_pair_match,
        "gdis_disasterno_match_rate": safe_ratio(total_disno_match, total_project),
        "gdis_pair_match_rate": safe_ratio(total_pair_match, total_project),
    }
    return out


def compute_gdis_geometry_validation(
    cfg: RunConfig,
    project_events: pd.DataFrame,
    gdis_disno_map: dict[str, Any],
    gdis_pair_map: dict[tuple[str, str], Any],
    sample_size: int,
) -> pd.DataFrame:
    cand = project_events.copy()
    cand = cand[cand["disasterno_base"].astype(str) != ""].copy()
    cand["has_gdis_geom"] = [
        ((str(d), str(i)) in gdis_pair_map) or (str(d) in gdis_disno_map)
        for d, i in zip(cand["disasterno_base"].astype(str), cand["ISO"].astype(str))
    ]
    cand = cand[cand["has_gdis_geom"]].copy()
    if cand.empty:
        return pd.DataFrame(
            columns=["year", "DisNo", "ISO", "gdis_key", "distance_km", "gdis_hit", "point_count", "selected_groups_count"]
        )

    if len(cand) > sample_size:
        cand = cand.sample(n=int(sample_size), random_state=int(cfg.seed)).sort_values(["year", "DisNo", "ISO"])

    rows: list[dict[str, Any]] = []
    for year, sdf in cand.groupby("year", dropna=False):
        nc_path = Path(str(sdf["final_nc"].iloc[0]))
        if not nc_path.exists():
            continue
        with open_netcdf_readonly(nc_path, cfg) as ds:
            for _, r in sdf.iterrows():
                groups = split_groups(r.get("selected_groups", ""))
                if not groups:
                    continue
                pts = extract_event_points(
                    ds,
                    selected_groups=groups,
                    pad_start=r.get("PadStart"),
                    pad_end=r.get("PadEnd"),
                    seed=cfg.seed + int(year),
                    max_points=1000,
                )
                if pts.empty:
                    continue
                track_geom = build_track_geometry(pts)
                if track_geom is None:
                    continue
                disno = str(r.get("disasterno_base", ""))
                iso = str(r.get("ISO", ""))
                key = (disno, iso)
                ggeom = gdis_pair_map.get(key, gdis_disno_map.get(disno))
                if ggeom is None:
                    continue

                try:
                    dist_series = gpd.GeoSeries([track_geom, ggeom], crs="EPSG:4326").to_crs(3857)
                    dist_km = float(dist_series.iloc[0].distance(dist_series.iloc[1]) / 1000.0)
                except Exception:
                    dist_km = float("nan")
                hit = int(np.isfinite(dist_km) and dist_km <= GDIS_HIT_DISTANCE_KM)
                rows.append(
                    {
                        "year": int(year),
                        "DisNo": str(r.get("DisNo", "")),
                        "ISO": iso,
                        "gdis_key": f"{disno}|{iso}" if key in gdis_pair_map else disno,
                        "distance_km": dist_km,
                        "gdis_hit": hit,
                        "point_count": int(len(pts)),
                        "selected_groups_count": int(len(groups)),
                    }
                )

    return pd.DataFrame(rows).sort_values(["year", "DisNo", "ISO"]).reset_index(drop=True)


def _plot_gdis(match_df: pd.DataFrame, geom_df: pd.DataFrame, out: dict[str, Path], cfg: RunConfig) -> None:
    plot_match_csv = out["csv"] / "gdis_match_rate_trend_plot_data.csv"
    plot_hit_csv = out["csv"] / "gdis_hit_rate_trend_plot_data.csv"
    plot_dist_csv = out["csv"] / "gdis_distance_boxplot_plot_data.csv"

    md = match_df[match_df["year"] != "ALL"].copy()
    md["year"] = pd.to_numeric(md["year"], errors="coerce").astype("Int64")
    md = md.dropna(subset=["year"]).sort_values("year")
    write_csv(md, plot_match_csv, cfg)

    if not md.empty:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.plot(md["year"], md["gdis_disasterno_match_rate"], marker="o", label="disasterno")
        ax.plot(md["year"], md["gdis_pair_match_rate"], marker="o", label="disasterno+iso")
        ax.set_title("GDIS Match Rate by Year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Rate")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        ax.legend()
        save_figure(fig, out["png"] / "gdis_match_rate_trend.png")

    if geom_df.empty:
        write_csv(
            pd.DataFrame(columns=["year", "sampled_events", "gdis_geometry_hit_rate", "gdis_geometry_distance_km_p50", "gdis_geometry_distance_km_p95"]),
            plot_hit_csv,
            cfg,
        )
        write_csv(pd.DataFrame(columns=["year", "distance_km"]), plot_dist_csv, cfg)
        return

    gy = (
        geom_df.groupby("year", as_index=False)
        .agg(
            sampled_events=("DisNo", "count"),
            gdis_geometry_hit_rate=("gdis_hit", "mean"),
            gdis_geometry_distance_km_p50=("distance_km", lambda x: float(np.nanquantile(x, 0.5))),
            gdis_geometry_distance_km_p95=("distance_km", lambda x: float(np.nanquantile(x, 0.95))),
        )
        .sort_values("year")
    )
    write_csv(gy, plot_hit_csv, cfg)
    write_csv(geom_df[["year", "distance_km"]], plot_dist_csv, cfg)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(gy["year"], gy["gdis_geometry_hit_rate"], marker="o")
    ax.set_title("GDIS Geometry Hit Rate by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Hit Rate")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    save_figure(fig, out["png"] / "gdis_hit_rate_trend.png")

    years = sorted(geom_df["year"].dropna().astype(int).unique().tolist())
    dist_data = [geom_df.loc[geom_df["year"] == y, "distance_km"].dropna().to_numpy(dtype=float) for y in years]
    if any(len(x) for x in dist_data):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(dist_data, labels=[str(y) for y in years], showfliers=False)
        ax.set_title("GDIS Geometry Distance Distribution")
        ax.set_xlabel("Year")
        ax.set_ylabel("Distance (km)")
        ax.grid(alpha=0.2)
        save_figure(fig, out["png"] / "gdis_distance_boxplot.png")


def load_gfd_catalog(gfd_root: Path, eval_years: list[int]) -> pd.DataFrame:
    recs: list[dict[str, Any]] = []
    for jpath in sorted(gfd_root.rglob("DFO_*_properties.json")):
        try:
            obj = json.loads(jpath.read_text(encoding="utf-8"))
        except Exception:
            continue
        event_id = str(obj.get("id", "")).strip()
        began = parse_date_safe(obj.get("began"))
        ended = parse_date_safe(obj.get("ended"))
        if not event_id or began is None or ended is None:
            continue
        if began.year not in eval_years:
            continue

        tif_candidates = sorted(jpath.parent.glob("DFO_*.tif"))
        tif_path = tif_candidates[0] if tif_candidates else None
        if tif_path is None:
            continue
        iso_set = parse_iso_set(obj.get("cc"))
        if not iso_set:
            continue
        recs.append(
            {
                "gfd_event_id": event_id,
                "began": began,
                "ended": ended,
                "year": int(began.year),
                "iso_set": ";".join(sorted(iso_set)),
                "dfo_main_cause": normalize_cause(obj.get("dfo_main_cause")),
                "tif_path": str(tif_path),
            }
        )

    if not recs:
        return pd.DataFrame(columns=["gfd_event_id", "began", "ended", "year", "iso_set", "dfo_main_cause", "tif_path"])

    df = pd.DataFrame(recs).drop_duplicates(subset=["gfd_event_id"], keep="first").reset_index(drop=True)
    return df


def _choose_best_link(candidates: list[dict[str, Any]], pad_start, pad_end) -> dict[str, Any] | None:
    if not candidates:
        return None

    def _key(x: dict[str, Any]) -> tuple[int, int, str]:
        ov = int(x.get("overlap_days", 0))
        began = x.get("began")
        anchor = pad_start if pad_start is not None else pad_end
        if anchor is None or began is None:
            gap = 10**9
        else:
            gap = abs((began - anchor).days)
        return (-ov, gap, str(x.get("gfd_event_id", "")))

    return sorted(candidates, key=_key)[0]


def build_gfd_linkage(project_events: pd.DataFrame, gfd_catalog: pd.DataFrame) -> pd.DataFrame:
    if project_events.empty:
        return pd.DataFrame()

    gfd_records = gfd_catalog.to_dict(orient="records")
    for rec in gfd_records:
        rec["iso_tokens"] = {x for x in str(rec.get("iso_set", "")).split(";") if x}

    rows: list[dict[str, Any]] = []
    for _, ev in project_events.iterrows():
        iso = str(ev.get("ISO", "")).upper().strip()
        p0 = ev.get("PadStart")
        p1 = ev.get("PadEnd")
        strict_cands: list[dict[str, Any]] = []
        hydro_cands: list[dict[str, Any]] = []

        for rec in gfd_records:
            if iso not in rec["iso_tokens"]:
                continue
            ov = overlap_days(p0, p1, rec.get("began"), rec.get("ended"))
            if ov <= 0:
                continue
            one = dict(rec)
            one["overlap_days"] = int(ov)
            cause = normalize_cause(rec.get("dfo_main_cause"))
            if cause in STRICT_CAUSES:
                strict_cands.append(one)
            if cause in HYDROMET_CAUSES:
                hydro_cands.append(one)

        strict_best = _choose_best_link(strict_cands, p0, p1)
        hydro_best = _choose_best_link(hydro_cands, p0, p1)
        rows.append(
            {
                "year": int(ev.get("year")),
                "DisNo": str(ev.get("DisNo", "")),
                "ISO": iso,
                "disasterno_base": str(ev.get("disasterno_base", "")),
                "L_usd": float(ev.get("L_usd", 0.0)),
                "status": str(ev.get("status", "")),
                "match_grade": str(ev.get("match_grade", "")),
                "PadStart": p0,
                "PadEnd": p1,
                "selected_groups": str(ev.get("selected_groups", "")),
                "final_nc": str(ev.get("final_nc", "")),
                "gfd_linked_tc": int(strict_best is not None),
                "gfd_event_id_tc": str(strict_best.get("gfd_event_id", "")) if strict_best else "",
                "gfd_overlap_days_tc": int(strict_best.get("overlap_days", 0)) if strict_best else 0,
                "gfd_main_cause_tc": str(strict_best.get("dfo_main_cause", "")) if strict_best else "",
                "gfd_linked_hydromet": int(hydro_best is not None),
                "gfd_event_id_hydromet": str(hydro_best.get("gfd_event_id", "")) if hydro_best else "",
                "gfd_overlap_days_hydromet": int(hydro_best.get("overlap_days", 0)) if hydro_best else 0,
                "gfd_main_cause_hydromet": str(hydro_best.get("dfo_main_cause", "")) if hydro_best else "",
            }
        )
    return pd.DataFrame(rows).sort_values(["year", "DisNo", "ISO"]).reset_index(drop=True)


def compute_gfd_linkage_metrics(link_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if link_df.empty:
        return pd.DataFrame(
            columns=[
                "year",
                "project_event_count",
                "gfd_link_count_tc",
                "gfd_event_link_rate_tc",
                "gfd_link_count_hydromet",
                "gfd_event_link_rate_hydromet",
            ]
        )
    for year, sdf in link_df.groupby("year", dropna=False):
        total = int(len(sdf))
        tc = int(sdf["gfd_linked_tc"].sum())
        hy = int(sdf["gfd_linked_hydromet"].sum())
        rows.append(
            {
                "year": int(year),
                "project_event_count": total,
                "gfd_link_count_tc": tc,
                "gfd_event_link_rate_tc": safe_ratio(tc, total),
                "gfd_link_count_hydromet": hy,
                "gfd_event_link_rate_hydromet": safe_ratio(hy, total),
            }
        )
    out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    total = int(out["project_event_count"].sum())
    tc_total = int(out["gfd_link_count_tc"].sum())
    hy_total = int(out["gfd_link_count_hydromet"].sum())
    out.loc[len(out)] = {
        "year": "ALL",
        "project_event_count": total,
        "gfd_link_count_tc": tc_total,
        "gfd_event_link_rate_tc": safe_ratio(tc_total, total),
        "gfd_link_count_hydromet": hy_total,
        "gfd_event_link_rate_hydromet": safe_ratio(hy_total, total),
    }
    return out


def compute_gfd_overlap_metrics(
    cfg: RunConfig,
    link_df: pd.DataFrame,
    gfd_catalog: pd.DataFrame,
    scope: str,
) -> pd.DataFrame:
    if link_df.empty or gfd_catalog.empty:
        return pd.DataFrame(
            columns=[
                "year",
                "DisNo",
                "ISO",
                "scope",
                "gfd_event_id",
                "gfd_main_cause",
                "overlap_days",
                "num_points",
                "num_valid_samples",
                "gfd_flooded_overlap_ratio",
                "gfd_duration_hazard_corr",
                "gfd_permwater_exclusion_ratio",
            ]
        )

    if scope not in {"tc", "hydromet"}:
        raise ValueError(f"invalid scope: {scope}")

    linked_col = "gfd_linked_tc" if scope == "tc" else "gfd_linked_hydromet"
    id_col = "gfd_event_id_tc" if scope == "tc" else "gfd_event_id_hydromet"
    overlap_col = "gfd_overlap_days_tc" if scope == "tc" else "gfd_overlap_days_hydromet"
    cause_col = "gfd_main_cause_tc" if scope == "tc" else "gfd_main_cause_hydromet"

    sdf = link_df[link_df[linked_col] == 1].copy()
    if sdf.empty:
        return pd.DataFrame(
            columns=[
                "year",
                "DisNo",
                "ISO",
                "scope",
                "gfd_event_id",
                "gfd_main_cause",
                "overlap_days",
                "num_points",
                "num_valid_samples",
                "gfd_flooded_overlap_ratio",
                "gfd_duration_hazard_corr",
                "gfd_permwater_exclusion_ratio",
            ]
        )

    gfd_map = {str(r["gfd_event_id"]): r for _, r in gfd_catalog.iterrows()}
    rows: list[dict[str, Any]] = []

    for year, ydf in sdf.groupby("year", dropna=False):
        nc_path = Path(str(ydf["final_nc"].iloc[0]))
        if not nc_path.exists():
            continue
        with open_netcdf_readonly(nc_path, cfg) as ds:
            for _, r in ydf.iterrows():
                gid = str(r.get(id_col, ""))
                grec = gfd_map.get(gid)
                if grec is None:
                    continue
                tif_path = Path(str(grec.get("tif_path", "")))
                if not tif_path.exists():
                    continue

                groups = split_groups(r.get("selected_groups", ""))
                if not groups:
                    continue
                pts = extract_event_points(
                    ds,
                    selected_groups=groups,
                    pad_start=r.get("PadStart"),
                    pad_end=r.get("PadEnd"),
                    seed=cfg.seed + int(year),
                    max_points=1200,
                )
                if pts.empty:
                    continue

                coords = list(zip(pts["lon"].astype(float).tolist(), pts["lat"].astype(float).tolist()))
                try:
                    with rasterio.open(tif_path) as rs:
                        samples = list(rs.sample(coords, indexes=[1, 2, 5], masked=True))
                except Exception:
                    continue
                if not samples:
                    continue

                arr = np.ma.vstack(samples)
                b1 = np.asarray(arr[:, 0], dtype=float)
                b2 = np.asarray(arr[:, 1], dtype=float)
                b5 = np.asarray(arr[:, 2], dtype=float)

                valid = np.isfinite(b1)
                if not np.any(valid):
                    continue
                flooded = b1 > 0
                perm = b5 > 0
                before = float(np.mean(flooded[valid]))
                after = float(np.mean((flooded & (~perm))[valid]))
                if before > 0:
                    perm_excl = float((before - after) / before)
                else:
                    perm_excl = 0.0

                hz = pd.to_numeric(pts["hazard_compound"], errors="coerce").to_numpy(dtype=float)
                corr_mask = valid & np.isfinite(b2) & np.isfinite(hz)
                if np.count_nonzero(corr_mask) >= 3 and np.nanstd(b2[corr_mask]) > 0 and np.nanstd(hz[corr_mask]) > 0:
                    corr = float(np.corrcoef(b2[corr_mask], hz[corr_mask])[0, 1])
                else:
                    corr = float("nan")

                rows.append(
                    {
                        "year": int(year),
                        "DisNo": str(r.get("DisNo", "")),
                        "ISO": str(r.get("ISO", "")),
                        "scope": scope,
                        "gfd_event_id": gid,
                        "gfd_main_cause": str(r.get(cause_col, "")),
                        "overlap_days": int(r.get(overlap_col, 0)),
                        "num_points": int(len(pts)),
                        "num_valid_samples": int(np.count_nonzero(valid)),
                        "gfd_flooded_overlap_ratio": before,
                        "gfd_duration_hazard_corr": corr,
                        "gfd_permwater_exclusion_ratio": perm_excl,
                    }
                )

    return pd.DataFrame(rows).sort_values(["year", "DisNo", "scope"]).reset_index(drop=True)


def _plot_gfd(link_metrics: pd.DataFrame, overlap_df: pd.DataFrame, out: dict[str, Path], cfg: RunConfig) -> None:
    plot_link_csv = out["csv"] / "gfd_link_rate_comparison_plot_data.csv"
    plot_overlap_csv = out["csv"] / "gfd_overlap_distribution_plot_data.csv"
    plot_corr_csv = out["csv"] / "gfd_duration_corr_scatter_plot_data.csv"
    plot_perm_csv = out["csv"] / "gfd_permwater_sensitivity_plot_data.csv"

    lm = link_metrics[link_metrics["year"] != "ALL"].copy()
    lm["year"] = pd.to_numeric(lm["year"], errors="coerce").astype("Int64")
    lm = lm.dropna(subset=["year"]).sort_values("year")
    write_csv(lm, plot_link_csv, cfg)

    if not lm.empty:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.plot(lm["year"], lm["gfd_event_link_rate_tc"], marker="o", label="strict: tropical cyclone")
        ax.plot(lm["year"], lm["gfd_event_link_rate_hydromet"], marker="o", label="sensitivity: hydromet")
        ax.set_title("GFD Event Link Rate by Year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Rate")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        ax.legend()
        save_figure(fig, out["png"] / "gfd_link_rate_comparison.png")

    if overlap_df.empty:
        write_csv(pd.DataFrame(columns=["scope", "gfd_flooded_overlap_ratio"]), plot_overlap_csv, cfg)
        write_csv(
            pd.DataFrame(columns=["scope", "gfd_flooded_overlap_ratio", "gfd_duration_hazard_corr"]),
            plot_corr_csv,
            cfg,
        )
        write_csv(pd.DataFrame(columns=["scope", "mean_permwater_exclusion_ratio"]), plot_perm_csv, cfg)
        return

    write_csv(overlap_df[["scope", "gfd_flooded_overlap_ratio"]], plot_overlap_csv, cfg)
    write_csv(overlap_df[["scope", "gfd_flooded_overlap_ratio", "gfd_duration_hazard_corr"]], plot_corr_csv, cfg)

    perm_df = (
        overlap_df.groupby("scope", as_index=False)["gfd_permwater_exclusion_ratio"]
        .mean()
        .rename(columns={"gfd_permwater_exclusion_ratio": "mean_permwater_exclusion_ratio"})
    )
    write_csv(perm_df, plot_perm_csv, cfg)

    data_tc = overlap_df.loc[overlap_df["scope"] == "tc", "gfd_flooded_overlap_ratio"].dropna().to_numpy(dtype=float)
    data_hy = overlap_df.loc[overlap_df["scope"] == "hydromet", "gfd_flooded_overlap_ratio"].dropna().to_numpy(dtype=float)
    if len(data_tc) or len(data_hy):
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        box_data = [d for d in [data_tc, data_hy] if len(d)]
        labels = [lab for lab, d in [("tc", data_tc), ("hydromet", data_hy)] if len(d)]
        ax.boxplot(box_data, labels=labels, showfliers=False)
        ax.set_title("GFD Flooded Overlap Ratio Distribution")
        ax.set_ylabel("Overlap Ratio")
        ax.grid(alpha=0.2)
        save_figure(fig, out["png"] / "gfd_overlap_distribution.png")

    sc = overlap_df.dropna(subset=["gfd_flooded_overlap_ratio", "gfd_duration_hazard_corr"]).copy()
    if len(sc) >= 3:
        x = sc["gfd_flooded_overlap_ratio"].to_numpy(dtype=float)
        y = sc["gfd_duration_hazard_corr"].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(7.6, 5.2))
        ax.scatter(x, y, alpha=0.75)
        if np.nanstd(x) > 0 and np.nanstd(y) > 0:
            b1, b0 = np.polyfit(x, y, 1)
            xx = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 100)
            ax.plot(xx, b1 * xx + b0, color="tab:red", linewidth=1.4, label="OLS fit")
            ax.legend()
        ax.set_title("GFD Duration vs Project Hazard Correlation")
        ax.set_xlabel("Flooded Overlap Ratio")
        ax.set_ylabel("Duration-Hazard Correlation")
        ax.grid(alpha=0.25)
        save_figure(fig, out["png"] / "gfd_duration_hazard_corr_scatter.png")

    if not perm_df.empty:
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        ax.bar(perm_df["scope"], perm_df["mean_permwater_exclusion_ratio"], color=["tab:blue", "tab:orange"][: len(perm_df)])
        ax.set_title("Perm-water Exclusion Sensitivity")
        ax.set_ylabel("Mean Exclusion Ratio")
        ax.grid(alpha=0.2, axis="y")
        save_figure(fig, out["png"] / "gfd_permwater_sensitivity.png")


def run_l6_external_validation(
    cfg: RunConfig,
    geometry_sample_size: int = 200,
) -> tuple[StageResult, dict[str, Any]]:
    out = _prepare_output_dirs(cfg.results_root)
    eval_years, not_eval_years = split_external_years(cfg.years)

    project_events, missing_inputs = load_project_events(cfg, eval_years)
    missing_df = pd.DataFrame(missing_inputs)
    if not missing_df.empty:
        write_csv(missing_df, out["root"] / "missing_inputs.csv", cfg)

    gdis_gpkg = cfg.raw_using_root / "gdis" / "pend-gdis-1960-2018-disasterlocations.gpkg"
    gfd_root = cfg.raw_using_root / "GFD" / "gfd_unwrap"

    hard_fail_reasons: list[str] = []
    warn_reasons: list[str] = []

    if not gdis_gpkg.exists():
        hard_fail_reasons.append("missing_gdis_gpkg")
        gdis_storm = gpd.GeoDataFrame(columns=["disasterno", "iso3", "geometry"])
    else:
        gdis_storm = load_gdis_storm(gdis_gpkg, eval_years)

    if not gfd_root.exists():
        hard_fail_reasons.append("missing_gfd_root")
        gfd_catalog = pd.DataFrame(columns=["gfd_event_id", "began", "ended", "year", "iso_set", "dfo_main_cause", "tif_path"])
    else:
        gfd_catalog = load_gfd_catalog(gfd_root, eval_years)

    if missing_inputs:
        hard_fail_reasons.append("missing_required_project_inputs")
    if project_events.empty:
        hard_fail_reasons.append("no_project_events_for_external_window")

    if not hard_fail_reasons:
        gdis_match_df = compute_gdis_match_by_year(project_events, gdis_storm)
    else:
        gdis_match_df = pd.DataFrame(
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
    write_csv(gdis_match_df, out["root"] / "gdis_match_metrics_by_year.csv", cfg)

    gdis_disno_map, gdis_pair_map = build_gdis_geom_maps(gdis_storm) if not gdis_storm.empty else ({}, {})
    geom_df = compute_gdis_geometry_validation(
        cfg,
        project_events=project_events,
        gdis_disno_map=gdis_disno_map,
        gdis_pair_map=gdis_pair_map,
        sample_size=int(max(1, geometry_sample_size)),
    )
    write_csv(geom_df, out["root"] / "gdis_geometry_validation.csv", cfg)

    link_df = build_gfd_linkage(project_events, gfd_catalog)
    link_metrics_df = compute_gfd_linkage_metrics(link_df)
    write_csv(link_metrics_df, out["root"] / "gfd_event_linkage_metrics.csv", cfg)

    tc_overlap_df = compute_gfd_overlap_metrics(cfg, link_df, gfd_catalog, scope="tc")
    hy_overlap_df = compute_gfd_overlap_metrics(cfg, link_df, gfd_catalog, scope="hydromet")
    overlap_df = pd.concat([tc_overlap_df, hy_overlap_df], ignore_index=True) if (not tc_overlap_df.empty or not hy_overlap_df.empty) else pd.DataFrame()
    write_csv(overlap_df, out["root"] / "gfd_overlap_metrics_by_event.csv", cfg)

    _plot_gdis(gdis_match_df, geom_df, out, cfg)
    _plot_gfd(link_metrics_df, overlap_df, out, cfg)

    # ---- status checks ----
    gdis_all = gdis_match_df[gdis_match_df["year"] == "ALL"]
    if not gdis_all.empty:
        disno_rate_all = float(gdis_all["gdis_disasterno_match_rate"].iloc[0])
        pair_rate_all = float(gdis_all["gdis_pair_match_rate"].iloc[0])
        if np.isfinite(disno_rate_all) and disno_rate_all < GDIS_DISASTERNO_MATCH_FAIL:
            hard_fail_reasons.append("gdis_disasterno_match_rate_below_threshold")
        if np.isfinite(pair_rate_all) and pair_rate_all < GDIS_PAIR_MATCH_FAIL:
            hard_fail_reasons.append("gdis_pair_match_rate_below_threshold")
    else:
        disno_rate_all = float("nan")
        pair_rate_all = float("nan")

    if not geom_df.empty:
        hit_rate_all = float(np.nanmean(pd.to_numeric(geom_df["gdis_hit"], errors="coerce")))
        p95_dist_all = float(np.nanquantile(pd.to_numeric(geom_df["distance_km"], errors="coerce"), 0.95))
        if np.isfinite(hit_rate_all) and hit_rate_all < GDIS_HIT_WARN:
            warn_reasons.append("gdis_geometry_hit_rate_low")
        if np.isfinite(p95_dist_all) and p95_dist_all > GDIS_P95_DISTANCE_WARN_KM:
            warn_reasons.append("gdis_geometry_distance_p95_high")
    else:
        hit_rate_all = float("nan")
        p95_dist_all = float("nan")
        warn_reasons.append("gdis_geometry_sample_empty")

    gfd_all = link_metrics_df[link_metrics_df["year"] == "ALL"]
    if not gfd_all.empty:
        gfd_tc_rate_all = float(gfd_all["gfd_event_link_rate_tc"].iloc[0])
        gfd_hy_rate_all = float(gfd_all["gfd_event_link_rate_hydromet"].iloc[0])
        gfd_tc_count_all = int(gfd_all["gfd_link_count_tc"].iloc[0])
    else:
        gfd_tc_rate_all = float("nan")
        gfd_hy_rate_all = float("nan")
        gfd_tc_count_all = 0
        hard_fail_reasons.append("missing_gfd_linkage_metrics")

    if np.isfinite(gfd_tc_rate_all) and gfd_tc_rate_all < GFD_LINK_RATE_TC_FAIL:
        hard_fail_reasons.append("gfd_event_link_rate_tc_low")
    if gfd_tc_count_all < GFD_STRICT_MIN_SAMPLES_WARN:
        warn_reasons.append("gfd_strict_scope_sample_insufficient")

    tc_stat = overlap_df[overlap_df["scope"] == "tc"].copy() if not overlap_df.empty else pd.DataFrame()
    if not tc_stat.empty:
        med_overlap_tc = float(np.nanmedian(pd.to_numeric(tc_stat["gfd_flooded_overlap_ratio"], errors="coerce")))
        med_corr_tc = float(np.nanmedian(pd.to_numeric(tc_stat["gfd_duration_hazard_corr"], errors="coerce")))
    else:
        med_overlap_tc = float("nan")
        med_corr_tc = float("nan")
        warn_reasons.append("gfd_tc_overlap_metrics_empty")

    if np.isfinite(med_overlap_tc) and np.isfinite(med_corr_tc):
        if med_overlap_tc < GFD_OVERLAP_LOW_FAIL and med_corr_tc <= 0:
            hard_fail_reasons.append("gfd_overlap_and_corr_direction_failure")

    if not_eval_years:
        warn_reasons.append("years_not_evaluable_by_external_data")

    stage_status = finalize_stage_status(has_fail=bool(hard_fail_reasons), has_warn=bool(warn_reasons))

    not_eval_df = pd.DataFrame(
        {
            "year": not_eval_years,
            "reason": ["Not-Evaluable by external datasets (coverage 2000-2018)"] * len(not_eval_years),
        }
    )
    write_csv(not_eval_df, out["root"] / "not_evaluable_years_2019_2024.csv", cfg)

    status_payload = {
        "level": "L6",
        "status": stage_status,
        "external_window": {"start": EXTERNAL_YEAR_START, "end": EXTERNAL_YEAR_END},
        "selected_years": cfg.years,
        "evaluated_years": eval_years,
        "not_evaluable_years": not_eval_years,
        "thresholds": {
            "gdis_disasterno_match_rate_fail": GDIS_DISASTERNO_MATCH_FAIL,
            "gdis_pair_match_rate_fail": GDIS_PAIR_MATCH_FAIL,
            "gfd_event_link_rate_tc_fail": GFD_LINK_RATE_TC_FAIL,
            "gfd_strict_min_samples_warn": GFD_STRICT_MIN_SAMPLES_WARN,
        },
        "metrics": {
            "gdis_disasterno_match_rate": disno_rate_all,
            "gdis_pair_match_rate": pair_rate_all,
            "gdis_geometry_hit_rate": hit_rate_all,
            "gdis_geometry_distance_km_p95": p95_dist_all,
            "gfd_event_link_rate_tc": gfd_tc_rate_all,
            "gfd_event_link_rate_hydromet": gfd_hy_rate_all,
            "gfd_tc_overlap_median": med_overlap_tc,
            "gfd_tc_duration_corr_median": med_corr_tc,
            "gfd_tc_link_count": gfd_tc_count_all,
        },
        "hard_fail_reasons": hard_fail_reasons,
        "warn_reasons": warn_reasons,
        "sample_sizes": {
            "project_events": int(len(project_events)),
            "gdis_geometry_samples": int(len(geom_df)),
            "gfd_catalog_events": int(len(gfd_catalog)),
            "gfd_overlap_rows": int(len(overlap_df)),
        },
    }
    write_json(status_payload, out["root"] / "l6_status.json", cfg)

    # summaries
    gdis_summary_lines = [
        "# L6-A GDIS Validation Summary",
        "",
        f"- external_window: {EXTERNAL_YEAR_START}-{EXTERNAL_YEAR_END}",
        f"- project_events: {len(project_events)}",
        f"- gdis_storm_rows: {len(gdis_storm)}",
        "",
        "## Core Rates",
        "",
    ]
    if not gdis_all.empty:
        gdis_summary_lines.append(f"- gdis_disasterno_match_rate: {disno_rate_all:.4f}")
        gdis_summary_lines.append(f"- gdis_pair_match_rate: {pair_rate_all:.4f}")
    else:
        gdis_summary_lines.append("- gdis_disasterno_match_rate: nan")
        gdis_summary_lines.append("- gdis_pair_match_rate: nan")
    gdis_summary_lines.extend(
        [
            f"- gdis_geometry_hit_rate: {hit_rate_all:.4f}" if np.isfinite(hit_rate_all) else "- gdis_geometry_hit_rate: nan",
            f"- gdis_geometry_distance_km_p95: {p95_dist_all:.2f}" if np.isfinite(p95_dist_all) else "- gdis_geometry_distance_km_p95: nan",
            "",
            "## Yearly Metrics",
            "",
            ("```text\n" + gdis_match_df.to_string(index=False) + "\n```") if not gdis_match_df.empty else "_no rows_",
            "",
        ]
    )
    write_text("\n".join(gdis_summary_lines) + "\n", out["root"] / "gdis_validation_summary.md", cfg)

    gfd_summary_lines = [
        "# L6-B GFD Validation Summary",
        "",
        f"- external_window: {EXTERNAL_YEAR_START}-{EXTERNAL_YEAR_END}",
        f"- gfd_catalog_events: {len(gfd_catalog)}",
        f"- strict_cause_count: {int((gfd_catalog['dfo_main_cause'].isin(list(STRICT_CAUSES))).sum()) if not gfd_catalog.empty else 0}",
        "",
        "## Linkage Metrics",
        "",
        ("```text\n" + link_metrics_df.to_string(index=False) + "\n```") if not link_metrics_df.empty else "_no rows_",
        "",
        "## Event Overlap Metrics (strict + hydromet)",
        "",
        ("```text\n" + overlap_df.head(50).to_string(index=False) + "\n```") if not overlap_df.empty else "_no rows_",
        "",
    ]
    write_text("\n".join(gfd_summary_lines) + "\n", out["root"] / "gfd_validation_summary.md", cfg)

    l6_summary_lines = [
        "# L6 External Validation Summary",
        "",
        "## Scope",
        "",
        f"- evaluated_years: {eval_years}",
        f"- not_evaluable_years: {not_eval_years}",
        "- note: 2019-2024 are flagged as `Not-Evaluable by external datasets` and do not trigger hard fail directly.",
        "",
        "## Final Status",
        "",
        f"- stage_status: **{stage_status}**",
        f"- hard_fail_reasons: {', '.join(hard_fail_reasons) if hard_fail_reasons else 'none'}",
        f"- warn_reasons: {', '.join(warn_reasons) if warn_reasons else 'none'}",
        "",
        "## Global Metrics",
        "",
        f"- gdis_disasterno_match_rate: {disno_rate_all:.4f}" if np.isfinite(disno_rate_all) else "- gdis_disasterno_match_rate: nan",
        f"- gdis_pair_match_rate: {pair_rate_all:.4f}" if np.isfinite(pair_rate_all) else "- gdis_pair_match_rate: nan",
        f"- gdis_geometry_hit_rate: {hit_rate_all:.4f}" if np.isfinite(hit_rate_all) else "- gdis_geometry_hit_rate: nan",
        f"- gdis_geometry_distance_km_p95: {p95_dist_all:.2f}" if np.isfinite(p95_dist_all) else "- gdis_geometry_distance_km_p95: nan",
        f"- gfd_event_link_rate_tc: {gfd_tc_rate_all:.4f}" if np.isfinite(gfd_tc_rate_all) else "- gfd_event_link_rate_tc: nan",
        f"- gfd_event_link_rate_hydromet: {gfd_hy_rate_all:.4f}" if np.isfinite(gfd_hy_rate_all) else "- gfd_event_link_rate_hydromet: nan",
        f"- gfd_tc_overlap_median: {med_overlap_tc:.4f}" if np.isfinite(med_overlap_tc) else "- gfd_tc_overlap_median: nan",
        f"- gfd_tc_duration_corr_median: {med_corr_tc:.4f}" if np.isfinite(med_corr_tc) else "- gfd_tc_duration_corr_median: nan",
        "",
    ]
    write_text("\n".join(l6_summary_lines) + "\n", out["root"] / "l6_validation_summary.md", cfg)

    fail_count = len(hard_fail_reasons)
    warn_count = len(warn_reasons)
    result = StageResult(
        level="L6",
        status=stage_status,
        fail_count=fail_count,
        warn_count=warn_count,
        metrics={
            "project_events": int(len(project_events)),
            "gdis_geometry_samples": int(len(geom_df)),
            "gfd_catalog_events": int(len(gfd_catalog)),
            "gfd_overlap_rows": int(len(overlap_df)),
            "gdis_disasterno_match_rate": disno_rate_all,
            "gdis_pair_match_rate": pair_rate_all,
            "gfd_event_link_rate_tc": gfd_tc_rate_all,
            "gfd_event_link_rate_hydromet": gfd_hy_rate_all,
        },
        artifacts=[
            str(out["root"] / "gdis_match_metrics_by_year.csv"),
            str(out["root"] / "gdis_geometry_validation.csv"),
            str(out["root"] / "gdis_validation_summary.md"),
            str(out["root"] / "gfd_event_linkage_metrics.csv"),
            str(out["root"] / "gfd_overlap_metrics_by_event.csv"),
            str(out["root"] / "gfd_validation_summary.md"),
            str(out["root"] / "l6_status.json"),
            str(out["root"] / "not_evaluable_years_2019_2024.csv"),
            str(out["root"] / "l6_validation_summary.md"),
        ],
        notes=[
            f"hard_fail_reasons={','.join(hard_fail_reasons) if hard_fail_reasons else 'none'}",
            f"warn_reasons={','.join(warn_reasons) if warn_reasons else 'none'}",
        ],
    )

    enriched = link_df.copy()
    if not enriched.empty:
        gdis_disno_set = set(gdis_storm["disasterno"].astype(str).tolist()) if not gdis_storm.empty else set()
        gdis_pair_set = (
            set(zip(gdis_storm["disasterno"].astype(str).tolist(), gdis_storm["iso3"].astype(str).tolist()))
            if not gdis_storm.empty
            else set()
        )
        enriched["gdis_disasterno_matched"] = enriched["disasterno_base"].astype(str).isin(gdis_disno_set).astype(int)
        enriched["gdis_pair_matched"] = [
            int((str(d), str(i)) in gdis_pair_set)
            for d, i in zip(enriched["disasterno_base"].astype(str), enriched["ISO"].astype(str))
        ]

    ctx = {
        "project_events": project_events,
        "event_linkage": enriched,
        "gdis_disno_geom_map": gdis_disno_map,
        "gdis_pair_geom_map": gdis_pair_map,
        "gfd_catalog": gfd_catalog,
        "eval_years": eval_years,
        "not_evaluable_years": not_eval_years,
        "l6_status": status_payload,
    }
    return result, ctx
