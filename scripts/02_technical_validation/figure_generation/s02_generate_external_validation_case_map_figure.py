#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from netCDF4 import Dataset, num2date
from shapely.geometry import LineString, MultiPoint, Point
from shapely.ops import unary_union

SCRIPT_DIR = Path(__file__).resolve().parent
REPRO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULTS_ROOT = REPRO_ROOT / "05_benchmark_results" / "results_v2"
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "using"
DEFAULT_FINAL_ROOT = REPRO_ROOT / "new_final_output" / "emdat_attribution_integration"
DEFAULT_OUTPUT_PNG = SCRIPT_DIR / "technical_validation_external_case_map.png"
DEFAULT_OUTPUT_PDF = SCRIPT_DIR / "technical_validation_external_case_map.pdf"

COLORS = {
    "track": "#0B4F8A",
    "track_window": "#0072B2",
    "gdis_fill": "#E69F00",
    "gdis_edge": "#B56900",
    "project_contour": "#CC79A7",
    "imerg_contour": "#009E73",
    "hazard_only_color": "#CC79A7",
    "loss_only_color": "#2EC4B6",
    "overlap_color": "#7B2CBF",
    "boundary": "#555555",
    "country_fill": "#F3F4F6",
    "ocean_fill": "#EEF2F5",
    "dark": "#1F2A36",
    "grid": "#D9DEE5",
}
FONT_FAMILY = "sans-serif"
FONT_STACK = ["Helvetica", "Arial", "DejaVu Sans"]
BASE_FONT_SIZE_PT = 14
TICK_LABEL_SIZE = 12
LEGEND_FONT_SIZE = 12
PANEL_LABEL_SIZE = 18
METRIC_FONT_SIZE = 12
AXIS_LABEL_SIZE = BASE_FONT_SIZE_PT

CASE_DISNO_DEFAULT = "2018-0227-VNM"
CASE_ISO_DEFAULT = "VNM"
RAIN_DISPLAY_FLOOR = 5.0
PANEL_TITLE_SIZE = BASE_FONT_SIZE_PT


@dataclass
class CaseContext:
    case_disno: str
    case_iso: str
    year: int
    disasterno_base: str
    case_country: str
    selected_group: str
    pad_start: date
    pad_end: date
    case_row: pd.Series
    gdis_row: pd.Series
    final_nc_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a map-led external validation figure for one representative case.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--final-root", type=Path, default=DEFAULT_FINAL_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--case-disno", type=str, default=CASE_DISNO_DEFAULT)
    parser.add_argument("--case-iso", type=str, default=CASE_ISO_DEFAULT)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def require_paths(paths: dict[str, Path]) -> None:
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(missing))


def read_case_context(results_root: Path, final_root: Path, case_disno: str, case_iso: str) -> CaseContext:
    case_catalog = results_root / "07_case_validation" / "case_catalog.csv"
    gdis_metrics = results_root / "06_external_validation" / "gdis_geometry_validation.csv"
    imerg_metrics = results_root / "06_external_validation" / "imerg_event_scale_metrics.csv"
    require_paths(
        {
            "case_catalog": case_catalog,
            "gdis_geometry_validation": gdis_metrics,
            "imerg_event_scale_metrics": imerg_metrics,
        }
    )

    case_df = pd.read_csv(case_catalog, encoding="utf-8-sig")
    gdis_df = pd.read_csv(gdis_metrics, encoding="utf-8-sig")
    imerg_df = pd.read_csv(imerg_metrics, encoding="utf-8-sig")

    case_match = case_df[(case_df["DisNo"].astype(str) == case_disno) & (case_df["ISO"].astype(str) == case_iso)]
    if case_match.empty:
        raise ValueError(f"Case not found in case_catalog.csv: {case_disno} / {case_iso}")
    case_row = case_match.iloc[0]

    imerg_match = imerg_df[(imerg_df["DisNo"].astype(str) == case_disno) & (imerg_df["ISO"].astype(str) == case_iso)]
    if imerg_match.empty:
        raise ValueError(f"Case not found in imerg_event_scale_metrics.csv: {case_disno} / {case_iso}")
    imerg_row = imerg_match.iloc[0]

    gdis_match = gdis_df[gdis_df["DisNo"].astype(str) == case_disno]
    if gdis_match.empty:
        raise ValueError(f"Case not found in gdis_geometry_validation.csv: {case_disno}")
    gdis_row = gdis_match.iloc[0]

    selected_groups = str(case_row["selected_groups"]).split(";")
    if len(selected_groups) != 1:
        raise ValueError(f"Expected a single selected group for the representative case, got: {case_row['selected_groups']}")
    selected_group = selected_groups[0]

    year = int(case_row["year"])
    final_nc_path = final_root / f"typhoon_{year}_ocean_litpop_poplight_emdat.nc"
    require_paths({"final_nc": final_nc_path})

    return CaseContext(
        case_disno=case_disno,
        case_iso=case_iso,
        year=year,
        disasterno_base=str(case_row["disasterno_base"]),
        case_country="",
        selected_group=selected_group,
        pad_start=pd.Timestamp(case_row["PadStart"]).date(),
        pad_end=pd.Timestamp(case_row["PadEnd"]).date(),
        case_row=imerg_row,
        gdis_row=gdis_row,
        final_nc_path=final_nc_path,
    )


def add_panel_label(ax, letter: str) -> None:
    label = f"({letter})"
    ax.text(
        -0.05,
        1.05,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        color=COLORS["dark"],
    )


def minimalist_map_axes(ax, mean_lat: float) -> None:
    ax.set_facecolor(COLORS["ocean_fill"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#BFC7D1")
    ax.spines["bottom"].set_color("#BFC7D1")
    ax.grid(color=COLORS["grid"], linewidth=0.55, alpha=0.55)
    ax.tick_params(labelsize=TICK_LABEL_SIZE, colors=COLORS["dark"])
    ax.set_aspect(1.0 / math.cos(math.radians(mean_lat)))
    ax.set_xlabel("Longitude", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Latitude", fontsize=AXIS_LABEL_SIZE)


def draw_legend_axis(
    ax,
    handles,
    ncol: int = 2,
    fontsize: float = LEGEND_FONT_SIZE,
    letter: str | None = None,
    bbox_to_anchor: tuple[float, float] | tuple[float, float, float, float] | None = None,
    mode: str | None = None,
    columnspacing: float = 1.5,
) -> None:
    x_start = 0.0
    y_anchor = 1.03
    if letter is not None:
        label = f"({letter})"
        ax.text(
            0.0,
            y_anchor,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=PANEL_LABEL_SIZE,
            fontweight="bold",
            color=COLORS["dark"],
        )
        x_start = 0.08
    if not handles:
        return
    if bbox_to_anchor is None:
        bbox_to_anchor = (x_start, y_anchor)
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        frameon=False,
        fontsize=fontsize,
        handlelength=2.0,
        handletextpad=0.6,
        columnspacing=columnspacing,
        borderaxespad=0.0,
        mode=mode,
    )


def daterange(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def decode_time_to_dates(var) -> list[datetime]:
    values = np.asarray(var[:], dtype=np.float64)
    units = str(var.units)
    return [datetime(int(dt.year), int(dt.month), int(dt.day), int(dt.hour), int(dt.minute), int(dt.second)) for dt in num2date(values, units=units, calendar="standard")]


def read_country_geometries(raw_root: Path, case_iso: str) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    gadm_path = raw_root / "gadm" / "gadm_410-levels.gpkg"
    require_paths({"gadm": gadm_path})
    country_gdf = gpd.read_file(gadm_path, layer="ADM_0", where=f"GID_0 = '{case_iso}'")
    if country_gdf.empty:
        raise ValueError(f"Failed to find case country in GADM layer ADM_0: {case_iso}")
    country_name = str(country_gdf.iloc[0]["COUNTRY"])
    return country_gdf, gpd.read_file(gadm_path, layer="ADM_0", bbox=tuple(country_gdf.total_bounds.tolist()))


def read_case_group_data(final_nc_path: Path, selected_group: str, pad_start: date, pad_end: date) -> dict[str, object]:
    with Dataset(final_nc_path) as ds:
        group = ds.groups.get(selected_group)
        if group is None:
            raise ValueError(f"Group not found in final nc: {selected_group}")

        all_times = decode_time_to_dates(group.variables["time"])
        mask = np.array([pad_start <= dt.date() <= pad_end for dt in all_times], dtype=bool)
        indices = np.where(mask)[0]
        if indices.size == 0:
            raise ValueError(f"No time steps found inside the validation window for group {selected_group}")

        center_lat = np.asarray(group.variables["center_lat"][:], dtype=float)
        center_lon = np.asarray(group.variables["center_lon"][:], dtype=float)
        window_lat = np.asarray(group.variables["window_lat"][:], dtype=float)
        window_lon = np.asarray(group.variables["window_lon"][:], dtype=float)
        hazard_rain = np.asarray(group.variables["hazard_rain_daily"][:], dtype=float)
        hazard_compound = np.asarray(group.variables["hazard_compound_daily"][:], dtype=float)
        loss = np.asarray(group.variables["emdat_loss_allocated_usd"][:], dtype=float)

        return {
            "all_times": all_times,
            "window_indices": indices,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "window_lat": window_lat,
            "window_lon": window_lon,
            "hazard_rain": hazard_rain,
            "hazard_compound": hazard_compound,
            "loss": loss,
        }


def extract_event_points(case_group: dict[str, object], pad_start: date, pad_end: date, hazard_q: float = 0.9) -> pd.DataFrame:
    all_times: list[datetime] = case_group["all_times"]
    center_lat = case_group["center_lat"]
    center_lon = case_group["center_lon"]
    window_lat = case_group["window_lat"]
    window_lon = case_group["window_lon"]
    hazard_compound = case_group["hazard_compound"]
    hazard_rain = case_group["hazard_rain"]
    loss = case_group["loss"]

    rows: list[pd.DataFrame] = []
    for idx, dt in enumerate(all_times):
        current_date = dt.date()
        if current_date < pad_start or current_date > pad_end:
            continue
        center_lat_i = float(center_lat[idx])
        center_lon_i = float(center_lon[idx])
        hazard_i = np.asarray(hazard_compound[idx], dtype=float)
        rain_i = np.asarray(hazard_rain[idx], dtype=float)
        loss_i = np.asarray(loss[idx], dtype=float)

        lat_abs = center_lat_i + np.broadcast_to(window_lat[:, None], hazard_i.shape)
        lon_abs = center_lon_i + np.broadcast_to(window_lon[None, :], hazard_i.shape)

        finite = np.isfinite(hazard_i)
        loss_mask = loss_i > 0
        if np.any(finite):
            qv = float(np.nanquantile(hazard_i[finite], hazard_q))
            hazard_top_mask = finite & (hazard_i >= qv)
        else:
            hazard_top_mask = np.zeros_like(loss_mask, dtype=bool)
        union_mask = hazard_top_mask | loss_mask

        if not np.any(union_mask):
            if np.any(np.isfinite(hazard_i)):
                arg = int(np.nanargmax(hazard_i))
                rr, cc = np.unravel_index(arg, hazard_i.shape)
                hazard_top_mask = np.zeros_like(hazard_i, dtype=bool)
                loss_mask = np.zeros_like(hazard_i, dtype=bool)
                hazard_top_mask[rr, cc] = True
            else:
                hazard_top_mask = np.zeros_like(hazard_i, dtype=bool)
                loss_mask = np.zeros_like(hazard_i, dtype=bool)
                hazard_top_mask[hazard_i.shape[0] // 2, hazard_i.shape[1] // 2] = True
            union_mask = hazard_top_mask | loss_mask

        rr, cc = np.where(union_mask)
        selected_hazard = hazard_top_mask[rr, cc]
        selected_loss = loss_mask[rr, cc]
        point_class = np.where(
            selected_hazard & selected_loss,
            "overlap",
            np.where(selected_hazard, "hazard_only", "loss_only"),
        )
        rows.append(
            pd.DataFrame(
                {
                    "time": [dt] * len(rr),
                    "date": [current_date] * len(rr),
                    "lat": lat_abs[rr, cc].astype(float),
                    "lon": lon_abs[rr, cc].astype(float),
                    "hazard_compound": hazard_i[rr, cc].astype(float),
                    "hazard_rain": rain_i[rr, cc].astype(float),
                    "loss_alloc": loss_i[rr, cc].astype(float),
                    "is_hazard_topq": selected_hazard.astype(bool),
                    "is_loss_alloc": selected_loss.astype(bool),
                    "point_class": point_class.astype(str),
                }
            )
        )

    if not rows:
        raise ValueError("Event point extraction returned no rows for the validation window")
    return pd.concat(rows, ignore_index=True)


def compute_event_geometry(event_points: pd.DataFrame) -> tuple[LineString | Point, object]:
    coords = event_points[["lon", "lat"]].dropna().to_numpy(dtype=float)
    unique_coords = []
    seen = set()
    for lon, lat in coords:
        xy = (float(lon), float(lat))
        if xy not in seen:
            unique_coords.append(xy)
            seen.add(xy)

    if len(unique_coords) == 1:
        line = Point(unique_coords[0])
    else:
        line = LineString(unique_coords)
    hull = MultiPoint(unique_coords).convex_hull
    return line, hull


def round_extent(bounds: tuple[float, float, float, float], pad_lon: float, pad_lat: float) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bounds
    return (
        math.floor((xmin - pad_lon) * 2.0) / 2.0,
        math.floor((ymin - pad_lat) * 2.0) / 2.0,
        math.ceil((xmax + pad_lon) * 2.0) / 2.0,
        math.ceil((ymax + pad_lat) * 2.0) / 2.0,
    )


def clamp_extent(extent: tuple[float, float, float, float], outer_extent: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = extent
    outer_xmin, outer_ymin, outer_xmax, outer_ymax = outer_extent
    width = xmax - xmin
    height = ymax - ymin
    outer_width = outer_xmax - outer_xmin
    outer_height = outer_ymax - outer_ymin

    if width >= outer_width or height >= outer_height:
        return outer_extent

    shift_x = 0.0
    shift_y = 0.0
    if xmin < outer_xmin:
        shift_x = outer_xmin - xmin
    elif xmax > outer_xmax:
        shift_x = outer_xmax - xmax

    if ymin < outer_ymin:
        shift_y = outer_ymin - ymin
    elif ymax > outer_ymax:
        shift_y = outer_ymax - ymax

    return (xmin + shift_x, ymin + shift_y, xmax + shift_x, ymax + shift_y)


def build_zoom_extent(
    bounds: tuple[float, float, float, float],
    local_extent: tuple[float, float, float, float],
    pad_lon: float,
    pad_lat: float,
    min_width: float,
    min_height: float,
) -> tuple[float, float, float, float]:
    xmin = bounds[0] - pad_lon
    ymin = bounds[1] - pad_lat
    xmax = bounds[2] + pad_lon
    ymax = bounds[3] + pad_lat
    width = max(xmax - xmin, min_width)
    height = max(ymax - ymin, min_height)
    target_ratio = (local_extent[2] - local_extent[0]) / (local_extent[3] - local_extent[1])

    if width / height < target_ratio:
        width = height * target_ratio
    else:
        height = width / target_ratio

    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    zoom_extent = (
        center_x - 0.5 * width,
        center_y - 0.5 * height,
        center_x + 0.5 * width,
        center_y + 0.5 * height,
    )
    return clamp_extent(zoom_extent, local_extent)


def build_gdis_geom_maps(gdis_storm: gpd.GeoDataFrame) -> tuple[dict[str, object], dict[tuple[str, str], object]]:
    disno_map: dict[str, object] = {}
    pair_map: dict[tuple[str, str], object] = {}

    for disno, sdf in gdis_storm.groupby("disasterno", dropna=False):
        geoms = [g for g in sdf.geometry.tolist() if g is not None and (not getattr(g, "is_empty", False))]
        if geoms:
            disno_map[str(disno)] = unary_union(geoms)

    for (disno, iso3), sdf in gdis_storm.groupby(["disasterno", "iso3"], dropna=False):
        geoms = [g for g in sdf.geometry.tolist() if g is not None and (not getattr(g, "is_empty", False))]
        if geoms:
            pair_map[(str(disno), str(iso3))] = unary_union(geoms)

    return disno_map, pair_map


def read_gdis_geometry(
    raw_root: Path,
    disasterno_base: str,
    project_iso: str,
    expected_key: str | None = None,
) -> tuple[gpd.GeoDataFrame, str]:
    gdis_path = raw_root / "gdis" / "pend-gdis-1960-2018-disasterlocations.gpkg"
    require_paths({"gdis": gdis_path})

    disno = str(disasterno_base).upper().strip()
    iso = str(project_iso).upper().strip()
    gdis_subset = gpd.read_file(gdis_path, layer="GPKG", where=f"disasterno = '{disno}'")
    if gdis_subset.empty:
        raise ValueError(f"Failed to find any GDIS geometry rows for disasterno {disno}")

    gdis_subset = gdis_subset.copy()
    gdis_subset["disastertype_norm"] = gdis_subset["disastertype"].astype(str).str.lower().str.strip()
    gdis_subset = gdis_subset[gdis_subset["disastertype_norm"] == "storm"].copy()
    gdis_subset["disasterno"] = gdis_subset["disasterno"].astype(str).str.upper().str.strip()
    gdis_subset["iso3"] = gdis_subset["iso3"].astype(str).str.upper().str.strip()

    disno_map, pair_map = build_gdis_geom_maps(gdis_subset)
    selected_key = f"{disno}|{iso}" if (disno, iso) in pair_map else disno
    selected_geom = pair_map.get((disno, iso), disno_map.get(disno))
    if selected_geom is None:
        raise ValueError(f"Failed to resolve a GDIS geometry using benchmark fallback logic for {disno} / {iso}")
    if expected_key is not None and str(expected_key) != selected_key:
        raise ValueError(f"Selected GDIS key {selected_key} does not match stored benchmark key {expected_key}")

    out = gpd.GeoDataFrame({"gdis_key": [selected_key]}, geometry=[selected_geom], crs=gdis_subset.crs)
    return out, selected_key


def read_region_countries(raw_root: Path, bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    gadm_path = raw_root / "gadm" / "gadm_410-levels.gpkg"
    return gpd.read_file(gadm_path, layer="ADM_0", bbox=bbox)


def find_imerg_file(raw_root: Path, target_date: date) -> Path:
    pattern = f"3B-DAY.MS.MRG.3IMERG.{target_date:%Y%m%d}-*.nc4"
    matches = sorted((raw_root / "gpm imerg" / "data").glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No IMERG daily file found for {target_date:%Y-%m-%d} with pattern {pattern}")
    return matches[0]


def build_imerg_event_grid(raw_root: Path, start_date: date, end_date: date, extent: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first_file = find_imerg_file(raw_root, start_date)
    with Dataset(first_file) as ds:
        full_lon = np.asarray(ds.variables["lon"][:], dtype=float)
        full_lat = np.asarray(ds.variables["lat"][:], dtype=float)

    lon_mask = (full_lon >= extent[0] - 0.05) & (full_lon <= extent[2] + 0.05)
    lat_mask = (full_lat >= extent[1] - 0.05) & (full_lat <= extent[3] + 0.05)
    local_lon = full_lon[lon_mask]
    local_lat = full_lat[lat_mask]
    event_grid = np.zeros((local_lat.size, local_lon.size), dtype=np.float64)

    for current_date in daterange(start_date, end_date):
        path = find_imerg_file(raw_root, current_date)
        with Dataset(path) as ds:
            field = np.asarray(ds.variables["precipitation"][0, :, :], dtype=float)
            field = field[np.ix_(lon_mask, lat_mask)].T
            event_grid += field

    return local_lon, local_lat, event_grid


def build_project_event_grid(case_group: dict[str, object], local_lon: np.ndarray, local_lat: np.ndarray, start_date: date, end_date: date) -> np.ndarray:
    all_times: list[datetime] = case_group["all_times"]
    center_lat = case_group["center_lat"]
    center_lon = case_group["center_lon"]
    window_lat = np.asarray(case_group["window_lat"], dtype=float)
    window_lon = np.asarray(case_group["window_lon"], dtype=float)
    hazard_rain = np.asarray(case_group["hazard_rain"], dtype=float)

    dlon = float(local_lon[1] - local_lon[0])
    dlat = float(local_lat[1] - local_lat[0])
    event_grid = np.zeros((local_lat.size, local_lon.size), dtype=np.float64)

    for current_date in daterange(start_date, end_date):
        day_indices = [i for i, dt in enumerate(all_times) if dt.date() == current_date]
        if not day_indices:
            continue
        day_sum = np.zeros_like(event_grid)
        for idx in day_indices:
            center_lat_i = float(center_lat[idx])
            center_lon_i = float(center_lon[idx])
            field = np.asarray(hazard_rain[idx], dtype=float)
            lat_abs = center_lat_i + np.broadcast_to(window_lat[:, None], field.shape)
            lon_abs = center_lon_i + np.broadcast_to(window_lon[None, :], field.shape)
            lat_idx = np.rint((lat_abs - local_lat[0]) / dlat).astype(int)
            lon_idx = np.rint((lon_abs - local_lon[0]) / dlon).astype(int)
            valid = (
                np.isfinite(field)
                & (lat_idx >= 0)
                & (lat_idx < local_lat.size)
                & (lon_idx >= 0)
                & (lon_idx < local_lon.size)
            )
            np.add.at(day_sum, (lat_idx[valid], lon_idx[valid]), field[valid])
        event_grid += day_sum / float(len(day_indices))

    return event_grid


def topq_mask(grid: np.ndarray, topq: float = 0.10) -> np.ndarray:
    valid = np.isfinite(grid)
    if not np.any(valid):
        return np.zeros_like(grid, dtype=bool)
    values = np.asarray(grid[valid], dtype=float)
    threshold = float(np.nanquantile(values, max(0.0, min(1.0, 1.0 - topq))))
    return valid & (grid >= threshold)


def case_window_line(case_group: dict[str, object], start_date: date, end_date: date) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_times: list[datetime] = case_group["all_times"]
    center_lat = np.asarray(case_group["center_lat"], dtype=float)
    center_lon = np.asarray(case_group["center_lon"], dtype=float)
    mask = np.array([start_date <= dt.date() <= end_date for dt in all_times], dtype=bool)
    return center_lon, center_lat, center_lon[mask], center_lat[mask]


def set_extent(ax, extent: tuple[float, float, float, float]) -> None:
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])


def plot_boundaries(ax, countries: gpd.GeoDataFrame, linecolor: str = COLORS["boundary"], linewidth: float = 1.5) -> None:
    if countries.empty:
        return
    countries.plot(ax=ax, facecolor=COLORS["country_fill"], edgecolor="none", linewidth=0.0, zorder=0)
    countries.boundary.plot(ax=ax, color=linecolor, linewidth=linewidth, zorder=2)


def plot_gdis_alignment(ax, countries: gpd.GeoDataFrame, gdis_geom: gpd.GeoDataFrame, full_lon: np.ndarray, full_lat: np.ndarray, window_lon: np.ndarray, window_lat: np.ndarray, event_points: pd.DataFrame, event_hull, extent: tuple[float, float, float, float], base_extent: tuple[float, float, float, float] | None = None) -> None:
    plot_boundaries(ax, countries, linewidth=1.5)

    if base_extent is None:
        base_extent = extent

    base_width = base_extent[2] - base_extent[0]
    current_width = extent[2] - extent[0]
    z_ratio = base_width / current_width

    lw_scale = z_ratio
    s_scale = z_ratio ** 2

    gdis_geom.plot(
        ax=ax,
        facecolor=COLORS["gdis_fill"],
        edgecolor="none",
        alpha=0.78,
        zorder=3,
    )

    gdis_geom.boundary.plot(
        ax=ax,
        color=COLORS["gdis_edge"],
        linewidth=0.8,
        zorder=11,
    )

    ax.plot(full_lon, full_lat, color=COLORS["track"], linewidth=1.8 * lw_scale, alpha=0.45, zorder=4)
    ax.plot(window_lon, window_lat, color=COLORS["track_window"], linewidth=2.6 * lw_scale, zorder=5)
    ax.scatter(window_lon, window_lat, s=24 * s_scale, color=COLORS["track_window"], zorder=6)

    point_specs = [
        ("hazard_only", COLORS["hazard_only_color"], 0.16, 7),
        ("loss_only", COLORS["loss_only_color"], 0.16, 8),
        ("overlap", COLORS["overlap_color"], 0.26, 9),
    ]

    base_s = 5 * s_scale

    for point_class, color, alpha, zorder in point_specs:
        sub = event_points[event_points["point_class"].astype(str) == point_class]
        if sub.empty:
            continue
        ax.scatter(sub["lon"], sub["lat"], s=base_s, marker="s", color=color, alpha=alpha, linewidths=0, zorder=zorder)

    if getattr(event_hull, "geom_type", "") == "Polygon":
        xs, ys = event_hull.exterior.xy
        ax.plot(xs, ys, color=COLORS["project_contour"], linewidth=1.5, linestyle="-", zorder=10)

    set_extent(ax, extent)
    minimalist_map_axes(ax, mean_lat=float(np.mean([base_extent[1], base_extent[3]])))


def plot_rain_map(ax, countries: gpd.GeoDataFrame, case_country: gpd.GeoDataFrame, rain_grid: np.ndarray, lon: np.ndarray, lat: np.ndarray, full_window_lon: np.ndarray, full_window_lat: np.ndarray, extent: tuple[float, float, float, float], norm: Normalize, cmap: str = "YlGnBu"):
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(alpha=0.0)
    rain_masked = np.ma.masked_where((~np.isfinite(rain_grid)) | (rain_grid <= RAIN_DISPLAY_FLOOR), rain_grid)
    mesh = ax.pcolormesh(lon, lat, rain_masked, shading="auto", cmap=cmap_obj, norm=norm, zorder=1)
    plot_boundaries(ax, countries, linewidth=1.5)
    case_country.boundary.plot(ax=ax, color="#2B2B2B", linewidth=1.6, zorder=3)
    ax.plot(full_window_lon, full_window_lat, color="white", linewidth=4.0, alpha=0.9, zorder=4)
    ax.plot(full_window_lon, full_window_lat, color=COLORS["track_window"], linewidth=2.6, zorder=5)
    set_extent(ax, extent)
    minimalist_map_axes(ax, mean_lat=float(np.mean([extent[1], extent[3]])))
    return mesh


def plot_imerg_comparison(ax, countries: gpd.GeoDataFrame, case_country: gpd.GeoDataFrame, imerg_grid: np.ndarray, project_mask: np.ndarray, imerg_mask: np.ndarray, lon: np.ndarray, lat: np.ndarray, full_window_lon: np.ndarray, full_window_lat: np.ndarray, extent: tuple[float, float, float, float], norm: Normalize, case_row: pd.Series):
    mesh = plot_rain_map(ax=ax, countries=countries, case_country=case_country, rain_grid=imerg_grid, lon=lon, lat=lat, full_window_lon=full_window_lon, full_window_lat=full_window_lat, extent=extent, norm=norm)
    if np.any(project_mask):
        ax.contour(lon, lat, project_mask.astype(float), levels=[0.5], colors=["white"], linewidths=4.0, zorder=6)
        ax.contour(lon, lat, project_mask.astype(float), levels=[0.5], colors=[COLORS["project_contour"]], linewidths=2.5, zorder=7)
    if np.any(imerg_mask):
        ax.contour(lon, lat, imerg_mask.astype(float), levels=[0.5], colors=["white"], linewidths=4.0, zorder=6)
        ax.contour(lon, lat, imerg_mask.astype(float), levels=[0.5], colors=[COLORS["imerg_contour"]], linewidths=2.5, zorder=7)
    ax.text(
        0.985,
        0.96,
        f"Intersection over union = {float(case_row['imerg_event_topq_footprint_iou']):.3f}\n"
        f"Recall = {float(case_row['imerg_event_topq_footprint_recall']):.3f}\n"
        f"Peak lag = {float(case_row['imerg_event_peak_day_lag']):,.0f} d",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=METRIC_FONT_SIZE,
        color=COLORS["dark"],
        bbox={"facecolor": "white", "edgecolor": "black", "linewidth": 0.8, "boxstyle": "round,pad=0.4", "alpha": 0.7},
        zorder=8,
    )
    return mesh


def make_figure(output_png: Path, output_pdf: Path, dpi: int, case_context: CaseContext, local_countries: gpd.GeoDataFrame, case_country: gpd.GeoDataFrame, gdis_geom: gpd.GeoDataFrame, full_lon: np.ndarray, full_lat: np.ndarray, window_lon: np.ndarray, window_lat: np.ndarray, event_points: pd.DataFrame, event_hull, local_extent: tuple[float, float, float, float], gdis_zoom_extent: tuple[float, float, float, float], local_lon: np.ndarray, local_lat: np.ndarray, project_grid: np.ndarray, imerg_grid: np.ndarray) -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.sans-serif": FONT_STACK,
            "font.size": BASE_FONT_SIZE_PT,
            "axes.labelsize": BASE_FONT_SIZE_PT,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
        }
    )

    fig = plt.figure(figsize=(19.2, 9.0), facecolor="white")
    outer = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1.0, 1.0], hspace=0.12, wspace=0.08)

    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    ax_c = fig.add_subplot(outer[1, 0])
    ax_d = fig.add_subplot(outer[1, 1])

    positive = np.concatenate([project_grid[np.isfinite(project_grid) & (project_grid > 0.0)], imerg_grid[np.isfinite(imerg_grid) & (imerg_grid > 0.0)]])
    vmax = float(np.nanpercentile(positive, 99.0)) if positive.size else 1.0
    norm = Normalize(vmin=RAIN_DISPLAY_FLOOR, vmax=vmax)
    # Mirror the upstream L6 footprint metric definition used for IoU/Recall.
    project_mask = topq_mask(project_grid, topq=0.10)
    imerg_mask = topq_mask(imerg_grid, topq=0.10)

    draw_legend_axis(
        ax_a,
        [
            Line2D([0], [0], color=COLORS["gdis_fill"], marker="o", linestyle="None", markersize=10, alpha=0.90, markeredgewidth=0, label="GDIS geometry"),
            Line2D([0], [0], color=COLORS["track_window"], lw=2.6, marker="o", markersize=7, label="Window track"),
            Line2D([0], [0], color=COLORS["hazard_only_color"], marker="s", linestyle="None", markersize=8, alpha=0.85, markeredgewidth=0, label="Hazard top 10% cells"),
            Line2D([0], [0], color=COLORS["loss_only_color"], marker="s", linestyle="None", markersize=8, alpha=0.85, markeredgewidth=0, label="Loss-allocated cells"),
            Line2D([0], [0], color=COLORS["overlap_color"], marker="s", linestyle="None", markersize=8, alpha=0.85, markeredgewidth=0, label="Overlap cells"),
            Line2D([0], [0], color=COLORS["project_contour"], lw=2.0, label="Project event footprint"),
        ],
        ncol=3,
        letter="a",
        bbox_to_anchor=(0.09, 1.03, 0.91, 0.24),
        mode="expand",
        columnspacing=1.0,
    )
    draw_legend_axis(ax_b, [], ncol=1, letter="b")
    draw_legend_axis(ax_c, [], ncol=1, letter="c")
    draw_legend_axis(
        ax_d,
        [
            Line2D([0], [0], color=COLORS["project_contour"], lw=2.5, label="Project top 10% footprint"),
            Line2D([0], [0], color=COLORS["imerg_contour"], lw=2.5, label="IMERG top 10% footprint"),
        ],
        ncol=2,
        letter="d",
    )

    plot_gdis_alignment(ax_a, local_countries, gdis_geom, full_lon, full_lat, window_lon, window_lat, event_points, event_hull, extent=local_extent, base_extent=local_extent)

    zx_min, zy_min, zx_max, zy_max = gdis_zoom_extent
    rect = patches.Rectangle(
        (zx_min, zy_min),
        zx_max - zx_min,
        zy_max - zy_min,
        linewidth=1.5,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
        zorder=10,
    )
    ax_a.add_patch(rect)

    plot_gdis_alignment(ax_b, local_countries, gdis_geom, full_lon, full_lat, window_lon, window_lat, event_points, event_hull, extent=gdis_zoom_extent, base_extent=local_extent)

    mesh = plot_rain_map(ax=ax_c, countries=local_countries, case_country=case_country, rain_grid=project_grid, lon=local_lon, lat=local_lat, full_window_lon=window_lon, full_window_lat=window_lat, extent=local_extent, norm=norm)
    plot_imerg_comparison(ax=ax_d, countries=local_countries, case_country=case_country, imerg_grid=imerg_grid, project_mask=project_mask, imerg_mask=imerg_mask, lon=local_lon, lat=local_lat, full_window_lon=window_lon, full_window_lat=window_lat, extent=local_extent, norm=norm, case_row=case_context.case_row)

    ax_a.set_xlabel("")
    ax_b.set_xlabel("")
    ax_b.set_ylabel("")
    ax_d.set_ylabel("")

    cbar = fig.colorbar(mesh, ax=[ax_c, ax_d], orientation="horizontal", fraction=0.05, pad=0.12)
    cbar.ax.tick_params(labelsize=TICK_LABEL_SIZE, colors=COLORS["dark"])
    cbar.set_label("Event-total rainfall (mm)", fontsize=BASE_FONT_SIZE_PT, labelpad=10)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    case_context = read_case_context(args.results_root, args.final_root, args.case_disno, args.case_iso)
    case_country_gdf, _ = read_country_geometries(args.raw_root, args.case_iso)
    case_context.case_country = str(case_country_gdf.iloc[0]["COUNTRY"])

    group_data = read_case_group_data(case_context.final_nc_path, case_context.selected_group, case_context.pad_start, case_context.pad_end)
    event_points = extract_event_points(group_data, case_context.pad_start, case_context.pad_end, hazard_q=0.9)
    _, event_hull = compute_event_geometry(event_points)

    full_lon, full_lat, window_lon, window_lat = case_window_line(group_data, case_context.pad_start, case_context.pad_end)
    gdis_geom, _ = read_gdis_geometry(args.raw_root, case_context.disasterno_base, case_context.case_iso, expected_key=str(case_context.gdis_row["gdis_key"]))

    local_bounds = (
        min(float(event_points["lon"].min()), float(gdis_geom.total_bounds[0])),
        min(float(event_points["lat"].min()), float(gdis_geom.total_bounds[1])),
        max(float(event_points["lon"].max()), float(gdis_geom.total_bounds[2])),
        max(float(event_points["lat"].max()), float(gdis_geom.total_bounds[3])),
    )
    local_extent = round_extent(local_bounds, pad_lon=1.2, pad_lat=0.9)
    gdis_zoom_extent = build_zoom_extent(
        bounds=tuple(float(value) for value in gdis_geom.total_bounds),
        local_extent=local_extent,
        pad_lon=0.3,
        pad_lat=0.25,
        min_width=1.5,
        min_height=1.0,
    )

    local_countries = read_region_countries(args.raw_root, local_extent)
    local_lon, local_lat, imerg_grid = build_imerg_event_grid(args.raw_root, case_context.pad_start, case_context.pad_end, local_extent)
    project_grid = build_project_event_grid(group_data, local_lon, local_lat, case_context.pad_start, case_context.pad_end)

    project_total = float(np.nansum(project_grid))
    expected_project_total = float(case_context.case_row["event_total_project_rain"])
    if not np.isclose(project_total, expected_project_total, rtol=0.02, atol=1e-6):
        raise ValueError(f"Project event-total rainfall mismatch: computed {project_total:.6f} vs stored {expected_project_total:.6f}")
    if int(case_context.gdis_row["gdis_hit"]) != 1 or not np.isclose(float(case_context.gdis_row["distance_km"]), 0.0, atol=1e-12):
        raise ValueError("Representative case no longer matches the expected GDIS hit / distance values")
    if not np.isclose(float(case_context.case_row["imerg_event_peak_day_lag"]), 0.0, atol=1e-12):
        raise ValueError("Representative case no longer matches the expected IMERG peak-day lag")

    make_figure(
        output_png=args.output_png,
        output_pdf=args.output_pdf,
        dpi=args.dpi,
        case_context=case_context,
        local_countries=local_countries,
        case_country=case_country_gdf,
        gdis_geom=gdis_geom,
        full_lon=full_lon,
        full_lat=full_lat,
        window_lon=window_lon,
        window_lat=window_lat,
        event_points=event_points,
        event_hull=event_hull,
        local_extent=local_extent,
        gdis_zoom_extent=gdis_zoom_extent,
        local_lon=local_lon,
        local_lat=local_lat,
        project_grid=project_grid,
        imerg_grid=imerg_grid,
    )

    print(f"Saved PNG: {args.output_png}")
    print(f"Saved PDF: {args.output_pdf}")
    print(
        f"Case {case_context.case_disno} / {case_context.selected_group}: "
        + f"GDIS distance={float(case_context.gdis_row['distance_km']):.1f} km, "
        + f"IoU={float(case_context.case_row['imerg_event_topq_footprint_iou']):.3f}, "
        + f"Recall={float(case_context.case_row['imerg_event_topq_footprint_recall']):.3f}"
    )


if __name__ == "__main__":
    main()



