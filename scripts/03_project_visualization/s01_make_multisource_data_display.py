#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import xarray as xr
from cartopy import crs as ccrs
from matplotlib.colors import LogNorm, Normalize, PowerNorm, TwoSlopeNorm
from matplotlib.ticker import FixedLocator
from rasterio.enums import Resampling
from shapely.geometry import box


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPRO_ROOT = PROJECT_ROOT / "reproduction_v6"

POP_ROOT = REPRO_ROOT / "02_processed_data" / "population_log_pchip_v1" / "01_population_annual"
NTL_ROOT = REPRO_ROOT / "02_processed_data" / "nightlight_harmonization_v1" / "09_harmonized_ntl"
LITPOP_ROOT = REPRO_ROOT / "02_processed_data" / "litpop_2000_2024_v1" / "01_litpop_annual"
ERA5_ROOT = PROJECT_ROOT / "data" / "raw" / "using" / "era5"
OCEAN_SURFACE_ROOT = PROJECT_ROOT / "data" / "raw" / "using" / "copernicus_download_split" / "surface"
OCEAN_THETAO_ROOT = PROJECT_ROOT / "data" / "raw" / "using" / "copernicus_download_split" / "thetao"
OUTPUT_DIR = REPRO_ROOT / "visualization" / "data_display"

STUDY_REGION = {
    "lon_min": 98.0,
    "lon_max": 182.079,
    "lat_min": -1.4,
    "lat_max": 64.17,
}

FIGURE_DEFAULT = REPRO_ROOT / "visualization" / "data_display" / "multisource_data_display_2013_2013-11-08_v7.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a publication-style multi-panel figure for key gridded data layers.")
    parser.add_argument("--year", type=int, default=2013, help="Reference year for annual layers.")
    parser.add_argument("--date", type=str, default="2013-11-08", help="Reference date (YYYY-MM-DD) for daily layers.")
    parser.add_argument("--output", type=Path, default=FIGURE_DEFAULT, help="Output figure path.")
    parser.add_argument("--downsample-width", type=int, default=960, help="Target width for annual raster downsampling.")
    return parser.parse_args()


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "stix",
            "font.size": 14,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.titlesize": 15.5,
            "axes.linewidth": 0.8,
            "savefig.dpi": 300,
        }
    )


def add_base_map(ax, show_xlabels: bool, show_ylabels: bool) -> None:
    ax.set_facecolor("#dce8f2")
    ax.add_feature(cfeature.LAND, facecolor="#f3efe6", edgecolor="#8a8a8a", linewidth=0.25, zorder=1)
    ax.coastlines(resolution="110m", color="#666666", linewidth=0.35, zorder=2)
    ax.set_global()
    ax.set_extent([-180, 180, -60, 85], crs=ccrs.PlateCarree())

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.28,
        color="#9aa7b3",
        alpha=0.55,
        linestyle="--",
        zorder=0,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = show_xlabels
    gl.left_labels = show_ylabels
    gl.xlocator = FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = FixedLocator(np.arange(-60, 91, 30))
    gl.xlabel_style = {"size": 12}
    gl.ylabel_style = {"size": 12}

    region_box = box(
        STUDY_REGION["lon_min"],
        STUDY_REGION["lat_min"],
        STUDY_REGION["lon_max"],
        STUDY_REGION["lat_max"],
    )
    ax.add_geometries(
        [region_box],
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="#c62828",
        linewidth=1.6,
        zorder=4,
    )


def robust_limits(arr: np.ndarray, low: float, high: float, positive: bool = False) -> tuple[float, float]:
    vals = np.asarray(arr, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if positive:
        vals = vals[vals > 0]
    if vals.size == 0:
        return (1.0, 2.0) if positive else (-1.0, 1.0)
    vmin = float(np.nanpercentile(vals, low))
    vmax = float(np.nanpercentile(vals, high))
    if positive:
        vmin = max(vmin, np.finfo(np.float32).tiny)
        if vmax <= vmin:
            vmax = vmin * 10.0
    else:
        if vmax <= vmin:
            vmax = vmin + 1.0
    return vmin, vmax


def read_downsampled_raster(path: Path, width: int) -> tuple[np.ndarray, list[float]]:
    with rasterio.open(path) as ds:
        height = max(1, int(round(ds.height * (width / ds.width))))
        arr = ds.read(1, out_shape=(height, width), resampling=Resampling.average).astype(np.float32)
        nodata = ds.nodata
        bounds = ds.bounds
    arr[~np.isfinite(arr)] = np.nan
    if nodata is not None:
        arr[np.isclose(arr, nodata)] = np.nan
    return arr, [bounds.left, bounds.right, bounds.bottom, bounds.top]


def read_population(year: int, width: int) -> tuple[np.ndarray, list[float]]:
    arr, extent = read_downsampled_raster(POP_ROOT / f"population_{year}_30as.tif", width)
    arr = np.where(np.isfinite(arr), np.maximum(arr, 0.0), np.nan)
    return arr, extent


def read_nightlight(year: int, width: int) -> tuple[np.ndarray, list[float]]:
    arr, extent = read_downsampled_raster(NTL_ROOT / f"ntl_harmonized_{year}_dn_30as.tif", width)
    arr = np.where(np.isfinite(arr), np.maximum(arr, 0.0), np.nan)
    return arr, extent


def read_litpop(year: int, width: int) -> tuple[np.ndarray, list[float]]:
    arr, extent = read_downsampled_raster(LITPOP_ROOT / f"litpop_{year}_30as.tif", width)
    arr = np.where(np.isfinite(arr), np.maximum(arr, 0.0), np.nan)
    return arr, extent


def find_era5_files(year: int) -> dict[str, Path]:
    candidates = {p.name: p for p in ERA5_ROOT.rglob(f"*{year}.nc")}
    return {
        "u10": candidates[f"u10max_{year}.nc"],
        "v10": candidates[f"v10max_{year}.nc"],
        "tp": candidates[f"tp_{year}.nc"],
    }


def read_era5_var(reference_date: str, varname: str) -> tuple[np.ndarray, list[float]]:
    year = int(reference_date[:4])
    key_map = {"u10": "u10", "v10": "v10", "tp": "tp"}
    files = find_era5_files(year)
    with xr.open_dataset(files[varname], engine="h5netcdf") as ds:
        arr = ds[key_map[varname]].sel(valid_time=np.datetime64(reference_date)).values.astype(np.float32)
        lon = ds["longitude"].values.astype(np.float32)
        lat = ds["latitude"].values.astype(np.float32)
    arr[~np.isfinite(arr)] = np.nan
    extent = [float(lon.min()), float(lon.max() + 0.25), float(lat.min()), float(lat.max())]
    return arr, extent


def find_surface_file(reference_date: str) -> Path:
    dt = datetime.strptime(reference_date, "%Y-%m-%d")
    for path in sorted(OCEAN_SURFACE_ROOT.glob(f"glo_phy_surface_{dt.year}*.nc")):
        with xr.open_dataset(path) as ds:
            if np.datetime64(reference_date) in ds["time"].values:
                return path
    raise FileNotFoundError(f"No surface ocean file contains {reference_date}")


def read_surface_ocean(reference_date: str, variable: str) -> tuple[np.ndarray, list[float]]:
    path = find_surface_file(reference_date)
    with xr.open_dataset(path) as ds:
        arr = ds[variable].sel(time=np.datetime64(reference_date)).values.astype(np.float32)
        lon = ds["longitude"].values.astype(np.float32)
        lat = ds["latitude"].values.astype(np.float32)
    arr[~np.isfinite(arr)] = np.nan
    dlon = float(np.diff(lon[:2])[0])
    dlat = float(np.diff(lat[:2])[0])
    extent = [float(lon.min()), float(lon.max() + dlon), float(lat.min() - abs(dlat)), float(lat.max())]
    return arr, extent


def read_thetao_surface(reference_date: str) -> tuple[np.ndarray, list[float], float]:
    dt = datetime.strptime(reference_date, "%Y-%m-%d")
    path = OCEAN_THETAO_ROOT / f"{dt.year}" / f"{dt.month:02d}" / f"glo_phy_thetao_{dt:%Y%m%d}_{dt:%Y%m%d}.nc"
    if not path.exists():
        raise FileNotFoundError(f"Thetao file not found: {path}")
    with xr.open_dataset(path) as ds:
        depth0 = float(ds["depth"].values[0])
        arr = ds["thetao"].isel(time=0, depth=0).values.astype(np.float32)
        lon = ds["longitude"].values.astype(np.float32)
        lat = ds["latitude"].values.astype(np.float32)
    arr[~np.isfinite(arr)] = np.nan
    dlon = float(np.diff(lon[:2])[0])
    dlat = float(np.diff(lat[:2])[0])
    extent = [float(lon.min()), float(lon.max() + dlon), float(lat.min() - abs(dlat)), float(lat.max())]
    return arr, extent, depth0


def make_panel(ax, data: np.ndarray, extent: list[float], title: str, cmap_name: str, norm, cbar_label: str,
               show_xlabels: bool, show_ylabels: bool, under_color: str | None = None,
               origin: str = "upper") -> None:
    add_base_map(ax, show_xlabels=show_xlabels, show_ylabels=show_ylabels)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(alpha=0.0)
    if under_color is not None:
        cmap.set_under(under_color)
    im = ax.imshow(
        data,
        extent=extent,
        origin=origin,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        zorder=3,
        alpha=0.96,
    )
    ax.add_feature(
        cfeature.LAND,
        facecolor=(0.96, 0.94, 0.88, 0.18),
        edgecolor="#7d7d7d",
        linewidth=0.18,
        zorder=4.2,
    )
    ax.coastlines(resolution="110m", color="#555555", linewidth=0.45, zorder=4.5)
    ax.set_title(title, pad=5, fontweight="bold")
    cbar = ax.figure.colorbar(im, ax=ax, orientation="horizontal", fraction=0.055, pad=0.02, aspect=22)
    cbar.set_label(cbar_label, fontsize=14)
    cbar.ax.tick_params(labelsize=12)


def main() -> int:
    args = parse_args()
    set_style()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    thetao, thetao_extent, thetao_depth = read_thetao_surface(args.date)
    zos, zos_extent = read_surface_ocean(args.date, "zos")
    mlotst, mlotst_extent = read_surface_ocean(args.date, "mlotst")
    pop, pop_extent = read_population(args.year, args.downsample_width)
    ntl, ntl_extent = read_nightlight(args.year, args.downsample_width)
    litpop, litpop_extent = read_litpop(args.year, args.downsample_width)
    u10, u10_extent = read_era5_var(args.date, "u10")
    v10, v10_extent = read_era5_var(args.date, "v10")
    tp, tp_extent = read_era5_var(args.date, "tp")

    thetao_norm = Normalize(*robust_limits(thetao, 2, 98, positive=False))
    zos_lo, zos_hi = robust_limits(zos, 2, 98, positive=False)
    zos_norm = TwoSlopeNorm(vmin=zos_lo, vcenter=0.0, vmax=zos_hi)
    mlotst_norm = Normalize(*robust_limits(mlotst, 2, 98, positive=False))

    pop_norm = LogNorm(*robust_limits(pop, 5, 99.5, positive=True))
    ntl_norm = Normalize(vmin=0.0, vmax=63.0)
    litpop_norm = LogNorm(*robust_limits(litpop, 5, 99.5, positive=True))

    u10_lo, u10_hi = robust_limits(u10, 2, 98, positive=False)
    u10_norm = TwoSlopeNorm(vmin=u10_lo, vcenter=0.0, vmax=u10_hi)
    v10_lo, v10_hi = robust_limits(v10, 2, 98, positive=False)
    v10_norm = TwoSlopeNorm(vmin=v10_lo, vcenter=0.0, vmax=v10_hi)
    tp_norm = LogNorm(*robust_limits(tp, 5, 99.5, positive=True))

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(18.0, 14.0),
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
    )
    plt.subplots_adjust(left=0.045, right=0.975, bottom=0.06, top=0.975, wspace=0.03, hspace=0.06)

    panel_specs = [
        (axes[0, 0], thetao, thetao_extent, f"(a) Surface thetao ({args.date}, depth={thetao_depth:.2f} m)", "RdYlBu_r", thetao_norm, r"Potential temperature ($^\circ$C)", False, True, None, "lower"),
        (axes[0, 1], zos, zos_extent, f"(b) Sea surface height anomaly, zos ({args.date})", "RdBu_r", zos_norm, "Sea surface height anomaly (m)", False, False, None, "lower"),
        (axes[0, 2], mlotst, mlotst_extent, f"(c) Mixed layer thickness, mlotst ({args.date})", "viridis", mlotst_norm, "Mixed layer thickness (m)", False, False, None, "lower"),
        (axes[1, 0], pop, pop_extent, f"(d) Annual population count ({args.year})", "YlOrRd", pop_norm, "Population (persons)", False, True, "#edf4fb", "upper"),
        (axes[1, 1], ntl, ntl_extent, f"(e) Harmonized nightlight ({args.year})", "magma", ntl_norm, "DN-like intensity (0-63)", False, False, None, "upper"),
        (axes[1, 2], litpop, litpop_extent, f"(f) Annual LitPop exposure ({args.year})", "cividis", litpop_norm, "LitPop (USD)", False, False, "#edf4fb", "upper"),
        (axes[2, 0], u10, u10_extent, f"(g) ERA5 daily maximum u10 ({args.date})", "PuOr_r", u10_norm, r"u10 (m s$^{-1}$)", True, True, None, "upper"),
        (axes[2, 1], v10, v10_extent, f"(h) ERA5 daily maximum v10 ({args.date})", "BrBG", v10_norm, r"v10 (m s$^{-1}$)", True, False, None, "upper"),
        (axes[2, 2], tp, tp_extent, f"(i) ERA5 daily total precipitation ({args.date})", "GnBu", tp_norm, "Total precipitation (m)", True, False, "#edf4fb", "upper"),
    ]

    for spec in panel_specs:
        make_panel(*spec)

    fig.savefig(args.output, dpi=300)
    plt.close(fig)
    print(f"Saved figure to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
