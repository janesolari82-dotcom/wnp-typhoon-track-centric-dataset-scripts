from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from matplotlib import pyplot as plt
from netCDF4 import Dataset, num2date
from PIL import Image
from rasterio.enums import Resampling
from rasterio.plot import show
from rasterio.windows import from_bounds

matplotlib.use("Agg")


@dataclass
class Inputs:
    nc_path: Path
    audit_match_path: Path
    gfd_tif_path: Path
    gfd_json_path: Path
    gadm_path: Path


def infer_project_root() -> Path:
    # benchmark/L6-7/make_haiyan_gif_compare.py -> project root
    return Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create GIF comparison for Typhoon Haiyan sliding-window integrated hazard vs GFD."
    )
    p.add_argument("--project-root", type=Path, default=infer_project_root())
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory, default: <project>/reproduction_v6/05_benchmark_results/test",
    )
    p.add_argument("--disno", type=str, default="2013-0433-CHN")
    p.add_argument("--group", type=str, default="2013_HAIYAN_89")
    p.add_argument("--gfd-event-id", type=str, default="4098")
    p.add_argument(
        "--compare-var",
        type=str,
        default="hazard_compound",
        choices=["hazard_compound", "hazard_rain", "hazard_wind_power"],
    )
    p.add_argument("--frame-stride", type=int, default=2)
    p.add_argument("--fps", type=int, default=3)
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional limit for debugging; 0 means use all sampled frames.",
    )
    return p.parse_args()


def resolve_inputs(project_root: Path, disno: str, gfd_event_id: str) -> Inputs:
    disno_year = int(str(disno).split("-")[0])
    repro = project_root / "reproduction_v6"
    raw_using = project_root / "data" / "raw" / "using"

    nc_path = (
        repro
        / "04_final_output"
        / "emdat_attribution_integration"
        / f"typhoon_{disno_year}_ocean_litpop_poplight_emdat.nc"
    )
    audit_match_path = (
        repro
        / "04_final_output"
        / "emdat_attribution_integration"
        / "audit"
        / f"emdat_record_match_{disno_year}.csv"
    )
    gfd_root = raw_using / "GFD" / "gfd_unwrap"
    gfd_candidates = sorted(gfd_root.glob(f"DFO_{gfd_event_id}_From_*"))
    if not gfd_candidates:
        raise FileNotFoundError(f"GFD folder not found for event id={gfd_event_id} under {gfd_root}")
    gfd_dir = gfd_candidates[0]
    tif_candidates = sorted(gfd_dir.glob(f"DFO_{gfd_event_id}_*.tif"))
    if not tif_candidates:
        raise FileNotFoundError(f"GFD tif not found in {gfd_dir}")
    gfd_tif_path = tif_candidates[0]
    gfd_json_path = gfd_dir / f"DFO_{gfd_event_id}_properties.json"
    if not gfd_json_path.exists():
        raise FileNotFoundError(f"GFD properties json missing: {gfd_json_path}")

    gadm_path = raw_using / "gadm" / "gadm_410-levels.gpkg"
    if not gadm_path.exists():
        raise FileNotFoundError(f"GADM gpkg missing: {gadm_path}")

    return Inputs(
        nc_path=nc_path,
        audit_match_path=audit_match_path,
        gfd_tif_path=gfd_tif_path,
        gfd_json_path=gfd_json_path,
        gadm_path=gadm_path,
    )


def get_compare_var_name(compare_var: str, var_names: set[str]) -> str:
    mapping = {
        "hazard_compound": ["hazard_compound_daily", "hazard_compound"],
        "hazard_rain": ["hazard_rain_daily", "hazard_rain"],
        "hazard_wind_power": ["hazard_wind_power_daily", "hazard_wind_power"],
    }
    candidates = mapping[compare_var]
    for c in candidates:
        if c in var_names:
            return c
    raise KeyError(f"compare variable {compare_var} not found in group variables")


def validate_binding(audit_match_path: Path, disno: str, group_name: str) -> pd.Series:
    if not audit_match_path.exists():
        raise FileNotFoundError(f"audit match csv missing: {audit_match_path}")
    mdf = pd.read_csv(audit_match_path)
    required = {"DisNo", "ISO", "EventName", "PadStart", "PadEnd", "selected_groups"}
    miss = sorted(required - set(mdf.columns))
    if miss:
        raise RuntimeError(f"audit csv missing required columns: {miss}")

    rows = mdf[mdf["DisNo"].astype(str).str.upper() == str(disno).upper()].copy()
    rows = rows[rows["selected_groups"].astype(str).str.contains(re.escape(group_name), na=False)].copy()
    if rows.empty:
        raise RuntimeError(f"cannot find disno={disno} and selected_groups containing {group_name} in {audit_match_path}")
    return rows.iloc[0]


def load_map_layer(gadm_path: Path, full_bounds: tuple[float, float, float, float]) -> tuple[gpd.GeoDataFrame, str]:
    minx, miny, maxx, maxy = full_bounds
    bbox = (minx - 6.0, miny - 6.0, maxx + 6.0, maxy + 6.0)

    # Prefer Natural Earth from geopandas if available.
    try:
        ne_path = gpd.datasets.get_path("naturalearth_lowres")
        gdf = gpd.read_file(ne_path, bbox=bbox)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        else:
            gdf = gdf.to_crs(epsg=4326)
        return gdf, "naturalearth_lowres"
    except Exception:
        pass

    # Fallback: local GADM ADM_0.
    gdf = gpd.read_file(gadm_path, layer="ADM_0", engine="pyogrio", bbox=bbox)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)
    return gdf, "gadm_adm0"


def draw_base_map(ax, land_gdf: gpd.GeoDataFrame, extent: tuple[float, float, float, float]) -> None:
    minx, miny, maxx, maxy = extent
    ax.set_facecolor("#d9ecff")
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    if not land_gdf.empty:
        try:
            sub = land_gdf.cx[minx:maxx, miny:maxy]
        except Exception:
            sub = land_gdf
        if not sub.empty:
            sub.plot(ax=ax, color="#eeeeee", edgecolor="none", alpha=1.0, zorder=0)
            sub.boundary.plot(ax=ax, color="#5e5e5e", linewidth=0.6, zorder=1)
    ax.grid(alpha=0.25, linewidth=0.4, linestyle="--")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def compute_duration_vmax(gfd_tif_path: Path) -> float:
    with rasterio.open(gfd_tif_path) as ds:
        out_h = min(600, max(120, ds.height // 16))
        out_w = min(600, max(120, ds.width // 16))
        b2 = ds.read(
            2,
            out_shape=(out_h, out_w),
            resampling=Resampling.bilinear,
        )
    vals = b2[np.isfinite(b2) & (b2 > 0)]
    if vals.size == 0:
        return 1.0
    vmax = float(np.quantile(vals, 0.99))
    return vmax if vmax > 0 else 1.0


def to_time_str(t) -> str:
    return f"{int(t.year):04d}-{int(t.month):02d}-{int(t.day):02d} {int(getattr(t, 'hour', 0)):02d}:00"


def compose_gif(frame_paths: list[Path], gif_path: Path, fps: int) -> None:
    if not frame_paths:
        raise RuntimeError("no frame png generated, cannot compose gif")
    images = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in frame_paths]
    duration_ms = int(round(1000.0 / max(1, fps)))
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=duration_ms,
        loop=0,
    )
    for im in images:
        im.close()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else (project_root / "reproduction_v6" / "05_benchmark_results" / "test").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = resolve_inputs(project_root, args.disno, args.gfd_event_id)
    bind_row = validate_binding(inputs.audit_match_path, args.disno, args.group)
    gfd_props = json.loads(inputs.gfd_json_path.read_text(encoding="utf-8"))

    with Dataset(inputs.nc_path, "r") as ds:
        if args.group not in ds.groups:
            raise KeyError(f"group {args.group} not found in nc: {inputs.nc_path}")
        grp = ds.groups[args.group]
        var_name = get_compare_var_name(args.compare_var, set(grp.variables.keys()))

        tvals = np.asarray(grp.variables["time"][:], dtype=np.float64)
        tunits = str(grp.variables["time"].units)
        times = num2date(tvals, units=tunits, calendar="standard")

        center_lat = np.asarray(grp.variables["center_lat"][:], dtype=np.float32)
        center_lon = np.asarray(grp.variables["center_lon"][:], dtype=np.float32)
        window_lat = np.asarray(grp.variables["window_lat"][:], dtype=np.float32)
        window_lon = np.asarray(grp.variables["window_lon"][:], dtype=np.float32)
        hazard = np.asarray(grp.variables[var_name][:], dtype=np.float32)

    if hazard.ndim != 3:
        raise RuntimeError(f"unexpected hazard shape: {hazard.shape}, expected 3D")

    frame_stride = max(1, int(args.frame_stride))
    frame_idx = list(range(0, hazard.shape[0], frame_stride))
    if int(args.max_frames) > 0:
        frame_idx = frame_idx[: int(args.max_frames)]
    if not frame_idx:
        raise RuntimeError("no frame index selected")

    # Color scale fixed across all selected frames for visual consistency.
    selected_vals = hazard[frame_idx, :, :]
    vals = selected_vals[np.isfinite(selected_vals)]
    if vals.size == 0:
        hz_vmin, hz_vmax = 0.0, 1.0
    else:
        hz_vmin = float(np.quantile(vals, 0.01))
        hz_vmax = float(np.quantile(vals, 0.99))
        if hz_vmax <= hz_vmin:
            hz_vmax = hz_vmin + 1e-6

    # Full trajectory envelope for map fallback loading.
    wlat2d = window_lat[:, None]
    wlon2d = window_lon[None, :]
    full_minx = float(np.min(center_lon + np.min(window_lon)))
    full_maxx = float(np.max(center_lon + np.max(window_lon)))
    full_miny = float(np.min(center_lat + np.min(window_lat)))
    full_maxy = float(np.max(center_lat + np.max(window_lat)))
    map_layer, map_source = load_map_layer(inputs.gadm_path, (full_minx, full_miny, full_maxx, full_maxy))

    gfd_duration_vmax = compute_duration_vmax(inputs.gfd_tif_path)

    out_prefix = f"typhoon_{args.disno.replace('-', '_')}_vs_gfd{args.gfd_event_id}"
    frames_dir = output_dir / f"{out_prefix}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for p in frames_dir.glob("frame_*.png"):
        p.unlink()

    frame_rows: list[dict[str, object]] = []
    frame_paths: list[Path] = []

    with rasterio.open(inputs.gfd_tif_path) as gfd_ds:
        for k, t_idx in enumerate(frame_idx, start=1):
            c_lat = float(center_lat[t_idx])
            c_lon = float(center_lon[t_idx])
            lat_abs = c_lat + wlat2d
            lon_abs = c_lon + wlon2d
            h2d = np.asarray(hazard[t_idx, :, :], dtype=np.float32)

            minx = float(np.nanmin(lon_abs))
            maxx = float(np.nanmax(lon_abs))
            miny = float(np.nanmin(lat_abs))
            maxy = float(np.nanmax(lat_abs))
            extent = (minx, miny, maxx, maxy)

            fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.8, 5.8), dpi=int(args.dpi))
            draw_base_map(ax_l, map_layer, extent)
            draw_base_map(ax_r, map_layer, extent)

            # Left panel: integrated variable.
            im_l = ax_l.pcolormesh(
                lon_abs,
                lat_abs,
                h2d,
                shading="auto",
                cmap="YlOrRd",
                vmin=hz_vmin,
                vmax=hz_vmax,
                alpha=0.9,
                zorder=2,
            )
            ax_l.plot(center_lon[: t_idx + 1], center_lat[: t_idx + 1], color="#1f77b4", linewidth=1.6, zorder=3)
            ax_l.scatter([c_lon], [c_lat], color="#d62728", s=25, zorder=4)
            ax_l.set_title(f"Integrated {args.compare_var}")
            cbar_l = fig.colorbar(im_l, ax=ax_l, fraction=0.046, pad=0.04)
            cbar_l.set_label(args.compare_var)

            # Right panel: GFD flooded + duration in same extent.
            try:
                w = from_bounds(minx, miny, maxx, maxy, transform=gfd_ds.transform)
                b1 = gfd_ds.read(1, window=w, boundless=True, fill_value=0)
                b2 = gfd_ds.read(2, window=w, boundless=True, fill_value=0)
                wt = gfd_ds.window_transform(w)

                duration = np.where(b2 > 0, b2.astype(np.float32), np.nan)
                flooded = np.where(b1 > 0, 1.0, np.nan).astype(np.float32)
                show(duration, transform=wt, ax=ax_r, cmap="Blues", alpha=0.65, vmin=0.0, vmax=gfd_duration_vmax, zorder=2)
                show(flooded, transform=wt, ax=ax_r, cmap="Reds", alpha=0.35, vmin=0.0, vmax=1.0, zorder=3)
                ax_r.set_xlim(minx, maxx)
                ax_r.set_ylim(miny, maxy)
            except Exception:
                ax_r.text(
                    0.5,
                    0.5,
                    "GFD crop unavailable",
                    transform=ax_r.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#444444",
                )
            ax_r.plot(center_lon[: t_idx + 1], center_lat[: t_idx + 1], color="#1f77b4", linewidth=1.6, zorder=4)
            ax_r.scatter([c_lon], [c_lat], color="#d62728", s=25, zorder=5)
            ax_r.set_title("GFD: duration (blue) + flooded mask (red)")

            ts = to_time_str(times[t_idx])
            fig.suptitle(
                f"{args.disno} | {args.group} | GFD {args.gfd_event_id} | frame {k}/{len(frame_idx)} | {ts}",
                fontsize=11,
            )

            frame_path = frames_dir / f"frame_{k:04d}.png"
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            fig.savefig(frame_path)
            plt.close(fig)

            frame_paths.append(frame_path)
            frame_rows.append(
                {
                    "frame_id": k,
                    "time_index": int(t_idx),
                    "time_iso": ts,
                    "center_lat": c_lat,
                    "center_lon": c_lon,
                    "lon_min": minx,
                    "lon_max": maxx,
                    "lat_min": miny,
                    "lat_max": maxy,
                    "png_file": frame_path.name,
                }
            )
            print(f"[{k:03d}/{len(frame_idx):03d}] saved {frame_path.name}")

    gif_path = output_dir / f"{out_prefix}.gif"
    compose_gif(frame_paths, gif_path, fps=max(1, int(args.fps)))
    print(f"GIF saved: {gif_path}")

    frame_index_csv = output_dir / f"{out_prefix}_frame_index.csv"
    pd.DataFrame(frame_rows).to_csv(frame_index_csv, index=False, encoding="utf-8-sig")

    notes_path = output_dir / f"{out_prefix}_notes.md"
    notes_lines = [
        "# Typhoon GIF Comparison Notes",
        "",
        "## Binding",
        "",
        f"- disno: {args.disno}",
        f"- group: {args.group}",
        f"- gfd_event_id: {args.gfd_event_id}",
        f"- compare_var: {args.compare_var}",
        f"- compare_var_nc_name: {var_name}",
        f"- map_layer_source: {map_source}",
        "",
        "## Data Sources",
        "",
        f"- nc_path: {inputs.nc_path}",
        f"- audit_match_path: {inputs.audit_match_path}",
        f"- gfd_tif_path: {inputs.gfd_tif_path}",
        f"- gfd_json_path: {inputs.gfd_json_path}",
        f"- gadm_path: {inputs.gadm_path}",
        "",
        "## Event Rows",
        "",
        f"- EventName: {bind_row.get('EventName', '')}",
        f"- ISO: {bind_row.get('ISO', '')}",
        f"- PadStart: {bind_row.get('PadStart', '')}",
        f"- PadEnd: {bind_row.get('PadEnd', '')}",
        "",
        "## Render Parameters",
        "",
        f"- frame_stride: {frame_stride}",
        f"- selected_frames: {len(frame_paths)}",
        f"- fps: {max(1, int(args.fps))}",
        f"- dpi: {int(args.dpi)}",
        f"- hazard_vmin: {hz_vmin:.6f}",
        f"- hazard_vmax: {hz_vmax:.6f}",
        f"- gfd_duration_vmax: {gfd_duration_vmax:.6f}",
        "",
        "## GFD Properties",
        "",
        f"- began: {gfd_props.get('began', '')}",
        f"- ended: {gfd_props.get('ended', '')}",
        f"- cc: {gfd_props.get('cc', '')}",
        f"- dfo_main_cause: {gfd_props.get('dfo_main_cause', '')}",
        "",
        "## Outputs",
        "",
        f"- gif: {gif_path.name}",
        f"- frames_dir: {frames_dir.name}",
        f"- frame_index: {frame_index_csv.name}",
        "",
        "Legend:",
        "- Left panel: integrated hazard field + trajectory.",
        "- Right panel: GFD duration (blue) + flooded mask (red) in same map extent.",
        "- Background: ocean (blue) / land (gray) / coastline.",
        "",
    ]
    notes_path.write_text("\n".join(notes_lines), encoding="utf-8")

    print(f"Frame index saved: {frame_index_csv}")
    print(f"Notes saved: {notes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
