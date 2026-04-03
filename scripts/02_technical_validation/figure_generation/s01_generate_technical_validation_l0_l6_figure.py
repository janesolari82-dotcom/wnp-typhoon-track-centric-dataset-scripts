#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, PercentFormatter, StrMethodFormatter

SCRIPT_DIR = Path(__file__).resolve().parent
REPRO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_ROOT = REPRO_ROOT / "05_benchmark_results" / "results_v2"
DEFAULT_OUTPUT_PNG = SCRIPT_DIR / "technical_validation_l0_l6_overview.png"
DEFAULT_OUTPUT_PDF = SCRIPT_DIR / "technical_validation_l0_l6_overview.pdf"

COLORS = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "gray": "#7A7A7A",
    "dark": "#1F2A36",
    "warning_fill": "#FBE6C2",
}

FONT_FAMILY = "sans-serif"
FONT_STACK = ["Helvetica", "Arial", "DejaVu Sans"]
BASE_FONT_SIZE_PT = 14
TICK_LABEL_SIZE = 12
LEGEND_FONT_SIZE = 12
PANEL_LABEL_SIZE = 18
METRIC_FONT_SIZE = 12
AXIS_LABEL_SIZE = BASE_FONT_SIZE_PT
COMMA_INT_FORMATTER = StrMethodFormatter("{x:,.0f}")

L5_STATUS_ORDER = ["allocated", "missing_loss", "zero_weight", "unmatched"]
L5_STATUS_COLORS = {
    "allocated": COLORS["green"],
    "missing_loss": COLORS["sky"],
    "zero_weight": COLORS["orange"],
    "unmatched": COLORS["vermillion"],
}
L3_STYLES = {
    "litpop": {"color": COLORS["blue"], "linestyle": "-", "marker": "o", "label": "LitPop"},
    "nightlight_intensity": {"color": COLORS["green"], "linestyle": "--", "marker": "s", "label": "Nightlight"},
    "population_count": {"color": COLORS["sky"], "linestyle": ":", "marker": "^", "label": "Population"},
}
L3_YEAR_OFFSETS = {"litpop": -0.10, "nightlight_intensity": 0.0, "population_count": 0.10}
ZERO_WEIGHT_LABELS = {
    "mask_all_zero": "Mask = 0",
    "litpop_all_zero_under_mask": "LitPop = 0",
}
ZERO_WEIGHT_COLORS = {
    "mask_all_zero": COLORS["orange"],
    "litpop_all_zero_under_mask": COLORS["sky"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a 7-panel Scientific Data-style L0-L6 validation figure.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT_PNG)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def require_paths(paths: dict[str, Path]) -> None:
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required results_v2 artifacts:\n" + "\n".join(missing))


def minimalist_axes(ax, grid_axis: str = "y", grid_alpha: float = 0.28) -> None:
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#D9DEE5", alpha=grid_alpha, linewidth=0.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#BFC7D1")
    ax.spines["bottom"].set_color("#BFC7D1")
    ax.tick_params(labelsize=TICK_LABEL_SIZE, colors=COLORS["dark"])


def set_year_ticks(ax, years: list[int], step: int = 4) -> None:
    years = sorted(int(y) for y in years)
    if not years:
        return
    ticks = years[::step]
    if years[-1] not in ticks:
        if years[-1] - ticks[-1] <= step / 2:
            ticks[-1] = years[-1]
        else:
            ticks.append(years[-1])
    ax.set_xticks(ticks)
    ax.set_xlim(min(years) - 0.6, max(years) + 0.6)

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


def style_legend(legend) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("#D5DCE5")
    frame.set_linewidth(0.8)
    frame.set_alpha(0.97)


def add_panel_header(ax, letter: str, handles: list | None = None, ncol: int = 2) -> None:
    pad_val = 10
    if handles:
        pad_val = 45 if len(handles) > 2 else 30

    # Use title padding to reserve real vertical space, then place the letter
    # and legend manually so letters with and without legends stay aligned.
    ax.set_title(" ", loc="left", fontsize=PANEL_LABEL_SIZE, pad=pad_val)
    header_y = 1.055
    label = f"({letter})"
    ax.text(
        -0.08,
        header_y,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        color=COLORS["dark"],
        va="bottom",
        ha="left",
    )

    if handles:
        ax.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0.08, 1.015),
            ncol=ncol,
            frameon=False,
            fontsize=LEGEND_FONT_SIZE,
            borderaxespad=0.0,
            columnspacing=1.2,
            handletextpad=0.4,
        )

def make_percent_boxplot(ax, data_arrays: list[np.ndarray], years: list[int], edge_color: str, ylim: tuple[float, float]) -> None:
    box = ax.boxplot(
        data_arrays,
        positions=years,
        widths=0.62,
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
        manage_ticks=False,
    )
    for artist in box["boxes"]:
        artist.set(facecolor="white", edgecolor=edge_color, linewidth=1.8)
    for artist in box["medians"]:
        artist.set(color=edge_color, linewidth=2.0)
    for artist in box["whiskers"]:
        artist.set(color=edge_color, linewidth=1.8)
    for artist in box["caps"]:
        artist.set(color=edge_color, linewidth=1.8)
    ax.set_ylim(*ylim)


def load_data(results_root: Path) -> dict[str, Any]:
    paths = {
        "l3_fill": results_root / "03_alignment" / "csv" / "fill_ratio_trend_by_year_plot_data.csv",
        "l5_allocation": results_root / "05_emdat" / "csv" / "allocation_status_stacked_by_year_plot_data.csv",
        "l5_conservation": results_root / "05_emdat" / "csv" / "conservation_error_boxplot_by_year_plot_data.csv",
        "l5_zero_diag": results_root / "05_emdat" / "csv" / "zero_weight_diagnosis_breakdown_plot_data.csv",
        "l6_gdis_match": results_root / "06_external_validation" / "csv" / "gdis_match_rate_trend_plot_data.csv",
        "l6_gdis_hit": results_root / "06_external_validation" / "csv" / "gdis_hit_rate_trend_plot_data.csv",
        "l6_gdis_dist": results_root / "06_external_validation" / "csv" / "gdis_distance_boxplot_plot_data.csv",
        "l6_imerg_conv": results_root / "06_external_validation" / "csv" / "imerg_rainfall_convergence_trend_plot_data.csv",
        "l6_imerg_overlap": results_root / "06_external_validation" / "csv" / "imerg_overlap_distribution_plot_data.csv",
        "l6_imerg_scatter": results_root / "06_external_validation" / "csv" / "imerg_event_total_rainfall_scatter_plot_data.csv",
    }
    require_paths(paths)
    return {name: read_csv(path) for name, path in paths.items()}


def build_metrics(data: dict[str, Any]) -> dict[str, Any]:
    fill_df = data["l3_fill"].copy()
    alloc_all = data["l5_allocation"].copy()
    alloc_all = alloc_all[alloc_all["scope"].astype(str) == "all_records"].copy()
    alloc_totals = alloc_all.groupby("status", dropna=False)["count"].sum().to_dict()

    gdis_match = data["l6_gdis_match"].copy()
    gdis_match = gdis_match[gdis_match["year"].astype(str) != "ALL"].copy()
    gdis_match["year"] = pd.to_numeric(gdis_match["year"], errors="coerce").astype(int)
    project_events = pd.to_numeric(gdis_match["project_event_count"], errors="coerce").sum()
    pair_match = pd.to_numeric(gdis_match["gdis_pair_match_count"], errors="coerce").sum()

    gdis_hit = data["l6_gdis_hit"].copy()
    gdis_hit["year"] = pd.to_numeric(gdis_hit["year"], errors="coerce").astype(int)
    sampled_events = pd.to_numeric(gdis_hit["sampled_events"], errors="coerce").to_numpy(dtype=float)
    hit_rate = pd.to_numeric(gdis_hit["gdis_geometry_hit_rate"], errors="coerce").to_numpy(dtype=float)

    gdis_dist = pd.to_numeric(data["l6_gdis_dist"]["distance_km"], errors="coerce").dropna().to_numpy(dtype=float)

    imerg_scatter = data["l6_imerg_scatter"].copy()
    x = pd.to_numeric(imerg_scatter["event_total_project_rain"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(imerg_scatter["event_total_imerg_rain"], errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    x = x[good]
    y = y[good]

    return {
        "l3_warn_years": sorted(
            pd.to_numeric(
                fill_df.loc[pd.to_numeric(fill_df["warn_3sigma"], errors="coerce").fillna(0) > 0, "year"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        ),
        "l5_total_records": int(sum(int(v) for v in alloc_totals.values())),
        "l5_totals": {str(k): int(v) for k, v in alloc_totals.items()},
        "l5_max_conservation_error": float(
            pd.to_numeric(data["l5_conservation"]["max_conservation_error"], errors="coerce").max()
        ),
        "l6_pair_match_rate": float(pair_match / project_events),
        "l6_geometry_hit_rate": float(np.sum(sampled_events * hit_rate) / np.sum(sampled_events)),
        "l6_distance_p95": float(np.nanquantile(gdis_dist, 0.95)),
        "l6_event_total_corr": float(np.corrcoef(x, y)[0, 1]),
        "l6_event_total_n": int(x.size),
        "l6_gdis_years": sorted(pd.to_numeric(data["l6_gdis_dist"]["year"], errors="coerce").dropna().astype(int).unique().tolist()),
        "l6_imerg_overlap_years": sorted(
            pd.to_numeric(data["l6_imerg_overlap"]["year"], errors="coerce").dropna().astype(int).unique().tolist()
        ),
    }


def run_sanity_checks(data: dict[str, Any], metrics: dict[str, Any]) -> None:
    if metrics["l3_warn_years"] != [2010]:
        raise ValueError("L3 expected a 2010-only fill anomaly")

    expected_l5 = {"allocated": 309, "missing_loss": 174, "zero_weight": 89, "unmatched": 4}
    if metrics["l5_total_records"] != 576 or any(int(metrics["l5_totals"].get(k, 0)) != v for k, v in expected_l5.items()):
        raise ValueError(f"L5 totals do not match the manuscript values: {metrics['l5_totals']}")
    if not np.isclose(metrics["l5_max_conservation_error"], 9.615743109163453e-08, atol=1e-15, rtol=0.0):
        raise ValueError("L5 conservation error does not match the stored benchmark result")

    if not np.isclose(metrics["l6_pair_match_rate"], 0.7471783295711061, atol=1e-12, rtol=0.0):
        raise ValueError("L6 GDIS pair match rate mismatch")
    if not np.isclose(metrics["l6_geometry_hit_rate"], 0.8578680203045685, atol=1e-12, rtol=0.0):
        raise ValueError("L6 GDIS hit rate mismatch")
    if not np.isclose(metrics["l6_distance_p95"], 610.4129687847316, atol=1e-9, rtol=0.0):
        raise ValueError("L6 GDIS p95 distance mismatch")

    conv_years = sorted(pd.to_numeric(data["l6_imerg_conv"]["year"], errors="coerce").dropna().astype(int).tolist())
    if conv_years != list(range(2000, 2021)):
        raise ValueError("L6 IMERG coverage window is not 2000-2020")
    if not (pd.to_numeric(data["l6_imerg_conv"]["imerg_hazard_rain_bias"], errors="coerce") < 0.0).all():
        raise ValueError("L6 IMERG annual bias should stay negative")
    if "imerg_top20pct_overlap_ratio" not in data["l6_imerg_overlap"].columns:
        raise ValueError("IMERG overlap distribution is missing the top20 overlap column")
    if metrics["l6_gdis_years"] != list(range(2000, 2019)):
        raise ValueError("GDIS distance distribution should cover 2000-2018 only")
    if metrics["l6_imerg_overlap_years"] != list(range(2000, 2021)):
        raise ValueError("IMERG overlap distribution should cover 2000-2020")
    if not np.isclose(metrics["l6_event_total_corr"], 0.9639234950666378, atol=1e-12, rtol=0.0):
        raise ValueError("L6 IMERG event-total correlation mismatch")


def plot_alignment_panel(ax, fill_df: pd.DataFrame, show_label: bool = True) -> None:
    if show_label:
        add_panel_label(ax, "a")
    fill_df = fill_df.copy()
    fill_df["year"] = pd.to_numeric(fill_df["year"], errors="coerce").astype(int)
    years = sorted(fill_df["year"].unique().tolist())
    ref = fill_df[fill_df["variable"].astype(str) == "litpop"].sort_values("year").copy()
    threshold = pd.to_numeric(ref["baseline_mean"], errors="coerce").to_numpy(dtype=float) + 3.0 * pd.to_numeric(
        ref["baseline_std"], errors="coerce"
    ).to_numpy(dtype=float)

    ax.axvspan(2009.5, 2010.5, color=COLORS["warning_fill"], alpha=0.85, zorder=0)
    ax.plot(ref["year"], threshold, color=COLORS["orange"], linewidth=2.5, linestyle="--", label="Baseline + 3 SD")

    year_2010_peak = 0.0
    for variable, style in L3_STYLES.items():
        sdf = fill_df[fill_df["variable"].astype(str) == variable].sort_values("year").copy()
        xs = pd.to_numeric(sdf["year"], errors="coerce").to_numpy(dtype=float) + L3_YEAR_OFFSETS[variable]
        ys = pd.to_numeric(sdf["fill_ratio"], errors="coerce").to_numpy(dtype=float)
        ax.plot(
            xs,
            ys,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.7,
            marker=style["marker"],
            markersize=6.5,
            label=style["label"],
            zorder=3,
        )
        if variable == "litpop":
            peak = sdf.loc[sdf["year"] == 2010, "fill_ratio"]
            if not peak.empty:
                year_2010_peak = float(peak.iloc[0])

    ax.set_ylabel("Missing ratio", fontsize=AXIS_LABEL_SIZE)
    ax.set_xlabel("Year", fontsize=AXIS_LABEL_SIZE)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{100.0 * y:.2f}%"))
    set_year_ticks(ax, years)
    minimalist_axes(ax)
    ax.annotate(
        "2010 missingness anomaly",
        xy=(2010, year_2010_peak),
        xycoords="data",
        xytext=(2018.0, 0.0038),
        textcoords="data",
        fontsize=METRIC_FONT_SIZE,
        color=COLORS["vermillion"],
        ha="center",
        va="bottom",
        arrowprops={
            "arrowstyle": "->",
            "lw": 1.5,
            "color": COLORS["vermillion"],
            "connectionstyle": "arc3,rad=-0.1",
        },
    )


def plot_audit_panel(ax, allocation_df: pd.DataFrame, show_label: bool = True) -> None:
    if show_label:
        add_panel_label(ax, "b")
    rows = allocation_df[allocation_df["scope"].astype(str) == "all_records"].copy()
    rows["year"] = pd.to_numeric(rows["year"], errors="coerce").astype(int)
    years = sorted(rows["year"].unique().tolist())
    year_to_vals = {year: {status: 0.0 for status in L5_STATUS_ORDER} for year in years}
    for row in rows.itertuples(index=False):
        year_to_vals[int(row.year)][str(row.status)] = float(row.ratio)

    bottom = np.zeros(len(years), dtype=float)
    for status in L5_STATUS_ORDER:
        values = np.array([year_to_vals[year][status] for year in years], dtype=float)
        ax.bar(
            years,
            values,
            bottom=bottom,
            width=0.82,
            color=L5_STATUS_COLORS[status],
            label=status.replace("_", "-"),
            zorder=3,
        )
        bottom += values

    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Audit record share", fontsize=AXIS_LABEL_SIZE)
    ax.set_xlabel("Year", fontsize=AXIS_LABEL_SIZE)
    set_year_ticks(ax, years)
    minimalist_axes(ax)


def plot_conservation_panel(ax, conservation_df: pd.DataFrame, show_label: bool = True) -> None:
    if show_label:
        add_panel_label(ax, "e")
    conservation_df = conservation_df.copy()
    conservation_df["year"] = pd.to_numeric(conservation_df["year"], errors="coerce").astype(int)
    conservation_df = conservation_df.sort_values("year")
    years = conservation_df["year"].tolist()
    cons_max = pd.to_numeric(conservation_df["max_conservation_error"], errors="coerce").to_numpy(dtype=float)
    positive = cons_max[np.isfinite(cons_max) & (cons_max > 0)]
    lower = max(float(np.min(positive)) * 0.8, 1e-9) if positive.size else 1e-9
    upper = max(1.1e-6, float(np.max(positive)) * 1.2) if positive.size else 1.1e-6

    ax.plot(years, cons_max, color=COLORS["blue"], linewidth=2.7, marker="o", markersize=6.5, zorder=3)
    ax.axhline(1e-6, color=COLORS["vermillion"], linestyle="--", linewidth=2.0, zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(lower, upper)
    ax.set_ylabel("Maximum relative error", fontsize=AXIS_LABEL_SIZE)
    ax.set_xlabel("Year", fontsize=AXIS_LABEL_SIZE)
    set_year_ticks(ax, years)
    minimalist_axes(ax)


def plot_zero_weight_panel(ax, zero_df: pd.DataFrame, show_label: bool = True) -> None:
    if show_label:
        add_panel_label(ax, "f")
    zero_df = zero_df.copy()
    zero_df["count"] = pd.to_numeric(zero_df["count"], errors="coerce").fillna(0).astype(int)
    zero_df = zero_df.sort_values("count", ascending=True)
    labels = [ZERO_WEIGHT_LABELS.get(str(x), str(x)) for x in zero_df["diagnosis_label"]]
    colors = [ZERO_WEIGHT_COLORS.get(str(x), COLORS["gray"]) for x in zero_df["diagnosis_label"]]
    positions = np.arange(len(labels))

    ax.barh(positions, zero_df["count"].to_numpy(dtype=float), color=colors, zorder=3)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=TICK_LABEL_SIZE)
    ax.set_xlabel("Count", fontsize=AXIS_LABEL_SIZE)
    ax.xaxis.set_major_formatter(COMMA_INT_FORMATTER)
    minimalist_axes(ax, grid_axis="x")


def plot_gdis_panel(ax_top, ax_bottom, match_df: pd.DataFrame, hit_df: pd.DataFrame, dist_df: pd.DataFrame, show_label: bool = True):
    if show_label:
        add_panel_label(ax_top, "c")

    match_df = match_df[match_df["year"].astype(str) != "ALL"].copy()
    match_df["year"] = pd.to_numeric(match_df["year"], errors="coerce").astype(int)
    match_df = match_df.sort_values("year")
    hit_df = hit_df.copy()
    hit_df["year"] = pd.to_numeric(hit_df["year"], errors="coerce").astype(int)
    hit_df = hit_df.sort_values("year")

    years = match_df["year"].tolist()
    pair_rate = pd.to_numeric(match_df["gdis_pair_match_rate"], errors="coerce").to_numpy(dtype=float)
    hit_rate = pd.to_numeric(hit_df["gdis_geometry_hit_rate"], errors="coerce").to_numpy(dtype=float)

    ax_top.plot(years, pair_rate, color=COLORS["blue"], linewidth=2.7, marker="o", markersize=6.5, label="Pair match", zorder=3)
    ax_top.plot(years, hit_rate, color=COLORS["green"], linewidth=2.7, marker="s", markersize=6.5, label="Geometry hit", zorder=3)
    ax_top.set_ylim(0.0, 1.05)
    ax_top.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax_top.set_ylabel("Validation rate", fontsize=AXIS_LABEL_SIZE)
    minimalist_axes(ax_top)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    dist_df = dist_df.copy()
    dist_df["year"] = pd.to_numeric(dist_df["year"], errors="coerce").astype(int)
    dist_df["distance_km"] = pd.to_numeric(dist_df["distance_km"], errors="coerce")
    dist_df = dist_df.dropna(subset=["distance_km"])
    dist_data = [dist_df.loc[dist_df["year"] == year, "distance_km"].to_numpy(dtype=float) for year in years]
    box = ax_bottom.boxplot(
        dist_data,
        positions=years,
        widths=0.62,
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
        manage_ticks=False,
    )
    for artist in box["boxes"]:
        artist.set(facecolor="white", edgecolor=COLORS["gray"], linewidth=1.8)
    for artist in box["medians"]:
        artist.set(color=COLORS["gray"], linewidth=2.0)
    for artist in box["whiskers"]:
        artist.set(color=COLORS["gray"], linewidth=1.8)
    for artist in box["caps"]:
        artist.set(color=COLORS["gray"], linewidth=1.8)
    ax_bottom.set_ylabel("Distance (km)", fontsize=AXIS_LABEL_SIZE)
    ax_bottom.yaxis.set_major_formatter(COMMA_INT_FORMATTER)
    ax_bottom.set_xlabel("Year", fontsize=AXIS_LABEL_SIZE)
    set_year_ticks(ax_bottom, years)
    minimalist_axes(ax_bottom, grid_axis="y", grid_alpha=0.22)

    return ax_top

def plot_imerg_panel(ax_top, ax_bottom, conv_df: pd.DataFrame, overlap_df: pd.DataFrame, show_label: bool = True):
    if show_label:
        add_panel_label(ax_top, "d")

    conv_df = conv_df.copy()
    conv_df["year"] = pd.to_numeric(conv_df["year"], errors="coerce").astype(int)
    conv_df = conv_df.sort_values("year")
    years = conv_df["year"].tolist()
    corr = pd.to_numeric(conv_df["imerg_hazard_rain_corr"], errors="coerce").to_numpy(dtype=float)
    overlap = pd.to_numeric(conv_df["imerg_top20pct_overlap_ratio"], errors="coerce").to_numpy(dtype=float)
    bias = pd.to_numeric(conv_df["imerg_hazard_rain_bias"], errors="coerce").to_numpy(dtype=float)

    ax_top.axvspan(2018.5, 2020.5, color=COLORS["warning_fill"], alpha=0.72, zorder=0)
    bias_ax = ax_top.twinx()
    bias_ax.bar(years, bias, color=COLORS["orange"], alpha=0.48, width=0.78, label="Bias (Released - IMERG)", zorder=1)
    bias_ax.axhline(0.0, color=COLORS["vermillion"], linewidth=1.8, zorder=2)
    bias_ax.set_ylabel("Rainfall bias", fontsize=AXIS_LABEL_SIZE, color=COLORS["vermillion"])
    bias_ax.tick_params(axis="y", colors=COLORS["vermillion"], labelsize=TICK_LABEL_SIZE)
    bias_ax.spines["top"].set_visible(False)
    bias_ax.spines["right"].set_color(COLORS["vermillion"])

    ax_top.set_zorder(2)
    ax_top.patch.set_alpha(0.0)
    ax_top.plot(years, corr, color=COLORS["blue"], linewidth=2.7, marker="o", markersize=6.5, label="Correlation", zorder=3)
    ax_top.plot(years, overlap, color=COLORS["green"], linewidth=2.7, marker="s", markersize=6.5, label="Top 20% overlap", zorder=3)
    ax_top.set_ylim(0.0, 1.0)
    ax_top.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax_top.set_ylabel("Annual agreement", fontsize=AXIS_LABEL_SIZE, color=COLORS["blue"])
    ax_top.tick_params(axis="y", colors=COLORS["blue"], labelsize=TICK_LABEL_SIZE)
    minimalist_axes(ax_top)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    overlap_df = overlap_df.copy()
    overlap_df["year"] = pd.to_numeric(overlap_df["year"], errors="coerce").astype(int)
    overlap_df["imerg_top20pct_overlap_ratio"] = pd.to_numeric(
        overlap_df["imerg_top20pct_overlap_ratio"], errors="coerce"
    )
    overlap_df = overlap_df.dropna(subset=["imerg_top20pct_overlap_ratio"])
    overlap_data = [
        overlap_df.loc[overlap_df["year"] == year, "imerg_top20pct_overlap_ratio"].to_numpy(dtype=float) for year in years
    ]
    make_percent_boxplot(ax_bottom, overlap_data, years, COLORS["green"], (0.0, 1.0))
    ax_bottom.axvspan(2018.5, 2020.5, color=COLORS["warning_fill"], alpha=0.52, zorder=0)
    ax_bottom.set_ylabel("Event-level\ntop 20% overlap", fontsize=AXIS_LABEL_SIZE)
    ax_bottom.set_xlabel("Year", fontsize=AXIS_LABEL_SIZE)
    ax_bottom.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    set_year_ticks(ax_bottom, years)
    minimalist_axes(ax_bottom, grid_axis="y", grid_alpha=0.22)

    return ax_top

def plot_imerg_scatter_panel(ax, scatter_df: pd.DataFrame, metrics: dict[str, Any], show_label: bool = True) -> None:
    if show_label:
        add_panel_label(ax, "g")
    x = pd.to_numeric(scatter_df["event_total_project_rain"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(scatter_df["event_total_imerg_rain"], errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    x = x[good]
    y = y[good]
    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))

    ax.scatter(x, y, s=45, color=COLORS["blue"], alpha=0.25, linewidths=0.0, zorder=3)
    ax.plot([lo, hi], [lo, hi], color=COLORS["gray"], linestyle="--", linewidth=2.0, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Released event-total rainfall", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("IMERG event-total rainfall", fontsize=AXIS_LABEL_SIZE)
    minimalist_axes(ax, grid_axis="both", grid_alpha=0.18)
    ax.set_box_aspect(1)
    ax.text(
        0.97,
        0.05,
        f"Correlation = {metrics['l6_event_total_corr']:.3f}\nEvents = {metrics['l6_event_total_n']:,}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=METRIC_FONT_SIZE,
        color=COLORS["dark"],
        bbox={"facecolor": "white", "edgecolor": "#D5DCE5", "boxstyle": "round,pad=0.4", "alpha": 0.96},
    )


def build_figure(data: dict[str, Any], metrics: dict[str, Any]):
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.sans-serif": FONT_STACK,
            "font.size": BASE_FONT_SIZE_PT,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
        }
    )

    fig = plt.figure(figsize=(15.0, 14.0), facecolor="white", layout="constrained")
    outer = fig.add_gridspec(nrows=4, ncols=6, height_ratios=[1.3, 1.0, 1.1, 1.5])

    ax_a = fig.add_subplot(outer[0, 0:3])
    ax_b = fig.add_subplot(outer[0, 3:6])

    ax_c_top = fig.add_subplot(outer[1, 0:3])
    ax_c_bot = fig.add_subplot(outer[2, 0:3], sharex=ax_c_top)

    ax_d_top = fig.add_subplot(outer[1, 3:6])
    ax_d_bot = fig.add_subplot(outer[2, 3:6], sharex=ax_d_top)

    ax_e = fig.add_subplot(outer[3, 0:2])
    ax_f = fig.add_subplot(outer[3, 2:4])
    ax_g = fig.add_subplot(outer[3, 4:6])

    plot_alignment_panel(ax_a, data["l3_fill"], show_label=False)
    plot_audit_panel(ax_b, data["l5_allocation"], show_label=False)

    plot_gdis_panel(ax_c_top, ax_c_bot, data["l6_gdis_match"], data["l6_gdis_hit"], data["l6_gdis_dist"], show_label=False)
    plot_imerg_panel(ax_d_top, ax_d_bot, data["l6_imerg_conv"], data["l6_imerg_overlap"], show_label=False)

    plot_conservation_panel(ax_e, data["l5_conservation"], show_label=False)
    plot_zero_weight_panel(ax_f, data["l5_zero_diag"], show_label=False)
    plot_imerg_scatter_panel(ax_g, data["l6_imerg_scatter"], metrics, show_label=False)

    add_panel_header(
        ax_a,
        "a",
        handles=[
            Line2D([0], [0], color=COLORS["orange"], lw=2.5, linestyle="--", label="Baseline + 3 SD"),
            Line2D([0], [0], color=L3_STYLES["litpop"]["color"], lw=2.7, linestyle=L3_STYLES["litpop"]["linestyle"], marker=L3_STYLES["litpop"]["marker"], markersize=7, label=L3_STYLES["litpop"]["label"]),
            Line2D([0], [0], color=L3_STYLES["nightlight_intensity"]["color"], lw=2.7, linestyle=L3_STYLES["nightlight_intensity"]["linestyle"], marker=L3_STYLES["nightlight_intensity"]["marker"], markersize=7, label=L3_STYLES["nightlight_intensity"]["label"]),
            Line2D([0], [0], color=L3_STYLES["population_count"]["color"], lw=2.7, linestyle=L3_STYLES["population_count"]["linestyle"], marker=L3_STYLES["population_count"]["marker"], markersize=7, label=L3_STYLES["population_count"]["label"]),
        ],
        ncol=2,
    )

    add_panel_header(
        ax_b,
        "b",
        handles=[
            Patch(facecolor=L5_STATUS_COLORS["allocated"], edgecolor="none", label="Allocated"),
            Patch(facecolor=L5_STATUS_COLORS["zero_weight"], edgecolor="none", label="Zero weight"),
            Patch(facecolor=L5_STATUS_COLORS["missing_loss"], edgecolor="none", label="Missing loss"),
            Patch(facecolor=L5_STATUS_COLORS["unmatched"], edgecolor="none", label="Unmatched"),
        ],
        ncol=2,
    )

    add_panel_header(
        ax_c_top,
        "c",
        handles=[
            Line2D([0], [0], color=COLORS["blue"], lw=2.7, marker="o", markersize=7, label="Pair match"),
            Line2D([0], [0], color=COLORS["green"], lw=2.7, marker="s", markersize=7, label="Geometry hit"),
        ],
        ncol=2,
    )

    add_panel_header(
        ax_d_top,
        "d",
        handles=[
            Line2D([0], [0], color=COLORS["blue"], lw=2.7, marker="o", markersize=7, label="Correlation"),
            Line2D([0], [0], color=COLORS["green"], lw=2.7, marker="s", markersize=7, label="Top 20% overlap"),
            Patch(facecolor=COLORS["orange"], edgecolor="none", alpha=0.48, label="Bias (Released)"),
        ],
        ncol=2,
    )

    add_panel_header(ax_e, "e", handles=[])
    add_panel_header(ax_f, "f", handles=[])
    add_panel_header(ax_g, "g", handles=[])

    return fig

def make_figure(data: dict[str, Any], metrics: dict[str, Any], output_png: Path, output_pdf: Path, dpi: int) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig = build_figure(data, metrics)
    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)

    fig = build_figure(data, metrics)
    fig.savefig(output_pdf, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data = load_data(args.results_root)
    metrics = build_metrics(data)
    run_sanity_checks(data, metrics)
    make_figure(data, metrics, args.output_png, args.output_pdf, args.dpi)
    print(f"Saved PNG: {args.output_png}")
    print(f"Saved PDF: {args.output_pdf}")
    print(
        "Key checks: "
        + f"L5 totals={metrics['l5_totals']}, "
        + f"GDIS pair={metrics['l6_pair_match_rate']:.4f}, "
        + f"GDIS hit={metrics['l6_geometry_hit_rate']:.4f}, "
        + f"IMERG event-total r={metrics['l6_event_total_corr']:.3f}"
    )


if __name__ == "__main__":
    main()
