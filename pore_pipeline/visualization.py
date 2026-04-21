"""
Visualization: expert diagnostic dashboard rendering.

Produces a single composite figure with:
    Panel A: original image.
    Panel B: skin-isolated view (non-skin at 30% opacity grey).
    Panel C: per-zone visibility heatmap (viridis colormap).
    Panel D: per-zone diagnostic table.
    Panel F: composite score + IBSA grade + plausibility interval.

Colormap discipline: viridis for ordinal quantitative overlays (colorblind-
safe, perceptually uniform). No red/green traffic-light semantics.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np

from .config import ALL_ZONES
from .metrology import PoreRecord, WholeFace, ZoneAggregate
from .parsing import SkinMaskResult
from .zones import ZoneSet


def render_dashboard(
    bgr: np.ndarray,
    skin: SkinMaskResult,
    zones: ZoneSet,
    records: List[PoreRecord],
    zone_aggs: Dict[str, ZoneAggregate],
    whole_face: WholeFace,
    fitzpatrick: str,
    warnings: Optional[List[str]] = None,
    figsize=(16, 11),
):
    """Render the expert dashboard as a matplotlib Figure."""
    warnings = warnings or []
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[2.0, 2.0, 1.0])

    # --- Panel A: original ---
    axA = fig.add_subplot(gs[0, 0:2])
    axA.imshow(rgb)
    axA.set_title("A. Original image", fontsize=11)
    axA.axis("off")

    # --- Panel B: skin-isolated view ---
    axB = fig.add_subplot(gs[0, 2:4])
    skin_bool = skin.skin_mask.astype(bool)
    overlay = rgb.copy()
    # Dim non-skin to 30% brightness greyscale.
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    non_skin = np.dstack([gray, gray, gray])
    dimmed = (non_skin.astype(np.float32) * 0.3).astype(np.uint8)
    overlay[~skin_bool] = dimmed[~skin_bool]
    axB.imshow(overlay)
    axB.set_title(f"B. Skin-isolated view (parser: {skin.parser_backend})", fontsize=11)
    axB.axis("off")

    # --- Panel C: per-zone visibility heatmap ---
    axC = fig.add_subplot(gs[1, 0:2])
    heatmap = np.zeros((h, w), dtype=np.float32)
    for zid, agg in zone_aggs.items():
        zone = zones.zones.get(zid)
        if zone is None or not zone.measurable:
            continue
        heatmap[zone.measurable_mask.astype(bool)] = agg.visibility_mean
    vmin, vmax = 0.0, 100.0
    axC.imshow(rgb)
    # Overlay the heatmap with transparency.
    heatmap_masked = np.ma.masked_where(heatmap == 0, heatmap)
    axC.imshow(heatmap_masked, cmap="viridis", alpha=0.55, vmin=vmin, vmax=vmax)
    # Overlay per-pore points, small, color-coded by visibility.
    if records:
        xs = [r.x for r in records]
        ys = [r.y for r in records]
        vs = [r.visibility_index for r in records]
        axC.scatter(xs, ys, c=vs, cmap="viridis", vmin=vmin, vmax=vmax,
                    s=6, edgecolors="white", linewidths=0.2)
    axC.set_title("C. Per-zone visibility heatmap (viridis, 0-100)", fontsize=11)
    axC.axis("off")
    cax = axC.inset_axes([0.02, 0.05, 0.02, 0.3])
    fig.colorbar(ScalarMappable(norm=Normalize(vmin, vmax), cmap="viridis"),
                 cax=cax, orientation="vertical")

    # --- Panel D: diagnostic table ---
    axD = fig.add_subplot(gs[1, 2:4])
    axD.axis("off")
    axD.set_title("D. Per-zone diagnostics", fontsize=11)
    table_rows = []
    for zid in ALL_ZONES:
        agg = zone_aggs.get(zid)
        if agg is None:
            continue
        if not agg.measurable:
            table_rows.append([zid, "—", "—", "—", "—", "—", "not measurable"])
            continue
        table_rows.append([
            zid,
            f"{agg.count}",
            f"{agg.density_per_cm2:.1f}",
            f"{agg.mean_diameter_mm*1000:.0f}µm",
            f"{agg.visibility_mean:.0f}",
            f"{agg.flament_score:.1f}",
            f"{agg.coverage_ratio*100:.0f}%" + (" ⚠" if not agg.density_plausible else ""),
        ])
    col_labels = ["Zone", "N", "Density\n/cm²", "Mean Ø", "Vis", "Flament\n0-9", "Coverage"]
    table = axD.table(cellText=table_rows, colLabels=col_labels,
                      cellLoc="center", colLoc="center", loc="upper center",
                      bbox=[0.0, 0.0, 1.0, 0.95])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)

    # --- Panel F: composite + flags ---
    axF = fig.add_subplot(gs[2, 0:4])
    axF.axis("off")
    lines = [
        f"Fitzpatrick phototype: {fitzpatrick}",
        f"Composite pore score: {whole_face.composite_score_0_100:.1f} / 100 "
        f"(plausibility {whole_face.composite_score_plausibility_interval[0]:.1f} – "
        f"{whole_face.composite_score_plausibility_interval[1]:.1f})",
        f"IBSA pore grade: {whole_face.ibsa_pore_grade_1_5} / 5",
        f"Dominant zone: {whole_face.dominant_zone_id}",
        f"L–R asymmetry index: {whole_face.asymmetry_index:.1f}",
        f"Measurable zones: {len(whole_face.measurable_zones)}/{len(ALL_ZONES)}",
    ]
    if warnings:
        lines.append("Warnings: " + " | ".join(warnings[:3]))
    text = "\n".join(lines)
    axF.text(0.01, 0.98, text, va="top", ha="left", fontsize=10, family="monospace",
             transform=axF.transAxes)
    axF.set_title("F. Composite summary", fontsize=11, loc="left")

    return fig


def render_per_pore_overlay(bgr: np.ndarray,
                             records: List[PoreRecord],
                             alpha: float = 0.7) -> np.ndarray:
    """Return a BGR image with each detected pore drawn as a thin circle.

    Color encodes visibility_index (viridis). Useful for standalone export.
    """
    out = bgr.copy()
    norm = Normalize(0, 100)
    cmap = plt.get_cmap("viridis")
    for r in records:
        rgba = cmap(norm(r.visibility_index))
        bgr_color = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))
        cv2.circle(out, (r.x, r.y), max(2, int(r.diameter_mm * 30)), bgr_color, 1)
    return out
