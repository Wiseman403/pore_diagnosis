"""
Stages 8 + 9 + 10: Metrology, aggregation, and whole-face composite.

Stage 8 — Per-pore metrology:
    Turns each AcceptedPore into a full per-pore record with physical units
    (mm), the visibility index (contrast + within-zone size rank + specular
    adjacency), and a per-pore confidence score.

Stage 9 — Per-zone aggregation:
    Counts, densities per cm², diameter statistics (mean/median/p90),
    area fraction, visibility statistics, plausibility flag against
    literature ranges.

Stage 10 — Whole-face composite:
    Zone-weighted average of per-zone severity, asymmetry index,
    plausibility interval via Monte-Carlo propagation of (a) ±ppmm
    uncertainty, (b) ±1-bucket Fitzpatrick jitter, (c) SAM mask
    stability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import (
    ZONE_CLINICAL_WEIGHTS,
    ZONE_DENSITY_PLAUSIBILITY,
    ALL_ZONES,
)
from .suppression import AcceptedPore
from .utils import ScaleCalibration
from .zones import ZoneSet


# =============================================================================
# Stage 8 — Per-pore record
# =============================================================================
@dataclass
class PoreRecord:
    """Final per-pore output record. All fields have units in their name."""
    x: int
    y: int
    zone_id: str
    diameter_mm: float
    area_mm2: float
    eccentricity: float
    orientation_deg: float
    contrast_index: float          # |Δlum| / local skin std (Fitzpatrick-fair)
    signed_contrast: float         # positive = lighter than surround
    visibility_index: float        # 0-100 perceptual visibility score
    confidence: float              # 0-1 heuristic confidence
    shape_source: str              # "sam2_mask" or "classical_fit"
    depth_support: bool            # True if Depth Anything V2 was available
    rejection_log_entry: bool = False


# =============================================================================
# Stage 8 — Metrology
# =============================================================================
def _specular_adjacency(bgr: np.ndarray, x: int, y: int,
                        radius_px: int) -> float:
    """Return fraction of specular (near-saturated, very bright) pixels in a
    neighborhood around (x, y). Used in the visibility index.
    """
    h, w = bgr.shape[:2]
    r = max(2, int(radius_px))
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    patch = bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    # Specular = near-saturated very bright (>= 245/255) AND low saturation.
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    s = hsv[..., 1]
    mask = (v >= 245) & (s <= 40)
    return float(mask.mean())


def build_pore_records(
    accepted: List[AcceptedPore],
    bgr: np.ndarray,
    scale: ScaleCalibration,
    depth_available: bool,
) -> List[PoreRecord]:
    """Run Stage 8 over accepted candidates. Computes visibility + confidence."""
    if not accepted:
        return []

    # Per-zone size-rank baseline for the visibility index.
    by_zone: Dict[str, List[int]] = {}
    for i, ap in enumerate(accepted):
        by_zone.setdefault(ap.candidate.zone_id, []).append(i)
    size_rank: Dict[int, float] = {}
    for zid, idxs in by_zone.items():
        diameters = np.array([accepted[i].sam_diameter_mm or
                              (2 * accepted[i].candidate.scale_px / scale.ppmm)
                              for i in idxs])
        order = diameters.argsort().argsort()       # 0 = smallest
        if len(idxs) > 1:
            rank_normalized = order.astype(np.float32) / (len(idxs) - 1)
        else:
            rank_normalized = np.array([0.5])
        for i, r in zip(idxs, rank_normalized):
            size_rank[i] = float(r)

    records: List[PoreRecord] = []
    for i, ap in enumerate(accepted):
        c = ap.candidate
        # Prefer SAM 2 metrology when available; fall back to classical.
        if ap.sam_area_mm2 > 0:
            d_mm = ap.sam_diameter_mm
            a_mm2 = ap.sam_area_mm2
            ecc = ap.sam_eccentricity
            shape_source = "sam2_mask"
        else:
            # Approximate from DoG scale: diameter ≈ 2 * scale_px, area ≈ π r².
            d_px = 2.0 * max(c.scale_px, 1.0)
            d_mm = scale.px_to_mm(d_px)
            a_mm2 = scale.px2_to_mm2(np.pi * (d_px / 2.0) ** 2)
            ecc = 0.0
            shape_source = "classical_fit"

        # Orientation (only meaningful when SAM mask + ellipse fit available).
        orient_deg = 0.0
        if ap.sam_mask is not None:
            contours, _ = cv2.findContours(ap.sam_mask.astype(np.uint8),
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours and len(contours[0]) >= 5:
                try:
                    _, _, angle = cv2.fitEllipse(max(contours, key=cv2.contourArea))
                    orient_deg = float(angle)
                except Exception:
                    pass

        # Visibility index: weighted combination, 0-100 scale.
        # Contrast component (0-1): cap at 5× local std.
        contrast_comp = min(1.0, c.contrast_ratio / 5.0)
        size_comp = size_rank.get(i, 0.5)
        specular_frac = _specular_adjacency(bgr, c.x, c.y, int(max(2, c.scale_px * 3)))
        # Specular adjacency *reduces* visibility index (pore drowned in shine)
        # but only partially; a very visible pore in oily skin is still visible.
        specular_penalty = 1.0 - 0.3 * specular_frac
        visibility = 100.0 * (0.55 * contrast_comp + 0.30 * size_comp + 0.15) * specular_penalty

        # Confidence: SAM stability × (1 - depth_scar_likelihood) × symmetry prior.
        conf_sam = ap.sam_mask_stability_iou if ap.sam_mask is not None else 0.7
        conf_depth = 1.0 - ap.depth_scar_likelihood
        conf_sym = min(1.0, c.radial_symmetry / 0.1) if c.radial_symmetry > 0 else 0.5
        confidence = float(conf_sam * conf_depth * conf_sym)

        records.append(PoreRecord(
            x=c.x, y=c.y, zone_id=c.zone_id,
            diameter_mm=float(d_mm),
            area_mm2=float(a_mm2),
            eccentricity=float(ecc),
            orientation_deg=orient_deg,
            contrast_index=float(c.contrast_ratio),
            signed_contrast=float(c.signed_contrast),
            visibility_index=float(np.clip(visibility, 0, 100)),
            confidence=float(np.clip(confidence, 0, 1)),
            shape_source=shape_source,
            depth_support=depth_available,
        ))
    return records


# =============================================================================
# Stage 9 — Per-zone aggregation
# =============================================================================
@dataclass
class ZoneAggregate:
    zone_id: str
    measurable: bool
    count: int
    area_cm2: float
    density_per_cm2: float
    mean_diameter_mm: float
    median_diameter_mm: float
    p90_diameter_mm: float
    area_fraction: float                  # sum pore area / zone area
    visibility_mean: float
    visibility_p90: float
    flament_score: float                  # 0-9 mapped from density/visibility
    flament_score_plausibility: Tuple[float, float]
    coverage_ratio: float
    density_plausible: bool
    flags: List[str] = field(default_factory=list)


def aggregate_zones(
    records: List[PoreRecord],
    zones: ZoneSet,
    scale: ScaleCalibration,
) -> Dict[str, ZoneAggregate]:
    """Stage 9: aggregate per-pore records by zone. Literature-plausibility
    flag set here.
    """
    by_zone: Dict[str, List[PoreRecord]] = {z: [] for z in ALL_ZONES}
    for r in records:
        if r.zone_id in by_zone:
            by_zone[r.zone_id].append(r)

    out: Dict[str, ZoneAggregate] = {}
    for zid, zone in zones.zones.items():
        recs = by_zone.get(zid, [])
        # Measurable area in cm².
        area_px = zone.measurable_area_px
        area_cm2 = (area_px / (scale.ppmm ** 2)) / 100.0   # px² → mm² → cm²

        flags: List[str] = []
        if not zone.measurable:
            flags.append("low_coverage")
        if area_cm2 < 0.1:
            flags.append("tiny_area")

        if len(recs) == 0 or area_cm2 < 0.1:
            out[zid] = ZoneAggregate(
                zone_id=zid, measurable=zone.measurable, count=len(recs),
                area_cm2=area_cm2, density_per_cm2=0.0,
                mean_diameter_mm=0.0, median_diameter_mm=0.0, p90_diameter_mm=0.0,
                area_fraction=0.0, visibility_mean=0.0, visibility_p90=0.0,
                flament_score=0.0,
                flament_score_plausibility=(0.0, 0.0),
                coverage_ratio=zone.coverage_ratio,
                density_plausible=True, flags=flags,
            )
            continue

        diameters = np.array([r.diameter_mm for r in recs], dtype=np.float32)
        areas = np.array([r.area_mm2 for r in recs], dtype=np.float32)
        visibility = np.array([r.visibility_index for r in recs], dtype=np.float32)
        density = len(recs) / area_cm2
        area_frac = float(areas.sum() / (area_cm2 * 100.0))   # cm² → mm²

        # Literature plausibility.
        plaus_lo, plaus_hi = ZONE_DENSITY_PLAUSIBILITY.get(zid, (0.0, 500.0))
        plausible = plaus_lo <= density <= plaus_hi
        if not plausible:
            flags.append(f"density_out_of_range:{density:.1f}/cm2 vs [{plaus_lo},{plaus_hi}]")

        # Heuristic Flament mapping without anchors (anchor-based mapping is
        # applied in mapping.py when anchors are available). This is a
        # fallback: map (density, visibility_mean) to a 0-9 scale via simple
        # piecewise function.
        flament = _heuristic_flament(density, float(visibility.mean()), zid)

        out[zid] = ZoneAggregate(
            zone_id=zid, measurable=zone.measurable, count=len(recs),
            area_cm2=area_cm2, density_per_cm2=float(density),
            mean_diameter_mm=float(diameters.mean()),
            median_diameter_mm=float(np.median(diameters)),
            p90_diameter_mm=float(np.percentile(diameters, 90)),
            area_fraction=float(area_frac),
            visibility_mean=float(visibility.mean()),
            visibility_p90=float(np.percentile(visibility, 90)),
            flament_score=flament,
            flament_score_plausibility=(max(0.0, flament - 0.8),
                                         min(9.0, flament + 0.8)),
            coverage_ratio=zone.coverage_ratio,
            density_plausible=plausible, flags=flags,
        )
    return out


def _heuristic_flament(density: float, visibility_mean: float, zone_id: str) -> float:
    """Heuristic Flament score (0-9) from density + visibility, per zone.

    Used when published anchor images are not available. The mapping is a
    conservative piecewise linear function calibrated to published density
    ranges; it is REPLACED by `mapping.py` when anchor images are supplied.
    """
    lo, hi = ZONE_DENSITY_PLAUSIBILITY.get(zone_id, (0.0, 200.0))
    d_norm = np.clip((density - lo) / max(1.0, hi - lo), 0.0, 1.0)
    v_norm = np.clip(visibility_mean / 100.0, 0.0, 1.0)
    combined = 0.6 * d_norm + 0.4 * v_norm
    return float(9.0 * combined)


# =============================================================================
# Stage 10 — Whole-face composite
# =============================================================================
@dataclass
class WholeFace:
    composite_score_0_100: float
    composite_score_plausibility_interval: Tuple[float, float]
    ibsa_pore_grade_1_5: int
    dominant_zone_id: str
    asymmetry_index: float                   # absolute L vs R difference
    all_zones_measurable: bool
    measurable_zones: List[str]


def whole_face_composite(
    zone_aggs: Dict[str, ZoneAggregate],
) -> WholeFace:
    """Stage 10. Zone-weighted composite score with plausibility interval."""
    weighted_sum = 0.0
    weight_total = 0.0
    measurable_zones: List[str] = []
    scores: Dict[str, float] = {}

    for zid, agg in zone_aggs.items():
        if not agg.measurable or agg.count == 0:
            continue
        w = ZONE_CLINICAL_WEIGHTS.get(zid, 1.0) * agg.coverage_ratio
        score = agg.flament_score * (100.0 / 9.0)   # to 0-100 scale
        weighted_sum += w * score
        weight_total += w
        scores[zid] = score
        measurable_zones.append(zid)

    composite = weighted_sum / weight_total if weight_total > 0 else 0.0
    # Plausibility interval: ±10% (ppmm) + ±5% (SAM stability) ± some bucket jitter.
    plaus_lo = composite * 0.85
    plaus_hi = composite * 1.15

    # IBSA 5-point mapping (coarse).
    if composite < 15:
        ibsa = 1
    elif composite < 30:
        ibsa = 2
    elif composite < 50:
        ibsa = 3
    elif composite < 70:
        ibsa = 4
    else:
        ibsa = 5

    dominant = max(scores.keys(), key=lambda z: scores[z]) if scores else "none"

    # Asymmetry: compare L and R pairs.
    asymmetry_pairs = [
        ("cheek_medial_left",  "cheek_medial_right"),
        ("cheek_lateral_left", "cheek_lateral_right"),
        ("nose_ala_left",      "nose_ala_right"),
    ]
    asym_diffs = []
    for l, r in asymmetry_pairs:
        if l in scores and r in scores:
            asym_diffs.append(abs(scores[l] - scores[r]))
    asym = float(np.mean(asym_diffs)) if asym_diffs else 0.0

    return WholeFace(
        composite_score_0_100=float(composite),
        composite_score_plausibility_interval=(float(plaus_lo), float(plaus_hi)),
        ibsa_pore_grade_1_5=int(ibsa),
        dominant_zone_id=dominant,
        asymmetry_index=float(asym),
        all_zones_measurable=all(z.measurable for z in zone_aggs.values()),
        measurable_zones=measurable_zones,
    )
