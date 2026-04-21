"""
Stage 5: Per-zone ROI extraction.

Produces eleven anatomical zones (forehead, glabella, nose dorsum,
nose tip, nose ala L/R, medial cheek L/R, lateral cheek L/R, chin,
perioral) as convex-hull polygons over fixed MediaPipe landmark indices.

Each zone is intersected with the skin mask from Stage 3 to get the
*measurable* area for that zone. Coverage ratio (measurable / nominal
hull area) is recorded and is a first-class output field — zones with
coverage < ZONE_MIN_COVERAGE_RATIO are flagged as not measurable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np

from .config import (
    ALL_ZONES,
    ZONE_LANDMARKS,
    ZONE_MAP_VERSION,
    ZONE_MIN_COVERAGE_RATIO,
)


@dataclass
class Zone:
    """A single anatomical zone's geometry + masks."""
    zone_id: str
    hull_mask: np.ndarray              # HxW uint8, 1 inside the zone's convex hull
    measurable_mask: np.ndarray        # HxW uint8, hull ∩ skin_mask
    hull_area_px: int
    measurable_area_px: int
    coverage_ratio: float              # measurable / hull
    measurable: bool                   # coverage >= threshold


@dataclass
class ZoneSet:
    """All zones for a face."""
    zones: Dict[str, Zone]
    version: str = ZONE_MAP_VERSION


def build_zones(
    shape_hw: tuple,
    landmarks_px: np.ndarray,
    skin_mask: np.ndarray,
) -> ZoneSet:
    """Construct all 11 zones from landmarks + skin mask."""
    h, w = shape_hw
    out: Dict[str, Zone] = {}
    skin_bool = skin_mask.astype(bool)

    for zone_id, idx_list in ZONE_LANDMARKS.items():
        # Extract landmark points for this zone, clipped to image.
        pts = landmarks_px[idx_list, :2].astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

        hull_mask = np.zeros((h, w), dtype=np.uint8)
        if len(pts) >= 3:
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(hull_mask, hull, 1)

        hull_area = int(hull_mask.sum())
        measurable = (hull_mask.astype(bool) & skin_bool).astype(np.uint8)
        measurable_area = int(measurable.sum())
        coverage = float(measurable_area) / max(1, hull_area)

        out[zone_id] = Zone(
            zone_id=zone_id,
            hull_mask=hull_mask,
            measurable_mask=measurable,
            hull_area_px=hull_area,
            measurable_area_px=measurable_area,
            coverage_ratio=coverage,
            measurable=coverage >= ZONE_MIN_COVERAGE_RATIO,
        )

    return ZoneSet(zones=out)


def zone_of_point(zones: ZoneSet, x: int, y: int) -> str | None:
    """Given an (x, y) pixel, return which zone it falls in (first match)."""
    for zid in ALL_ZONES:
        z = zones.zones.get(zid)
        if z is not None and z.measurable_mask[y, x]:
            return zid
    return None
