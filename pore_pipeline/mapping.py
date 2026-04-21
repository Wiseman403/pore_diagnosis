"""
Clinical scale mapping (Flament 10-point, IBSA 5-point).

Provides two modes:

    (1) ANCHOR MODE — when the published Flament and/or IBSA anchor
        images are available as files, run the full pipeline on each
        anchor once, record the pipeline feature vector, and fit a
        monotone piecewise-linear mapping from features → published
        grade. This is the "calibration from published scale anchors"
        path described in §6 of the plan.

    (2) HEURISTIC MODE — when no anchor images are available, fall back
        to the heuristic mapping in metrology.py. The JSON output flags
        `calibration_basis = "heuristic_no_anchors"` in this mode.

The anchor images themselves are NOT distributed with this code
(licensing). The user is expected to provide them via `AnchorSet`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Anchor:
    """One anchor: an image path + its published grade on a specific scale."""
    image_path: str
    published_grade: float
    scale_name: str                            # "flament_10" or "ibsa_pore_5"


@dataclass
class AnchorSet:
    flament_anchors: List[Anchor] = field(default_factory=list)
    ibsa_anchors: List[Anchor] = field(default_factory=list)

    def has_flament(self) -> bool:
        return len(self.flament_anchors) >= 2

    def has_ibsa(self) -> bool:
        return len(self.ibsa_anchors) >= 2


@dataclass
class AnchorCalibration:
    """Fitted monotone piecewise-linear calibration from a 1-D feature to
    a published grade."""
    feature_name: str
    feature_values: np.ndarray
    grade_values: np.ndarray
    scale_name: str
    n_anchors: int

    def map_grade(self, feature: float) -> float:
        """Interpolate, clipping outside the anchor range."""
        xs = self.feature_values
        ys = self.grade_values
        if feature <= xs[0]:
            return float(ys[0])
        if feature >= xs[-1]:
            return float(ys[-1])
        return float(np.interp(feature, xs, ys))


def fit_monotone_piecewise_linear(
    feature_values: List[float],
    grade_values: List[float],
    feature_name: str,
    scale_name: str,
) -> AnchorCalibration:
    """Fit a monotone piecewise-linear function through the anchor pairs.

    Enforces monotonicity by isotonic regression on (feature, grade) pairs
    after sorting by feature. This is the simplest defensible calibration
    given only a handful of anchor points.
    """
    pairs = sorted(zip(feature_values, grade_values), key=lambda p: p[0])
    xs = np.array([p[0] for p in pairs], dtype=np.float64)
    ys = np.array([p[1] for p in pairs], dtype=np.float64)

    # Isotonic enforcement (pool adjacent violators, simple implementation).
    y_iso = ys.copy()
    changed = True
    while changed:
        changed = False
        for i in range(len(y_iso) - 1):
            if y_iso[i] > y_iso[i + 1]:
                avg = 0.5 * (y_iso[i] + y_iso[i + 1])
                y_iso[i] = avg
                y_iso[i + 1] = avg
                changed = True

    return AnchorCalibration(
        feature_name=feature_name,
        feature_values=xs,
        grade_values=y_iso,
        scale_name=scale_name,
        n_anchors=len(xs),
    )


def calibrate_flament_from_anchors(
    anchor_feature_extractor,
    anchors: List[Anchor],
) -> Optional[AnchorCalibration]:
    """Build the Flament calibration from anchor images.

    anchor_feature_extractor: callable (image_path) -> dict of zone/face-level
        features. We use the 'face_mean_visibility' scalar for mapping by
        default.
    """
    if len(anchors) < 2:
        return None
    features = []
    grades = []
    for a in anchors:
        try:
            f = anchor_feature_extractor(a.image_path)
            features.append(f.get("face_mean_visibility", 0.0))
            grades.append(a.published_grade)
        except Exception:
            continue
    if len(features) < 2:
        return None
    return fit_monotone_piecewise_linear(
        features, grades, "face_mean_visibility", "flament_10"
    )


def calibrate_ibsa_from_anchors(
    anchor_feature_extractor,
    anchors: List[Anchor],
) -> Optional[AnchorCalibration]:
    """Build IBSA 5-point pore-axis calibration from anchors."""
    if len(anchors) < 2:
        return None
    features = []
    grades = []
    for a in anchors:
        try:
            f = anchor_feature_extractor(a.image_path)
            features.append(f.get("whole_face_composite", 0.0))
            grades.append(a.published_grade)
        except Exception:
            continue
    if len(features) < 2:
        return None
    return fit_monotone_piecewise_linear(
        features, grades, "whole_face_composite", "ibsa_pore_5"
    )
