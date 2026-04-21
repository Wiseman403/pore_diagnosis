"""
Stage 1: Capture quality gate.

Runs BEFORE foundation-model loading so a bad image doesn't waste GPU time.

Checks, in order:
    1. Face presence
    2. Head pose (yaw / pitch / roll gates)
    3. Blur on skin region (variance of Laplacian)
    4. Exposure (clipped-pixel fraction, L* IQR)
    5. Capture distance (pixels-per-mm from IOD)
    6. Texture-energy floor (advisory beauty-filter detector, classical)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import (
    BLUR_VAR_LAPLACIAN_MIN,
    CAPTURE_DISTANCE_MAX_MM,
    CAPTURE_DISTANCE_MIN_MM,
    EXPOSURE_CLIPPED_PIXEL_FRACTION_MAX,
    EXPOSURE_L_CHANNEL_IQR_MIN,
    PHOTOTYPE_PARAMS,
    POSE_MAX_PITCH_DEG,
    POSE_MAX_ROLL_DEG,
    POSE_MAX_YAW_DEG,
    PPMM_MAX,
    PPMM_MIN,
)
from .face import FaceLandmarks
from .utils import bgr_to_lab


@dataclass
class QualityReport:
    passed: bool
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    blur_var_laplacian: float = 0.0
    clipped_fraction: float = 0.0
    l_iqr: float = 0.0
    pose_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ppmm: float = 0.0
    capture_distance_mm: float = 0.0
    texture_energy_rms: float = 0.0
    fitzpatrick_prior: str = "III"


def _wavelet_detail_rms(gray: np.ndarray, mask: np.ndarray) -> float:
    """Classical high-frequency-energy measure.

    Uses a simple 2-level Haar-like decomposition: high-pass detail band
    is `gray - blur(gray)`. RMS on skin pixels is a robust proxy for
    "how much pore-scale texture is present." Values far below phototype
    norms suggest beauty smoothing.

    This is a *classical* diagnostic — not a learned filter detector.
    """
    g = gray.astype(np.float32)
    blur = cv2.GaussianBlur(g, (0, 0), sigmaX=1.0, sigmaY=1.0)
    detail = g - blur
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return 0.0
    return float(np.sqrt(np.mean(detail[mask_bool] ** 2)))


def capture_quality_gate(
    bgr: np.ndarray,
    face: Optional[FaceLandmarks],
    skin_mask: Optional[np.ndarray] = None,
    fitzpatrick_prior: str = "III",
) -> QualityReport:
    """Run the full capture-quality gate.

    skin_mask is optional; if None, a quick face-hull mask is computed from
    landmarks. fitzpatrick_prior is used only for the texture-energy floor;
    we don't know phototype yet at Stage 1 so default is III.
    """
    reasons: List[str] = []
    warnings: List[str] = []

    if face is None:
        reasons.append("no_face_detected")
        return QualityReport(passed=False, reasons=reasons)

    # --- Pose ---
    yaw, pitch, roll = face.head_pose_deg
    if abs(yaw) > POSE_MAX_YAW_DEG:
        reasons.append(f"yaw_out_of_range:{yaw:.1f}deg")
    if abs(pitch) > POSE_MAX_PITCH_DEG:
        reasons.append(f"pitch_out_of_range:{pitch:.1f}deg")
    if abs(roll) > POSE_MAX_ROLL_DEG:
        reasons.append(f"roll_out_of_range:{roll:.1f}deg")

    # --- Skin-region mask for remaining checks ---
    if skin_mask is None:
        # Quick face-hull approximation.
        from .parsing import FACE_OVAL_IDX
        h, w = bgr.shape[:2]
        pts = face.landmarks_px[:, :2][FACE_OVAL_IDX].astype(np.int32)
        hull = cv2.convexHull(pts)
        skin_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(skin_mask, hull, 1)

    # --- Blur ---
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    mask_bool = skin_mask.astype(bool)
    if mask_bool.any():
        blur_var = float(lap[mask_bool].var())
    else:
        blur_var = 0.0
    if blur_var < BLUR_VAR_LAPLACIAN_MIN:
        reasons.append(f"blurry:var_laplacian={blur_var:.1f}<{BLUR_VAR_LAPLACIAN_MIN}")

    # --- Exposure ---
    lab = bgr_to_lab(bgr)
    L = lab[..., 0]
    if mask_bool.any():
        L_skin = L[mask_bool]
        clipped_high = (L_skin >= 99.0).mean()
        clipped_low = (L_skin <= 1.0).mean()
        clipped_frac = float(clipped_high + clipped_low)
        q1, q3 = np.percentile(L_skin, [25, 75])
        l_iqr = float(q3 - q1)
    else:
        clipped_frac = 1.0
        l_iqr = 0.0
    if clipped_frac > EXPOSURE_CLIPPED_PIXEL_FRACTION_MAX:
        reasons.append(f"overexposed_or_underexposed:clipped_frac={clipped_frac:.2%}")
    if l_iqr < EXPOSURE_L_CHANNEL_IQR_MIN:
        reasons.append(f"flat_exposure:L_IQR={l_iqr:.1f}<{EXPOSURE_L_CHANNEL_IQR_MIN}")

    # --- Capture distance (from ppmm) ---
    ppmm = face.scale.ppmm
    # Estimate distance: at a given ppmm, the physical mm covered per pixel
    # determines capture distance (very loosely — varies by sensor/focal length).
    # We use ppmm thresholds directly rather than back-solving distance.
    if ppmm < PPMM_MIN:
        reasons.append(f"too_far:ppmm={ppmm:.2f}<{PPMM_MIN}")
    elif ppmm > PPMM_MAX:
        reasons.append(f"too_close:ppmm={ppmm:.2f}>{PPMM_MAX}")
    # Heuristic distance estimate (35mm-equiv focal length ~28mm typical
    # front camera, sensor width ~5.6mm). Rough: distance_mm ≈ 63_mm / (ppmm * sensor_mm_per_px).
    # We just approximate:
    dist_mm = 63.0 / max(ppmm, 1e-6) * 20.0   # rough coefficient — for display only
    dist_mm = float(np.clip(dist_mm, 50.0, 1500.0))

    # --- Texture-energy floor (advisory) ---
    texture_rms = _wavelet_detail_rms(gray, skin_mask)
    phot = PHOTOTYPE_PARAMS.get(fitzpatrick_prior, PHOTOTYPE_PARAMS["III"])
    if texture_rms < phot.texture_energy_rms_min:
        warnings.append(
            f"low_texture_energy={texture_rms:.2f}<{phot.texture_energy_rms_min} "
            f"(possible beauty filter or ISP smoothing; pore counts may be underestimated)"
        )

    passed = len(reasons) == 0
    return QualityReport(
        passed=passed, reasons=reasons, warnings=warnings,
        blur_var_laplacian=blur_var, clipped_fraction=clipped_frac, l_iqr=l_iqr,
        pose_deg=(float(yaw), float(pitch), float(roll)),
        ppmm=float(ppmm), capture_distance_mm=dist_mm,
        texture_energy_rms=texture_rms, fitzpatrick_prior=fitzpatrick_prior,
    )
