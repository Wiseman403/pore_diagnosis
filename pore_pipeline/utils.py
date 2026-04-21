"""
Utility helpers: color-space conversion, geometry, image I/O.

Nothing here is model-dependent; all classical / mathematical.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


# =============================================================================
# Image I/O
# =============================================================================
def load_image_bgr(path: str) -> np.ndarray:
    """Load an image from disk as BGR uint8. Raises on failure."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    return img


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def image_size(img: np.ndarray) -> Tuple[int, int]:
    """Return (width, height) for an H×W×C image."""
    h, w = img.shape[:2]
    return w, h


# =============================================================================
# Color conversion
# =============================================================================
def bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 → Lab float32 in OpenCV convention.

    OpenCV's Lab for 8-bit images uses:
        L* in [0, 255] (true L* * 255/100)
        a*, b* in [0, 255] (true value + 128)
    We normalize to the 'natural' ranges:
        L* in [0, 100], a*, b* in [-128, 127].
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0         # L* → [0, 100]
    lab[..., 1] -= 128.0                  # a* → signed
    lab[..., 2] -= 128.0                  # b* → signed
    return lab


def bgr_to_gray_f32(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR → grayscale float32 in [0, 1]."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return g.astype(np.float32) / 255.0


def lab_l_channel(bgr: np.ndarray) -> np.ndarray:
    """Extract L* channel in [0, 100] as float32."""
    return bgr_to_lab(bgr)[..., 0]


# =============================================================================
# Colour constancy: Shades-of-Gray (Minkowski p=6)
# =============================================================================
def shades_of_gray(bgr: np.ndarray, p: float = 6.0) -> np.ndarray:
    """Shades-of-Gray colour constancy (Finlayson & Trezzi 2004).

    Estimates the scene illuminant as the Minkowski p-norm of each channel
    and rescales to a common grey point. p=6 is the standard.
    """
    img = bgr.astype(np.float32)
    channel_means = np.power(np.mean(np.power(img + 1e-6, p), axis=(0, 1)), 1.0 / p)
    grey = channel_means.mean()
    scale = grey / (channel_means + 1e-6)
    out = img * scale
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


# =============================================================================
# ITA° computation
# =============================================================================
def compute_ita_deg(lab: np.ndarray, skin_mask: np.ndarray) -> Tuple[float, float]:
    """Compute Individual Typology Angle (ITA°) over skin pixels.

    ITA° = arctan((L* − 50) / b*) × 180 / π

    Args:
        lab: HxWx3 float32 Lab (L in [0,100], a,b signed).
        skin_mask: HxW bool/uint8, nonzero for skin pixels.

    Returns:
        (ita_mean_deg, ita_std_deg) over the masked region.
    """
    mask_bool = skin_mask.astype(bool)
    if not mask_bool.any():
        return float("nan"), float("nan")
    L = lab[..., 0][mask_bool]
    b = lab[..., 2][mask_bool]
    # Per-pixel ITA° — note arctan2-like guard against b≈0.
    denom = np.where(np.abs(b) < 1e-3, np.sign(b + 1e-6) * 1e-3, b)
    ita = np.degrees(np.arctan((L - 50.0) / denom))
    return float(np.mean(ita)), float(np.std(ita))


# =============================================================================
# Geometry
# =============================================================================
def polygon_mask(shape_hw: Tuple[int, int], points_xy: np.ndarray) -> np.ndarray:
    """Rasterize a polygon to a binary uint8 mask of shape (H, W).

    Args:
        shape_hw: (H, W).
        points_xy: Nx2 array of polygon vertices in (x, y) pixel coords.
    """
    mask = np.zeros(shape_hw, dtype=np.uint8)
    if len(points_xy) >= 3:
        pts = points_xy.astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def convex_hull_mask(shape_hw: Tuple[int, int], points_xy: np.ndarray) -> np.ndarray:
    """Convex hull of the given points, rasterized to a mask."""
    if len(points_xy) < 3:
        return np.zeros(shape_hw, dtype=np.uint8)
    pts = points_xy.astype(np.int32)
    hull = cv2.convexHull(pts)
    mask = np.zeros(shape_hw, dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask


def dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))
    return cv2.dilate(mask.astype(np.uint8), k)


def erode_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))
    return cv2.erode(mask.astype(np.uint8), k)


# =============================================================================
# Scale calibration
# =============================================================================
@dataclass
class ScaleCalibration:
    """Mapping between pixels and millimeters, with uncertainty."""
    ppmm: float                    # pixels per millimeter
    ppmm_uncertainty_pct: float    # e.g. 10.0 for ±10%
    iod_px: float
    iod_mm_assumed: float

    def px_to_mm(self, px: float) -> float:
        return px / self.ppmm

    def mm_to_px(self, mm: float) -> float:
        return mm * self.ppmm

    def mm2_to_px2(self, mm2: float) -> float:
        return mm2 * (self.ppmm ** 2)

    def px2_to_mm2(self, px2: float) -> float:
        return px2 / (self.ppmm ** 2)


def calibrate_from_iod(iod_px: float,
                       iod_mm_assumed: float = 63.0,
                       iod_uncertainty_pct: float = 10.0) -> ScaleCalibration:
    """Build a ScaleCalibration from a measured IOD in pixels."""
    if iod_px <= 0:
        raise ValueError("iod_px must be positive")
    ppmm = iod_px / iod_mm_assumed
    return ScaleCalibration(
        ppmm=ppmm,
        ppmm_uncertainty_pct=iod_uncertainty_pct,
        iod_px=iod_px,
        iod_mm_assumed=iod_mm_assumed,
    )


# =============================================================================
# Safe numerical helpers
# =============================================================================
def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    if abs(b) < 1e-12:
        return default
    return a / b
