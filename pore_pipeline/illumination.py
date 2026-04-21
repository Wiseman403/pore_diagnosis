"""
Stage 4: Illumination normalization + ITA° estimation + Fitzpatrick bucket.

Pipeline inside this stage:
  1. Shades-of-Gray color constancy on the full image (already applied
     upstream in Stage 1 if desired; idempotent to re-apply).
  2. Convert to L*a*b*.
  3. Compute ITA° over skin-masked pixels → Fitzpatrick bucket.
  4. Homomorphic filtering on L* (masked) to flatten multiplicative shading.
  5. Phototype-adaptive CLAHE on the normalized L* (skin-only).

All operations are classical / mathematical. No learned components.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from .config import ita_to_fitzpatrick
from .utils import bgr_to_lab, compute_ita_deg, shades_of_gray


@dataclass
class IlluminationResult:
    """Stage 4 output."""
    bgr_color_constant: np.ndarray      # shades-of-gray applied
    lab: np.ndarray                     # HxWx3 float32 L*a*b*
    L_normalized: np.ndarray            # HxW float32 L* after homomorphic + CLAHE
    ita_deg_mean: float
    ita_deg_std: float
    fitzpatrick: str
    chromatic_imbalance_flag: bool


# =============================================================================
# Homomorphic filter (multiplicative-shading removal)
# =============================================================================
def _homomorphic_filter(L: np.ndarray,
                        mask: np.ndarray,
                        low_cut: float = 0.5,
                        high_gain: float = 1.5,
                        sigma_px: float = 30.0) -> np.ndarray:
    """Apply homomorphic filter to L* channel over a mask.

    I(x) = reflectance(x) * illumination(x)
        log I = log R + log L
    Low-pass log I → illumination estimate; subtract to get reflectance.

    We implement the filter in the log domain with a Gaussian low-pass.
    Pixels outside the mask are filled with the mask-mean before filtering
    (avoids boundary artefacts), then re-masked.
    """
    L_f = L.astype(np.float32)
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return L_f

    # Fill outside-mask with mean for stable low-pass.
    mean_L = float(L_f[mask_bool].mean())
    filled = L_f.copy()
    filled[~mask_bool] = mean_L

    # Add 1 to avoid log(0); L in [0, 100] so safe.
    log_I = np.log1p(filled)
    illumination_est = cv2.GaussianBlur(log_I, (0, 0), sigmaX=sigma_px, sigmaY=sigma_px)
    reflectance = log_I - low_cut * illumination_est
    # Exponentiate back.
    out = np.expm1(reflectance * high_gain)
    # Rescale to original L* range on the mask.
    if mask_bool.any():
        m_old = L_f[mask_bool].mean()
        m_new = out[mask_bool].mean()
        s_old = L_f[mask_bool].std() + 1e-6
        s_new = out[mask_bool].std() + 1e-6
        out = (out - m_new) * (s_old / s_new) + m_old
    out = np.clip(out, 0.0, 100.0).astype(np.float32)
    # Preserve original values outside the mask.
    out[~mask_bool] = L_f[~mask_bool]
    return out


# =============================================================================
# Phototype-adaptive CLAHE
# =============================================================================
def _clahe_on_mask(L: np.ndarray,
                   mask: np.ndarray,
                   clip_limit: float,
                   tile_grid_size: int) -> np.ndarray:
    """CLAHE on L* channel, applied to the whole image but blended by mask.

    Note: CLAHE needs uint8. We scale [0,100] → [0,255], CLAHE, then back.
    """
    L_u8 = np.clip(L * (255.0 / 100.0), 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    L_eq = clahe.apply(L_u8)
    L_eq_f = L_eq.astype(np.float32) * (100.0 / 255.0)
    mask_bool = mask.astype(bool)
    out = L.copy()
    out[mask_bool] = L_eq_f[mask_bool]
    return out


# =============================================================================
# Chromatic imbalance detector
# =============================================================================
def _chromatic_imbalance(lab: np.ndarray, mask: np.ndarray) -> bool:
    """Flag images with extreme chromatic cast (e.g., warm tungsten, colored gel).

    Check: if mean |a*| or mean |b*| on skin is extreme, our ITA° estimate
    becomes unreliable and phototype-conditioned thresholds should fall back.
    """
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return True
    a_mean = float(lab[..., 1][mask_bool].mean())
    b_mean = float(lab[..., 2][mask_bool].mean())
    # Empirical thresholds: healthy skin typically has a ∈ [0, 20], b ∈ [5, 30].
    # Extreme values suggest colored illuminant.
    if a_mean < -5.0 or a_mean > 35.0:
        return True
    if b_mean < -10.0 or b_mean > 50.0:
        return True
    return False


# =============================================================================
# Main Stage 4 entry point
# =============================================================================
def normalize_illumination(
    bgr: np.ndarray,
    skin_mask: np.ndarray,
    clahe_clip_limit: float = 1.5,
    clahe_tile_px: int = 32,
) -> IlluminationResult:
    """Run full Stage 4 pipeline.

    Args:
        bgr: HxWx3 uint8 BGR image.
        skin_mask: HxW uint8, nonzero on skin.
        clahe_clip_limit: CLAHE clip limit. Lower = softer enhancement.
            Recommend 1.0-2.0 for clinical-grade output (we do not want to
            synthesize contrast where none exists).
        clahe_tile_px: CLAHE tile size in pixels. Should be ~15× expected
            pore diameter; in practice 24-48 works well across capture ranges.

    Returns:
        IlluminationResult with all intermediate products.
    """
    # 1. Color constancy.
    bgr_cc = shades_of_gray(bgr, p=6.0)

    # 2. Lab conversion (ranges: L [0,100], a,b signed).
    lab = bgr_to_lab(bgr_cc)

    # 3. ITA° + Fitzpatrick.
    ita_mean, ita_std = compute_ita_deg(lab, skin_mask)
    fitzpatrick = ita_to_fitzpatrick(ita_mean) if not np.isnan(ita_mean) else "III"
    chrom_flag = _chromatic_imbalance(lab, skin_mask)

    # 4. Homomorphic filter on L*.
    L = lab[..., 0]
    L_homo = _homomorphic_filter(L, skin_mask, low_cut=0.5, high_gain=1.0, sigma_px=30.0)

    # 5. CLAHE on L* inside skin mask.
    L_final = _clahe_on_mask(L_homo, skin_mask, clahe_clip_limit, max(4, clahe_tile_px // 8))

    return IlluminationResult(
        bgr_color_constant=bgr_cc,
        lab=lab,
        L_normalized=L_final,
        ita_deg_mean=ita_mean,
        ita_deg_std=ita_std,
        fitzpatrick=fitzpatrick,
        chromatic_imbalance_flag=chrom_flag,
    )
