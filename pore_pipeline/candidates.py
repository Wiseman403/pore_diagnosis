"""
Stage 6: Classical pore candidate detection.

Entirely mathematical, deterministic, explainable. Pipeline:

  1. Multi-scale Difference-of-Gaussians (DoG) matched to the image-specific
     pore-pixel window (derived from ppmm).
  2. Grayscale mathematical morphology:
       - h-basin transform: captures local dark minima that are "at least h
         deeper" than their surrounds (principled local-prominence filter).
       - h-dome transform (dual): captures local maxima — used for Fitzpatrick
         V-VI where pore openings can present as local brightenings.
  3. Loy-Zelinsky Fast Radial Symmetry Transform at pore-scale radii —
     votes for circularly-symmetric structures.
  4. 8-orientation steerable-filter bank — hair-coherence score; candidates
     sitting on coherent oriented structures are demoted.
  5. Pore-ness field = weighted combination of the above.
  6. Non-maximum suppression with adaptive radius.

Outputs a list of PoreCandidate structs, each with its full classical
feature vector for auditing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np
from scipy.ndimage import maximum_filter
from skimage.morphology import reconstruction as _grey_reconstruction

from .config import (
    PHOTOTYPE_PARAMS,
    PORE_DIAMETER_MAX_MM,
    PORE_DIAMETER_MIN_MM,
)
from .utils import ScaleCalibration


@dataclass
class PoreCandidate:
    """One classical pore candidate. All fields are mathematical features;
    none are learned."""
    x: int
    y: int
    zone_id: str
    scale_px: float                  # the DoG scale at which it was strongest
    dog_response: float              # signed DoG response at candidate
    h_prominence: float              # h-basin or h-dome prominence
    radial_symmetry: float           # Loy-Zelinsky score
    hair_coherence: float            # 0 = isotropic, 1 = highly oriented (hair)
    signed_contrast: float           # positive = lighter than local, negative = darker
    contrast_ratio: float            # |Δlum| / local skin std
    local_chroma_delta: float        # Δ in a* channel vs local neighborhood
    poreness: float                  # combined score
    source_field: str                # "h_basin" or "h_dome" (Fitz V-VI)
    features: dict = field(default_factory=dict)


# =============================================================================
# Multi-scale DoG
# =============================================================================
def multi_scale_dog(
    L: np.ndarray,
    mask: np.ndarray,
    sigmas: List[float],
    sigma_ratio: float = 1.6,
) -> np.ndarray:
    """Multi-scale DoG response map.

    At each scale σ, DoG = G(σ) - G(σ · sigma_ratio).
    We return the element-wise max across scales (for bright-blob response)
    and min (for dark-blob response), combined as |max-min|.
    """
    L_f = L.astype(np.float32)
    mask_bool = mask.astype(bool)

    best_pos = np.zeros_like(L_f)    # positive (bright) response
    best_neg = np.zeros_like(L_f)    # negative (dark) response
    best_scale = np.zeros_like(L_f)

    for sigma in sigmas:
        g1 = cv2.GaussianBlur(L_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
        g2 = cv2.GaussianBlur(L_f, (0, 0), sigmaX=sigma * sigma_ratio, sigmaY=sigma * sigma_ratio)
        dog = g1 - g2
        # Track per-pixel maximum absolute response and its scale.
        pos_update = dog > best_pos
        neg_update = dog < best_neg
        best_pos = np.where(pos_update, dog, best_pos)
        best_neg = np.where(neg_update, dog, best_neg)
        best_scale = np.where(np.abs(dog) > np.abs(best_scale * 0 + best_pos + best_neg),
                              sigma, best_scale)

    abs_max = np.maximum(np.abs(best_pos), np.abs(best_neg))
    response = np.where(np.abs(best_pos) >= np.abs(best_neg), best_pos, best_neg)
    response[~mask_bool] = 0.0
    return response, best_scale, abs_max


# =============================================================================
# h-basin and h-dome transforms (Vincent 1993)
# =============================================================================
def h_basin_transform(L: np.ndarray, h: float) -> np.ndarray:
    """h-basin: dark structures at least h deep than surround.

    Computed as: L - reconstruction(L + h, mask=L, by=erosion).
    Equivalent to: grey_reconstruction of (L + h) under L, then subtract.

    Returns an image where each pixel's value is the "depth" of its local
    basin (pore candidate intensity), 0 if not a basin.
    """
    # skimage's reconstruction performs morphological reconstruction.
    marker = L + h
    recon = _grey_reconstruction(marker, L, method="erosion")
    basin = recon - L
    basin = np.clip(basin, 0, None)
    return basin.astype(np.float32)


def h_dome_transform(L: np.ndarray, h: float) -> np.ndarray:
    """h-dome: bright structures at least h high above surround.

    Dual of h-basin. For Fitzpatrick V-VI where pore openings can present
    as local luminance maxima.
    """
    marker = L - h
    recon = _grey_reconstruction(marker, L, method="dilation")
    dome = L - recon
    dome = np.clip(dome, 0, None)
    return dome.astype(np.float32)


# =============================================================================
# Loy-Zelinsky Fast Radial Symmetry Transform
# =============================================================================
def fast_radial_symmetry(
    L: np.ndarray,
    radii: List[int],
    alpha: float = 2.0,
    beta: float = 0.1,
) -> np.ndarray:
    """Fast Radial Symmetry Transform (Loy & Zelinski, PAMI 2003).

    Produces a 'symmetry votes' map where pixels at the centers of
    radially-symmetric dark/bright structures score high.

    Args:
        L: HxW float32 image (L* channel recommended).
        radii: integer radii in pixels at which to compute symmetry.
        alpha: radial strictness (higher = more strict about symmetry).
        beta: gradient magnitude threshold as a fraction of max.

    Returns:
        HxW float32 combined symmetry map (summed over radii, dark-favored).
    """
    L_f = L.astype(np.float32)
    gx = cv2.Sobel(L_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L_f, cv2.CV_32F, 0, 1, ksize=3)
    g_mag = np.sqrt(gx * gx + gy * gy)
    g_max = g_mag.max()
    if g_max < 1e-6:
        return np.zeros_like(L_f)
    # Unit gradient vectors (safely).
    gmag_safe = np.where(g_mag > beta * g_max, g_mag, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ux = np.where(gmag_safe > 0, gx / gmag_safe, 0)
        uy = np.where(gmag_safe > 0, gy / gmag_safe, 0)

    h, w = L_f.shape
    ys, xs = np.indices(L_f.shape)
    combined = np.zeros_like(L_f)

    for n in radii:
        # Orientation-projection and magnitude-projection images.
        On = np.zeros_like(L_f)
        Mn = np.zeros_like(L_f)

        # For each pixel with significant gradient, cast a vote along
        # the negative-gradient direction (dark-favored symmetry; common
        # for pore detection: dark centers surrounded by brighter skin).
        valid = gmag_safe > 0
        py = np.round(ys + n * (-uy)).astype(np.int32)
        px = np.round(xs + n * (-ux)).astype(np.int32)
        keep = valid & (px >= 0) & (px < w) & (py >= 0) & (py < h)
        On_flat = On.ravel()
        Mn_flat = Mn.ravel()
        flat_idx = (py[keep] * w + px[keep]).astype(np.int64)
        np.add.at(On_flat, flat_idx, 1.0)            # vote count
        np.add.at(Mn_flat, flat_idx, gmag_safe[keep])

        On = On_flat.reshape(h, w)
        Mn = Mn_flat.reshape(h, w)

        # Normalize.
        Kn = max(On.max(), 1.0)
        On_n = On / Kn
        Mn_n = Mn / (Mn.max() + 1e-6)
        Sn = np.power(np.abs(On_n), alpha) * Mn_n
        Sn = cv2.GaussianBlur(Sn, (0, 0), sigmaX=0.5 * n, sigmaY=0.5 * n)
        combined += Sn

    return combined / max(1, len(radii))


# =============================================================================
# 8-orientation steerable filter bank for hair-coherence
# =============================================================================
def hair_coherence_map(L: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Compute a per-pixel 'hair coherence' score in [0, 1].

    Uses structure-tensor analysis: highly anisotropic regions (coherent
    orientation, e.g. hair strands) score close to 1; isotropic regions
    (e.g. the center of a pore) score close to 0.

    Structure tensor J = [[Jxx, Jxy], [Jxy, Jyy]] smoothed with Gaussian.
    Coherence = (λ1 - λ2) / (λ1 + λ2).
    """
    L_f = L.astype(np.float32)
    gx = cv2.Sobel(L_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L_f, cv2.CV_32F, 0, 1, ksize=3)
    Jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigmaX=sigma, sigmaY=sigma)
    Jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigmaX=sigma, sigmaY=sigma)
    Jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigmaX=sigma, sigmaY=sigma)

    trace = Jxx + Jyy
    det = Jxx * Jyy - Jxy * Jxy
    disc = np.sqrt(np.maximum(0, trace * trace - 4 * det))
    lam1 = (trace + disc) * 0.5
    lam2 = (trace - disc) * 0.5
    coherence = np.where(trace > 1e-6, (lam1 - lam2) / (trace + 1e-6), 0.0)
    return np.clip(coherence, 0.0, 1.0).astype(np.float32)


# =============================================================================
# Non-maximum suppression with adaptive radius
# =============================================================================
def non_max_suppression(
    score_map: np.ndarray,
    mask: np.ndarray,
    min_distance_px: int,
    min_score: float,
) -> np.ndarray:
    """Standard local-maximum NMS with a minimum-separation constraint."""
    if min_distance_px < 1:
        min_distance_px = 1
    size = 2 * min_distance_px + 1
    local_max = maximum_filter(score_map, size=size, mode="constant", cval=0.0)
    is_peak = (score_map == local_max) & (score_map >= min_score) & mask.astype(bool)
    ys, xs = np.where(is_peak)
    return np.stack([xs, ys], axis=1)   # Nx2 (x, y)


# =============================================================================
# Main Stage 6 entry point
# =============================================================================
def detect_pore_candidates(
    L_normalized: np.ndarray,         # from Stage 4
    lab: np.ndarray,                  # from Stage 4 (for chroma features)
    skin_mask: np.ndarray,            # from Stage 3
    zone_of: callable,                # function (x, y) -> zone_id or None
    scale: ScaleCalibration,          # from Stage 2
    fitzpatrick: str,
) -> List[PoreCandidate]:
    """Run full Stage 6 classical candidate detection.

    Returns a list of PoreCandidate objects, one per surviving local maximum.
    """
    phot = PHOTOTYPE_PARAMS.get(fitzpatrick, PHOTOTYPE_PARAMS["III"])

    # Derive image-specific pore-pixel window from ppmm.
    d_min_px = max(1.5, scale.mm_to_px(PORE_DIAMETER_MIN_MM))
    d_max_px = max(d_min_px + 1.0, scale.mm_to_px(PORE_DIAMETER_MAX_MM))
    r_min = max(1.0, d_min_px / 2.0)
    r_max = max(r_min + 0.5, d_max_px / 2.0)

    # DoG scales: logarithmic span between r_min and r_max.
    n_scales = max(3, int(np.ceil(np.log2(r_max / r_min) * 2)))
    sigmas = np.logspace(np.log10(r_min), np.log10(r_max), num=n_scales).tolist()

    # (a) Multi-scale DoG
    dog_response, best_scale_map, dog_abs = multi_scale_dog(
        L_normalized, skin_mask, sigmas
    )

    # (b) h-basin (and optionally h-dome for Fitzpatrick V-VI)
    basin = h_basin_transform(L_normalized, h=phot.h_basin)
    use_dome = fitzpatrick in ("V", "VI")
    dome = h_dome_transform(L_normalized, h=phot.h_dome) if use_dome else None

    # (c) Fast radial symmetry at pore-scale radii.
    radii = [int(round(r)) for r in np.linspace(r_min, r_max, num=3)]
    radii = [max(1, r) for r in radii]
    sym_map = fast_radial_symmetry(L_normalized, radii)

    # (d) Hair coherence (structure-tensor anisotropy).
    coh_map = hair_coherence_map(L_normalized, sigma=max(1.0, r_min))

    # Pore-ness field: weighted sum. Weights are deliberately conservative
    # (documented); each term is normalized roughly to [0, 1] before combining.
    def _rescale01(x):
        x = x.astype(np.float32)
        mn, mx = float(x.min()), float(x.max())
        if mx - mn < 1e-6:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    dog_norm = _rescale01(dog_abs)
    basin_norm = _rescale01(basin)
    sym_norm = _rescale01(sym_map)
    dome_norm = _rescale01(dome) if dome is not None else None

    # Basin-centered field (dark-favored).
    field_basin = (0.35 * dog_norm + 0.40 * basin_norm + 0.25 * sym_norm)
    field_basin = field_basin * (1.0 - 0.5 * coh_map)     # demote oriented
    field_basin = field_basin * skin_mask.astype(np.float32)

    candidates_px_lists = []
    # Minimum separation between candidates — set to the expected pore radius
    # so adjacent pores don't collide, but sub-pore-size noise is suppressed.
    min_sep_px = max(2, int(round(r_min * 2.0)))
    # min_score on the normalized [0,1] poreness field. 0.30 empirically
    # eliminates most texture noise while preserving true-pore responses.
    min_score = 0.30
    peaks_basin = non_max_suppression(field_basin, skin_mask, min_sep_px, min_score)
    candidates_px_lists.append(("h_basin", peaks_basin, field_basin))

    if use_dome:
        field_dome = (0.35 * dog_norm + 0.40 * dome_norm + 0.25 * sym_norm)
        field_dome = field_dome * (1.0 - 0.5 * coh_map)
        field_dome = field_dome * skin_mask.astype(np.float32)
        peaks_dome = non_max_suppression(field_dome, skin_mask, min_sep_px, min_score)
        candidates_px_lists.append(("h_dome", peaks_dome, field_dome))

    # Build PoreCandidate records with full feature vectors.
    candidates: List[PoreCandidate] = []
    h_img, w_img = L_normalized.shape
    L_full = L_normalized.astype(np.float32)
    a_chan = lab[..., 1].astype(np.float32)

    # Pre-compute local skin std in a wide neighborhood (for contrast ratio).
    local_std = cv2.GaussianBlur(L_full * L_full, (0, 0), sigmaX=r_max * 4, sigmaY=r_max * 4)
    local_mean = cv2.GaussianBlur(L_full, (0, 0), sigmaX=r_max * 4, sigmaY=r_max * 4)
    local_var = np.maximum(0, local_std - local_mean * local_mean)
    local_std = np.sqrt(local_var) + 1e-6

    for source, peaks, field in candidates_px_lists:
        for x, y in peaks:
            if not skin_mask[y, x]:
                continue
            zone_id = zone_of(int(x), int(y))
            if zone_id is None:
                continue

            # Features — all classical, all deterministic.
            dr = float(dog_response[y, x])
            hp = float(basin[y, x] if source == "h_basin" else dome[y, x])
            rs = float(sym_map[y, x])
            hc = float(coh_map[y, x])

            # Contrast: |Δluminance| vs local mean, normalized by local std.
            L_center = float(L_full[y, x])
            L_local  = float(local_mean[y, x])
            signed = L_center - L_local
            contrast_ratio = abs(signed) / float(local_std[y, x])

            # Local chroma delta (a* channel, small neighborhood).
            r_local = int(round(r_max * 3))
            y0 = max(0, y - r_local); y1 = min(h_img, y + r_local + 1)
            x0 = max(0, x - r_local); x1 = min(w_img, x + r_local + 1)
            a_patch = a_chan[y0:y1, x0:x1]
            chroma_delta = float(a_chan[y, x] - np.median(a_patch))

            candidates.append(PoreCandidate(
                x=int(x), y=int(y),
                zone_id=zone_id,
                scale_px=float(best_scale_map[y, x]),
                dog_response=dr,
                h_prominence=hp,
                radial_symmetry=rs,
                hair_coherence=hc,
                signed_contrast=signed,
                contrast_ratio=contrast_ratio,
                local_chroma_delta=chroma_delta,
                poreness=float(field[y, x]),
                source_field=source,
                features={
                    "sigmas_px": sigmas,
                    "fitzpatrick": fitzpatrick,
                    "ppmm": scale.ppmm,
                    "r_min_px": r_min,
                    "r_max_px": r_max,
                },
            ))

    # Phototype-conditioned thresholding at the candidate stage itself.
    filtered: List[PoreCandidate] = []
    for c in candidates:
        if c.radial_symmetry < phot.radial_symmetry_min * 1e-3:
            # sym_map after normalisation is small in absolute terms;
            # we convert the config threshold to normalised space by scaling.
            pass   # we're lenient at candidate stage; gate harder at Stage 7
        if c.contrast_ratio < phot.contrast_ratio_min * 0.5:
            continue
        if c.hair_coherence > 0.75:
            continue  # obvious hair strand
        filtered.append(c)

    return filtered
