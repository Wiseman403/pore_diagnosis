"""
Stage 7: Zero-shot false-positive suppression.

Three signals, combined by rule:
    (A) DINOv2 within-zone outlier scoring.
        Rationale: true pores in a zone look like each other in a
        self-supervised embedding space; outliers are disproportionately
        hairs, scars, seborrheic keratoses, makeup specks.
    (B) SAM 2 point-prompted segmentation.
        Rationale: gives pixel-accurate per-candidate masks → real area
        in mm², real eccentricity, real boundary smoothness. Reject by
        shape gates (too big, too elongated, too ragged).
    (C) Depth Anything V2 pseudo-depth profile.
        Rationale: ice-pick scars have distinctive deep concave profiles
        different from pores' shallower profiles. Pseudo-depth is relative,
        not metric, so we use profile shape (curvature, asymmetry).

All models are loaded lazily (first use) and run zero-shot (no training,
no fine-tuning). If a model fails to load (missing GPU, no network),
its signal is skipped — the pipeline still runs, with a warning recorded
in the output.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .candidates import PoreCandidate
from .config import PHOTOTYPE_PARAMS, PORE_DIAMETER_MAX_MM, PORE_ECCENTRICITY_MAX
from .utils import ScaleCalibration


log = logging.getLogger(__name__)


# =============================================================================
# Per-candidate enriched record after Stage 7
# =============================================================================
@dataclass
class AcceptedPore:
    """A candidate that passed Stage 7. Carries all Stage 6 features plus
    the Stage 7 signals (shape from SAM 2, outlier score from DINOv2,
    scar-likelihood from Depth Anything V2)."""
    candidate: PoreCandidate
    # Signal A: DINOv2
    dino_outlier_score: float               # Mahalanobis distance (higher = more outlier)
    dino_outlier_percentile: float          # rank within its zone (0-100)
    # Signal B: SAM 2
    sam_mask: Optional[np.ndarray] = None   # HxW uint8, 1 on the pore pixels
    sam_mask_bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) tight bbox
    sam_area_px: float = 0.0
    sam_area_mm2: float = 0.0
    sam_diameter_mm: float = 0.0
    sam_eccentricity: float = 0.0
    sam_boundary_entropy: float = 0.0
    sam_mask_stability_iou: float = 1.0
    # Signal C: Depth Anything V2
    depth_profile_curvature: float = 0.0
    depth_profile_asymmetry: float = 0.0
    depth_scar_likelihood: float = 0.0      # 0=pore-like, 1=scar-like
    # Combined
    accepted: bool = True
    rejection_reasons: List[str] = field(default_factory=list)


# =============================================================================
# Foundation-model wrappers (lazy-loading, graceful degradation)
# =============================================================================
class DinoV2Encoder:
    """Zero-shot DINOv2 patch encoder. Uses HuggingFace transformers.

    We use ViT-S/14 (~22M params, fits on T4 with room to spare).
    """

    def __init__(self, model_id: str = "facebook/dinov2-small", device: str = "auto"):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._processor = AutoImageProcessor.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(self.model_id).to(self.device)
            self._model.eval()
        except Exception as e:
            raise RuntimeError(f"DINOv2 failed to load ({self.model_id}): {e}")

    def encode_patches(self, bgr: np.ndarray, centers_xy: np.ndarray,
                       patch_size: int = 64) -> np.ndarray:
        """Encode NxCxHxW patches → (N, D) DINOv2 CLS embeddings.

        Args:
            bgr: HxWx3 source image (full face).
            centers_xy: (N, 2) integer pixel centers.
            patch_size: patch side in pixels (will be resized to 224 for DINOv2).
        """
        self._ensure_loaded()
        import torch
        from PIL import Image

        h, w = bgr.shape[:2]
        half = patch_size // 2
        patches: List[Image.Image] = []
        for (cx, cy) in centers_xy:
            x0 = max(0, int(cx) - half); x1 = min(w, int(cx) + half)
            y0 = max(0, int(cy) - half); y1 = min(h, int(cy) + half)
            patch = bgr[y0:y1, x0:x1]
            if patch.shape[0] == 0 or patch.shape[1] == 0:
                patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
            # Pad if near image edge.
            pad_y = patch_size - patch.shape[0]
            pad_x = patch_size - patch.shape[1]
            if pad_y > 0 or pad_x > 0:
                patch = cv2.copyMakeBorder(patch, 0, max(0, pad_y), 0, max(0, pad_x),
                                           cv2.BORDER_REFLECT_101)
            # Convert BGR → RGB PIL.
            rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            patches.append(Image.fromarray(rgb))

        # Batch process.
        embeddings: List[np.ndarray] = []
        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                batch = patches[i:i + batch_size]
                inputs = self._processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self._model(**inputs)
                # Use CLS token (first token of last hidden state).
                cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls)
        return np.vstack(embeddings) if embeddings else np.zeros((0, 384), dtype=np.float32)


class Sam2Prompter:
    """Zero-shot SAM 2 with point prompting.

    Uses the HuggingFace `Sam2Model` interface. Each candidate gets a
    point prompt; we request multiple masks and pick the smallest that
    is topologically consistent with a pore.
    """

    def __init__(self, model_id: str = "facebook/sam2-hiera-base-plus",
                 device: str = "auto"):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import Sam2Model, Sam2Processor
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._processor = Sam2Processor.from_pretrained(self.model_id)
            self._model = Sam2Model.from_pretrained(self.model_id).to(self.device)
            self._model.eval()
        except Exception as e:
            raise RuntimeError(f"SAM 2 failed to load ({self.model_id}): {e}")

    def segment_points(self, bgr: np.ndarray,
                       points_xy: np.ndarray,
                       jitter_px: int = 1) -> List[np.ndarray]:
        """Segment each point independently. Returns a list of HxW uint8 masks.

        For each candidate, we prompt with the center point plus two jittered
        variants and keep the most stable (highest IoU agreement) mask.
        The jitter gives us a mask-stability signal for confidence scoring.
        """
        self._ensure_loaded()
        import torch
        from PIL import Image

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        masks_out: List[np.ndarray] = []
        stabilities: List[float] = []

        for (cx, cy) in points_xy:
            # Three prompt variants: center, +jitter, -jitter.
            prompts = [[int(cx), int(cy)],
                       [int(cx) + jitter_px, int(cy) + jitter_px],
                       [int(cx) - jitter_px, int(cy) - jitter_px]]
            variant_masks: List[np.ndarray] = []
            for p in prompts:
                try:
                    inputs = self._processor(
                        images=pil,
                        input_points=[[[p]]],
                        input_labels=[[[1]]],     # 1 = foreground
                        return_tensors="pt",
                    ).to(self.device)
                    with torch.no_grad():
                        outputs = self._model(**inputs, multimask_output=True)
                    # Post-process to HxW mask. Use the first (highest-scored) mask.
                    masks = self._processor.post_process_masks(
                        outputs.pred_masks.cpu(),
                        inputs["original_sizes"].cpu(),
                    )
                    # masks: list of tensors of shape (1, num_masks, H, W); pick best.
                    first_img_masks = masks[0][0].numpy()   # (num_masks, H, W) bool
                    iou_scores = outputs.iou_scores.cpu().numpy().ravel()
                    best_idx = int(np.argmax(iou_scores))
                    variant_masks.append(first_img_masks[best_idx].astype(np.uint8))
                except Exception as e:
                    log.warning("SAM2 segmentation failed on point (%d, %d): %s", cx, cy, e)
                    variant_masks.append(np.zeros(bgr.shape[:2], dtype=np.uint8))

            # Cross-variant IoU for stability.
            ious = []
            for i in range(len(variant_masks)):
                for j in range(i + 1, len(variant_masks)):
                    inter = np.logical_and(variant_masks[i], variant_masks[j]).sum()
                    uni = np.logical_or(variant_masks[i], variant_masks[j]).sum()
                    if uni > 0:
                        ious.append(inter / uni)
            stability = float(np.mean(ious)) if ious else 0.0

            # Use the median-area mask (robust to outlier prompts).
            areas = [int(m.sum()) for m in variant_masks]
            med_idx = int(np.argsort(areas)[len(areas) // 2])
            masks_out.append(variant_masks[med_idx])
            stabilities.append(stability)

        return masks_out, stabilities


class DepthAnythingV2Encoder:
    """Zero-shot Depth Anything V2 for pseudo-depth map."""

    def __init__(self, model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
                 device: str = "auto"):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._processor = AutoImageProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForDepthEstimation.from_pretrained(self.model_id).to(self.device)
            self._model.eval()
        except Exception as e:
            raise RuntimeError(f"Depth Anything V2 failed to load ({self.model_id}): {e}")

    def depth(self, bgr: np.ndarray) -> np.ndarray:
        """Return a HxW float32 pseudo-depth map in arbitrary units.

        Higher values = further (standard convention for DAv2).
        """
        self._ensure_loaded()
        import torch
        from PIL import Image

        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        with torch.no_grad():
            inputs = self._processor(images=pil, return_tensors="pt").to(self.device)
            outputs = self._model(**inputs)
            pred = outputs.predicted_depth[0].cpu().numpy()
        # Resize to original.
        depth = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        return depth


# =============================================================================
# Signal A: DINOv2 within-zone outlier scoring
# =============================================================================
def compute_dino_outliers(
    candidates: List[PoreCandidate],
    bgr: np.ndarray,
    dino: Optional[DinoV2Encoder],
    patch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Score each candidate by within-zone Mahalanobis outlier-ness.

    Returns:
        outlier_scores: (N,) float — larger = more outlier-like.
        outlier_percentiles: (N,) float in [0, 100] — rank within zone.
    """
    n = len(candidates)
    if n == 0 or dino is None:
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)

    centers = np.array([[c.x, c.y] for c in candidates], dtype=np.int32)
    try:
        embeddings = dino.encode_patches(bgr, centers, patch_size=patch_size)
    except Exception as e:
        log.warning("DINOv2 encoding failed, skipping Signal A: %s", e)
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)

    # Group by zone.
    zone_ids = np.array([c.zone_id for c in candidates])
    scores = np.zeros(n, dtype=np.float32)
    pctls = np.zeros(n, dtype=np.float32)

    for z in np.unique(zone_ids):
        idx = np.where(zone_ids == z)[0]
        if len(idx) < 3:
            # Too few candidates in this zone for robust stats; skip.
            continue
        emb = embeddings[idx]
        mean = emb.mean(axis=0, keepdims=True)
        centered = emb - mean
        # Regularized covariance (Ledoit-Wolf-ish shrinkage).
        cov = np.cov(centered, rowvar=False) + 1e-3 * np.eye(emb.shape[1])
        try:
            cov_inv = np.linalg.pinv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.eye(emb.shape[1])
        # Mahalanobis distance squared (per sample).
        diffs = centered
        md2 = np.einsum("ij,jk,ik->i", diffs, cov_inv, diffs)
        md = np.sqrt(np.maximum(md2, 0))
        scores[idx] = md
        # Percentile rank within zone.
        order = md.argsort().argsort().astype(np.float32)
        pctls[idx] = 100.0 * order / max(1, len(idx) - 1)

    return scores, pctls


# =============================================================================
# Signal B helpers: shape features from a binary mask
# =============================================================================
def mask_shape_features(mask: np.ndarray) -> Dict[str, float]:
    """Compute geometric features from a tight binary mask of a pore."""
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return {"area_px": 0.0, "eccentricity": 1.0, "boundary_entropy": 1.0,
                "bbox": (0, 0, 0, 0), "diameter_px": 0.0}
    # Contours for boundary-smoothness analysis.
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {"area_px": 0.0, "eccentricity": 1.0, "boundary_entropy": 1.0,
                "bbox": (0, 0, 0, 0), "diameter_px": 0.0}
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    # Equivalent-circle diameter.
    diameter = 2.0 * np.sqrt(area / np.pi) if area > 0 else 0.0
    # Eccentricity via second moments.
    if len(contour) >= 5:
        (_, _), (axis_a, axis_b), _ = cv2.fitEllipse(contour)
        major = max(axis_a, axis_b); minor = min(axis_a, axis_b)
        ecc = np.sqrt(1 - (minor ** 2) / (major ** 2)) if major > 0 else 0.0
    else:
        ecc = 0.0
    # Boundary entropy: angular turning angle variability. Low for smooth
    # pores, high for ragged artefacts.
    pts = contour[:, 0, :]
    if len(pts) < 8:
        boundary_entropy = 0.0
    else:
        d = np.diff(pts, axis=0)
        angles = np.arctan2(d[:, 1], d[:, 0])
        turns = np.diff(np.unwrap(angles))
        # Histogram entropy.
        hist, _ = np.histogram(turns, bins=16, range=(-np.pi / 2, np.pi / 2))
        p = hist / max(1, hist.sum())
        p = p[p > 0]
        boundary_entropy = float(-np.sum(p * np.log(p))) / np.log(16)

    x, y, w, h = cv2.boundingRect(contour)
    return {
        "area_px": area,
        "eccentricity": float(ecc),
        "boundary_entropy": float(boundary_entropy),
        "bbox": (int(x), int(y), int(w), int(h)),
        "diameter_px": float(diameter),
    }


# =============================================================================
# Signal C: Depth profile analysis for scar vs. pore discrimination
# =============================================================================
def depth_profile_scar_score(
    depth_map: np.ndarray,
    cx: int, cy: int,
    radius_px: int,
) -> Tuple[float, float, float]:
    """Analyse a cross-section of the pseudo-depth map at a candidate.

    Ice-pick scars have a deep, narrow concavity (high curvature, may be
    asymmetric if oblique). Pores have a shallow, broader concavity.
    We compute:
        curvature: 2nd derivative magnitude at center (normalized).
        asymmetry: |left_profile - right_profile|.
        scar_likelihood: heuristic combination, 0=pore-like, 1=scar-like.

    Returns (curvature, asymmetry, scar_likelihood).
    """
    h, w = depth_map.shape
    r = max(2, int(radius_px))
    # Extract horizontal cross-section.
    y0 = max(0, cy); x0 = max(0, cx - r); x1 = min(w, cx + r + 1)
    if x1 - x0 < 5:
        return 0.0, 0.0, 0.0
    profile = depth_map[y0, x0:x1].astype(np.float32)
    # Normalize to [0, 1] across the profile.
    pmin, pmax = profile.min(), profile.max()
    if pmax - pmin < 1e-6:
        return 0.0, 0.0, 0.0
    prof = (profile - pmin) / (pmax - pmin)
    center = len(prof) // 2
    # Curvature at center (finite-differences 2nd derivative).
    if center > 0 and center < len(prof) - 1:
        curv = float(prof[center - 1] - 2 * prof[center] + prof[center + 1])
    else:
        curv = 0.0
    # Asymmetry: compare left and right halves.
    left = prof[:center]; right = prof[center + 1:][::-1]
    mn = min(len(left), len(right))
    if mn == 0:
        asym = 0.0
    else:
        asym = float(np.mean(np.abs(left[-mn:] - right[:mn])))

    # Heuristic scar score:
    #   - deep narrow concavity (large |curv|) + asymmetry > 0.15 → scar-like
    #   - shallow symmetric concavity → pore-like
    scar = min(1.0, max(0.0, abs(curv) * 3.0 + asym * 2.0 - 0.3))
    return float(curv), float(asym), float(scar)


# =============================================================================
# Main Stage 7 entry point
# =============================================================================
def suppress_false_positives(
    candidates: List[PoreCandidate],
    bgr: np.ndarray,
    scale: ScaleCalibration,
    fitzpatrick: str,
    dino: Optional[DinoV2Encoder] = None,
    sam: Optional[Sam2Prompter] = None,
    depth_model: Optional[DepthAnythingV2Encoder] = None,
    max_candidates_to_sam: int = 2000,
) -> Tuple[List[AcceptedPore], List[AcceptedPore], List[str]]:
    """Apply the three-signal zero-shot false-positive rule.

    Args:
        max_candidates_to_sam: SAM 2 is the most expensive signal; we cap
            its use at this many top-ranked candidates (by poreness).
            Beyond this cap, SAM signal is skipped and decision falls on
            A + C + classical gates.

    Returns:
        accepted: candidates that passed all gates.
        rejected: candidates that failed (with rejection_reasons).
        warnings: pipeline-level warnings (e.g., a model failed to load).
    """
    warnings: List[str] = []
    if not candidates:
        return [], [], warnings

    phot = PHOTOTYPE_PARAMS.get(fitzpatrick, PHOTOTYPE_PARAMS["III"])

    # --- Signal A: DINOv2 within-zone outliers ---
    if dino is not None:
        try:
            dino_scores, dino_pctls = compute_dino_outliers(candidates, bgr, dino)
        except Exception as e:
            warnings.append(f"DINOv2 failed: {e}")
            dino_scores = np.zeros(len(candidates), dtype=np.float32)
            dino_pctls = np.zeros(len(candidates), dtype=np.float32)
    else:
        warnings.append("DINOv2 not loaded; Signal A skipped.")
        dino_scores = np.zeros(len(candidates), dtype=np.float32)
        dino_pctls = np.zeros(len(candidates), dtype=np.float32)

    # --- Signal B: SAM 2 masks (top-K by poreness) ---
    poreness = np.array([c.poreness for c in candidates], dtype=np.float32)
    sam_order = np.argsort(-poreness)            # descending
    sam_indices = set(sam_order[:max_candidates_to_sam].tolist())

    sam_masks: Dict[int, np.ndarray] = {}
    sam_stabilities: Dict[int, float] = {}
    if sam is not None and len(sam_indices) > 0:
        try:
            pts = np.array([[candidates[i].x, candidates[i].y]
                            for i in sorted(sam_indices)], dtype=np.int32)
            masks, stabs = sam.segment_points(bgr, pts)
            for (i, m, s) in zip(sorted(sam_indices), masks, stabs):
                sam_masks[i] = m
                sam_stabilities[i] = s
        except Exception as e:
            warnings.append(f"SAM 2 failed: {e}")
    else:
        if sam is None:
            warnings.append("SAM 2 not loaded; Signal B skipped (using classical shape features only).")

    # --- Signal C: Depth Anything V2 ---
    depth_map: Optional[np.ndarray] = None
    if depth_model is not None:
        try:
            depth_map = depth_model.depth(bgr)
        except Exception as e:
            warnings.append(f"Depth Anything V2 failed: {e}")
    else:
        warnings.append("Depth Anything V2 not loaded; Signal C skipped.")

    # --- Combine per-candidate ---
    accepted: List[AcceptedPore] = []
    rejected: List[AcceptedPore] = []

    area_max_px = (PORE_DIAMETER_MAX_MM * scale.ppmm / 2.0) ** 2 * np.pi
    max_dino_pctl = phot.dino_outlier_percentile

    for i, c in enumerate(candidates):
        reasons: List[str] = []
        # Signal A check.
        if dino_pctls[i] > max_dino_pctl:
            reasons.append(f"dino_outlier_pctl={dino_pctls[i]:.1f}>{max_dino_pctl:.1f}")

        # Signal B: SAM shape features.
        shape_features: Dict[str, float] = {}
        sam_mask = sam_masks.get(i)
        sam_stab = sam_stabilities.get(i, 1.0)
        if sam_mask is not None:
            shape_features = mask_shape_features(sam_mask)
            if shape_features["area_px"] > area_max_px:
                reasons.append(f"sam_too_large={shape_features['area_px']:.0f}>{area_max_px:.0f}")
            if shape_features["eccentricity"] > PORE_ECCENTRICITY_MAX:
                reasons.append(f"sam_eccentric={shape_features['eccentricity']:.2f}>{PORE_ECCENTRICITY_MAX}")
            if shape_features["boundary_entropy"] > 0.85:
                reasons.append(f"sam_ragged={shape_features['boundary_entropy']:.2f}")
            if sam_stab < 0.4:
                reasons.append(f"sam_unstable_iou={sam_stab:.2f}")

        # Signal C: depth profile.
        depth_curv = depth_asym = depth_scar = 0.0
        if depth_map is not None:
            r_px = max(2, int(round(c.scale_px)))
            depth_curv, depth_asym, depth_scar = depth_profile_scar_score(
                depth_map, c.x, c.y, r_px
            )
            if depth_scar > 0.75:
                reasons.append(f"depth_scar_like={depth_scar:.2f}")

        # Classical safety net.
        if c.hair_coherence > 0.75:
            reasons.append(f"hair_coherence={c.hair_coherence:.2f}")
        if c.contrast_ratio < phot.contrast_ratio_min * 0.5:
            reasons.append(f"low_contrast={c.contrast_ratio:.2f}")

        accepted_flag = len(reasons) == 0

        ap = AcceptedPore(
            candidate=c,
            dino_outlier_score=float(dino_scores[i]),
            dino_outlier_percentile=float(dino_pctls[i]),
            sam_mask=sam_mask,
            sam_mask_bbox=shape_features.get("bbox"),
            sam_area_px=shape_features.get("area_px", 0.0),
            sam_area_mm2=scale.px2_to_mm2(shape_features.get("area_px", 0.0)),
            sam_diameter_mm=scale.px_to_mm(shape_features.get("diameter_px", 0.0)),
            sam_eccentricity=shape_features.get("eccentricity", 0.0),
            sam_boundary_entropy=shape_features.get("boundary_entropy", 0.0),
            sam_mask_stability_iou=sam_stab,
            depth_profile_curvature=depth_curv,
            depth_profile_asymmetry=depth_asym,
            depth_scar_likelihood=depth_scar,
            accepted=accepted_flag,
            rejection_reasons=reasons,
        )
        if accepted_flag:
            accepted.append(ap)
        else:
            rejected.append(ap)

    return accepted, rejected, warnings
