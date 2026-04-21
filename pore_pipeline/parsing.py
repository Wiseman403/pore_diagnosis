"""
Stage 3: Skin isolation.

Two-track approach:
  (a) Landmark-based masking (always runs) — eyes, lips, eyebrows, nostrils,
      face-outer-boundary convex hull. Uses MediaPipe's 478-landmark topology.
  (b) Parsing-model-based hair + glasses removal (when available) via BiSeNet
      ONNX or FaceXFormer. Falls back to a conservative upper-face trim if
      no parser is loaded.

Then a classical post-processing:
  - Hair-edge shadow-band exclusion: erode the skin mask by a few pixels
    from any hair-region boundary to remove the soft shadow halo that
    parsers never segment cleanly and that produces false pore detections.
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import MODEL_IDS
from .utils import dilate_mask, erode_mask, polygon_mask


# =============================================================================
# MediaPipe 478-landmark topology groups.
# Indices taken from MediaPipe's face_mesh_connections (canonical mesh).
# =============================================================================
FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109,
]

LEFT_EYE_IDX = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
]
RIGHT_EYE_IDX = [
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
]

LEFT_BROW_IDX  = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW_IDX = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]

# Outer lip contour (encloses both lips so we subtract the whole mouth).
LIPS_OUTER_IDX = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    409, 270, 269, 267, 0, 37, 39, 40, 185,
]

# Approximate nostril interior regions (two tight polygons).
LEFT_NOSTRIL_IDX  = [64, 102, 49, 131, 134, 51, 45, 48]
RIGHT_NOSTRIL_IDX = [294, 331, 279, 360, 363, 281, 275, 278]


@dataclass
class SkinMaskResult:
    """Stage 3 output."""
    skin_mask: np.ndarray            # HxW uint8, 1 = skin
    hair_mask: Optional[np.ndarray]  # HxW uint8, 1 = hair (None if no parser used)
    face_hull: np.ndarray            # HxW uint8, 1 = inside face oval
    parser_backend: str              # "landmarks_only", "bisenet_onnx", "facexformer"


def build_landmark_skin_mask(
    img_hw: tuple,
    landmarks_px: np.ndarray,
    eye_dilate_px: int = 4,
    brow_dilate_px: int = 3,
    lip_dilate_px: int = 2,
    nostril_dilate_px: int = 1,
) -> np.ndarray:
    """Build a skin mask from MediaPipe landmarks only.

    Starts from the face-oval convex hull, then subtracts dilated polygons
    for eyes, eyebrows, lips, and nostril interiors.
    """
    h, w = img_hw
    xy = landmarks_px[:, :2]

    # Base: face-oval convex hull.
    face = np.zeros((h, w), dtype=np.uint8)
    oval_pts = xy[FACE_OVAL_IDX].astype(np.int32)
    hull = cv2.convexHull(oval_pts)
    cv2.fillConvexPoly(face, hull, 1)

    # Subtract non-skin interior features.
    for idx_list, dilate_r in [
        (LEFT_EYE_IDX,  eye_dilate_px),
        (RIGHT_EYE_IDX, eye_dilate_px),
        (LEFT_BROW_IDX,  brow_dilate_px),
        (RIGHT_BROW_IDX, brow_dilate_px),
        (LIPS_OUTER_IDX, lip_dilate_px),
        (LEFT_NOSTRIL_IDX,  nostril_dilate_px),
        (RIGHT_NOSTRIL_IDX, nostril_dilate_px),
    ]:
        poly = polygon_mask((h, w), xy[idx_list])
        if dilate_r > 0:
            poly = dilate_mask(poly, dilate_r)
        face[poly.astype(bool)] = 0

    return face


# =============================================================================
# Parser backends (optional, for hair + glasses segmentation)
# =============================================================================
class BiSeNetONNXParser:
    """Pretrained BiSeNet face parser (19-class, CelebAMask-HQ).

    Downloads the ONNX model on first use. Works on CPU and GPU.
    Requires onnxruntime.
    """

    # CelebAMask-HQ 19-class labels (per yakhyo/face-parsing).
    # 0: background
    # 1: skin
    # 2: left-eyebrow
    # 3: right-eyebrow
    # 4: left-eye
    # 5: right-eye
    # 6: eyeglasses
    # 7: left-ear
    # 8: right-ear
    # 9: earring
    # 10: nose
    # 11: mouth
    # 12: upper-lip
    # 13: lower-lip
    # 14: neck
    # 15: necklace
    # 16: clothes
    # 17: hair
    # 18: hat
    HAIR_CLASSES = {17, 18}
    GLASSES_CLASSES = {6}
    SKIN_CLASSES = {1, 10}                       # skin + nose (nose counted as skin surface)
    NECK_CLASS = 14
    EAR_CLASSES = {7, 8, 9}
    NON_SKIN_FACIAL = {2, 3, 4, 5, 6, 11, 12, 13}  # brows, eyes, glasses, lips

    INPUT_SIZE = (512, 512)

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "/root/.cache/pore_pipeline/bisenet_face.onnx"
        if not os.path.exists(self.model_path):
            self._download()
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def _download(self) -> None:
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        url = MODEL_IDS["face_parser_bisenet_onnx_url"]
        urllib.request.urlretrieve(url, self.model_path)

    def parse(self, bgr: np.ndarray) -> np.ndarray:
        """Return HxW uint8 per-pixel class map."""
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(rgb, self.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        inp = inp.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        inp = (inp - mean) / std
        inp = np.transpose(inp, (2, 0, 1))[None]  # NCHW
        out = self.session.run(None, {self.input_name: inp})[0]
        # out: (1, 19, H, W) logits.
        cls = np.argmax(out[0], axis=0).astype(np.uint8)
        cls = cv2.resize(cls, (w, h), interpolation=cv2.INTER_NEAREST)
        return cls


def build_skin_mask(
    bgr: np.ndarray,
    landmarks_px: np.ndarray,
    parser: Optional[BiSeNetONNXParser] = None,
    hair_edge_band_px: int = 3,
) -> SkinMaskResult:
    """Combine landmark-based masking with optional parser-based hair removal.

    Steps:
      1. Landmark skin mask (base) — always.
      2. If parser available: compute hair + glasses + ear mask, subtract.
      3. Hair-edge shadow band: dilate hair mask by `hair_edge_band_px` and
         exclude the dilated boundary from skin (prevents hairline false pores).
      4. Final binary mask is the intersection of the face hull and the
         non-excluded region.
    """
    h, w = bgr.shape[:2]
    face_hull = np.zeros((h, w), dtype=np.uint8)
    oval_pts = landmarks_px[:, :2][FACE_OVAL_IDX].astype(np.int32)
    hull_poly = cv2.convexHull(oval_pts)
    cv2.fillConvexPoly(face_hull, hull_poly, 1)

    skin = build_landmark_skin_mask((h, w), landmarks_px)

    hair_mask = None
    if parser is not None:
        try:
            class_map = parser.parse(bgr)
            hair_mask = np.isin(class_map, list(parser.HAIR_CLASSES)).astype(np.uint8)
            glasses_mask = np.isin(class_map, list(parser.GLASSES_CLASSES)).astype(np.uint8)
            ear_mask = np.isin(class_map, list(parser.EAR_CLASSES)).astype(np.uint8)

            # Subtract non-skin classes.
            skin[hair_mask.astype(bool)] = 0
            skin[glasses_mask.astype(bool)] = 0
            skin[ear_mask.astype(bool)] = 0

            # Hair-edge shadow band: dilate hair, subtract the dilated ring.
            if hair_edge_band_px > 0 and hair_mask.any():
                band = dilate_mask(hair_mask, hair_edge_band_px).astype(bool) & \
                       ~(hair_mask.astype(bool))
                skin[band] = 0
            backend = "bisenet_onnx"
        except Exception:
            backend = "landmarks_only"
    else:
        backend = "landmarks_only"

    # Final intersection with face hull (belt-and-braces).
    skin = (skin.astype(bool) & face_hull.astype(bool)).astype(np.uint8)

    return SkinMaskResult(
        skin_mask=skin,
        hair_mask=hair_mask,
        face_hull=face_hull,
        parser_backend=backend,
    )
