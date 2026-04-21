"""
Central configuration for the pore analysis pipeline.

All thresholds, zone definitions, phototype-conditioned parameters, and
published constants live here. Nothing here is learned or trained;
everything is either mathematical, anatomical, or literature-derived.

References:
    - Flament F. et al., JAAD 2004 (pore ruler)
    - Eiben-Nielson et al., JOCD 2021 (5-point pore scale)
    - IBSA Composite Skin Quality Scale, Dermatol Surg 2024
    - Dissanayake et al., Skin Res Technol 2019 (image pore characterization)
    - Fitzpatrick thresholds on ITA°: Del Bino & Bernerd 2013
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# =============================================================================
# Versioning
# =============================================================================
SCHEMA_VERSION = "pore.v1.0"
PIPELINE_VERSION = "pore-pipeline-0.1.0"

# =============================================================================
# Geometric constants
# =============================================================================
# Median adult inter-ocular distance (mm). Source: anthropometry literature;
# ±10% inter-subject variability is expected and propagated as uncertainty.
ADULT_IOD_MM = 63.0
IOD_UNCERTAINTY_PCT = 10.0

# Valid pore diameter range (mm), from §2 operational definition.
PORE_DIAMETER_MIN_MM = 0.10
PORE_DIAMETER_MAX_MM = 0.60
PORE_ECCENTRICITY_MAX = 0.85

# Acceptable capture distance range (mm, approximate).
CAPTURE_DISTANCE_MIN_MM = 150.0   # ~15 cm — too close, lens distortion
CAPTURE_DISTANCE_MAX_MM = 600.0   # ~60 cm — too far, pores unresolvable

# Acceptable pixels-per-millimeter range (derived bound for pore resolvability).
PPMM_MIN = 8.0
PPMM_MAX = 50.0

# =============================================================================
# Pose gate (Stage 1)
# =============================================================================
POSE_MAX_YAW_DEG = 20.0
POSE_MAX_PITCH_DEG = 15.0
POSE_MAX_ROLL_DEG = 15.0

# =============================================================================
# Blur / exposure gate (Stage 1)
# =============================================================================
# Variance-of-Laplacian on skin-only pixels. Threshold calibrated against
# qualitative sharp/blurry pair tests; conservative.
BLUR_VAR_LAPLACIAN_MIN = 50.0

EXPOSURE_CLIPPED_PIXEL_FRACTION_MAX = 0.02   # Max 2% clipped pixels
EXPOSURE_L_CHANNEL_IQR_MIN = 10.0            # L* IQR floor on skin

# Texture energy floor (Stage 1) — wavelet-detail RMS on skin-only.
# Values below this suggest ISP beauty smoothing or over-denoising.
# Phototype-dependent; see PHOTOTYPE_PARAMS below.

# =============================================================================
# Fitzpatrick / ITA° buckets (Del Bino & Bernerd 2013)
# =============================================================================
FITZPATRICK_FROM_ITA: List[Tuple[float, str]] = [
    (55.0, "I"),
    (41.0, "II"),
    (28.0, "III"),
    (10.0, "IV"),
    (-30.0, "V"),
    (-1e9, "VI"),
]


def ita_to_fitzpatrick(ita_deg: float) -> str:
    """Map ITA° to Fitzpatrick bucket via standard Del Bino thresholds."""
    for thresh, label in FITZPATRICK_FROM_ITA:
        if ita_deg > thresh:
            return label
    return "VI"


# =============================================================================
# Phototype-conditioned parameters
# =============================================================================
# These thresholds drive candidate detection and false-positive suppression.
# Values are conservative priors from literature + our operational definition.
# They are *parameters*, not learned weights, and can be tuned offline
# (but never at inference-time from user data).

@dataclass
class PhototypeParams:
    # Morphological h-basin / h-dome prominence threshold on normalized L*
    # (the local darkness/brightness a candidate must exceed its surround by).
    h_basin: float
    h_dome: float
    # Loy-Zelinsky radial symmetry acceptance threshold.
    radial_symmetry_min: float
    # Minimum |Δluminance| / local_std for a candidate to pass Stage 6.
    contrast_ratio_min: float
    # DINOv2 Mahalanobis outlier percentile for within-zone rejection.
    # Higher = more permissive (fewer rejections).
    dino_outlier_percentile: float
    # Texture energy floor (Stage 1 beauty-filter detection).
    texture_energy_rms_min: float


# Fitzpatrick I-III: sharper classical contrast, tighter DINOv2 clustering expected.
# Fitzpatrick IV-VI: softer contrast, wider clustering tolerance, dual h-basin/h-dome.
PHOTOTYPE_PARAMS: Dict[str, PhototypeParams] = {
    "I":   PhototypeParams(h_basin=4.0, h_dome=3.0, radial_symmetry_min=0.30, contrast_ratio_min=1.8, dino_outlier_percentile=85.0, texture_energy_rms_min=3.0),
    "II":  PhototypeParams(h_basin=4.0, h_dome=3.0, radial_symmetry_min=0.30, contrast_ratio_min=1.8, dino_outlier_percentile=85.0, texture_energy_rms_min=3.0),
    "III": PhototypeParams(h_basin=3.5, h_dome=3.0, radial_symmetry_min=0.28, contrast_ratio_min=1.7, dino_outlier_percentile=85.0, texture_energy_rms_min=2.8),
    "IV":  PhototypeParams(h_basin=3.0, h_dome=3.5, radial_symmetry_min=0.25, contrast_ratio_min=1.5, dino_outlier_percentile=88.0, texture_energy_rms_min=2.5),
    "V":   PhototypeParams(h_basin=2.5, h_dome=3.5, radial_symmetry_min=0.22, contrast_ratio_min=1.4, dino_outlier_percentile=90.0, texture_energy_rms_min=2.2),
    "VI":  PhototypeParams(h_basin=2.5, h_dome=3.5, radial_symmetry_min=0.22, contrast_ratio_min=1.4, dino_outlier_percentile=90.0, texture_energy_rms_min=2.0),
}


# =============================================================================
# Zone definitions (MediaPipe FaceMesh 478-landmark indices)
# =============================================================================
# These are the anatomical zones from §5 of the plan.
# Each zone is defined as a convex hull over a fixed set of landmark indices.
# Indices are versioned and must not be changed without bumping ZONE_MAP_VERSION.
#
# Reference for MediaPipe indices:
# https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py
ZONE_MAP_VERSION = "v1"

# Landmark index sets per zone. These hulls are intentionally conservative;
# the hair-edge shadow-band exclusion in Stage 3 trims further.
ZONE_LANDMARKS: Dict[str, List[int]] = {
    # Forehead: upper mid-face, between temples, below hairline.
    "forehead": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
                 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
                 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
    # Glabella: between eyebrows, above nasion.
    "glabella": [9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164,
                 336, 296, 334, 293, 300, 285, 295, 282, 283, 276,
                 107, 66, 105, 63, 70, 53, 52, 65, 55, 193],
    # Nose dorsum: bridge from nasion to tip.
    "nose_dorsum": [168, 6, 197, 195, 5, 4, 45, 51, 3, 236, 134, 220,
                    275, 281, 248, 456, 363, 440],
    # Nose tip.
    "nose_tip": [4, 45, 51, 3, 220, 75, 60, 166, 79, 218, 239,
                 275, 281, 248, 440, 305, 290, 392, 309, 438, 459],
    # Nose alae — left and right separately.
    "nose_ala_left":  [129, 209, 49, 64, 98, 97, 2, 326, 327, 278],
    "nose_ala_right": [358, 429, 279, 294, 327, 326, 2, 97, 98, 48],
    # Medial cheek — high-density pore zones adjacent to nasolabial.
    "cheek_medial_left":  [205, 50, 101, 36, 142, 126, 203, 206, 216,
                            212, 202, 57, 186],
    "cheek_medial_right": [425, 280, 330, 266, 371, 355, 423, 426, 436,
                            432, 422, 287, 410],
    # Lateral cheek — zygomatic prominence.
    "cheek_lateral_left":  [116, 117, 118, 119, 120, 121, 47, 126, 142, 36, 205,
                             187, 147, 123, 111, 31, 228, 229, 230, 231, 232, 233],
    "cheek_lateral_right": [345, 346, 347, 348, 349, 350, 277, 355, 371, 266, 425,
                             411, 376, 352, 340, 261, 448, 449, 450, 451, 452, 453],
    # Chin — below lower lip.
    "chin": [152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
             234, 93, 132, 58, 172, 136, 150, 149, 176, 148,
             377, 378, 379, 365, 397, 288, 361, 323, 454, 356,
             389, 251, 284, 332, 297, 338],
    # Perioral — around the mouth.
    "perioral": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                 409, 270, 269, 267, 0, 37, 39, 40, 185],
}


# Clinical weight of each zone in the whole-face composite score (§10).
# Weights reflect clinical concern (patients complain most about nose alae
# and nose tip) and sebaceous density.
ZONE_CLINICAL_WEIGHTS: Dict[str, float] = {
    "nose_ala_left":      1.5,
    "nose_ala_right":     1.5,
    "nose_tip":           1.3,
    "nose_dorsum":        1.0,
    "cheek_medial_left":  1.2,
    "cheek_medial_right": 1.2,
    "cheek_lateral_left": 0.8,
    "cheek_lateral_right":0.8,
    "forehead":           1.0,
    "glabella":           0.6,
    "chin":               0.7,
    "perioral":           0.5,
}


# Published per-zone pore density plausibility ranges (pores per cm²).
# Used for Stage 9 literature-plausibility sanity check.
# Values are adult population ranges from Dissanayake 2019 and related
# pore metrology papers; broad to accept Fitzpatrick variation.
ZONE_DENSITY_PLAUSIBILITY: Dict[str, Tuple[float, float]] = {
    "nose_ala_left":      (60.0, 250.0),
    "nose_ala_right":     (60.0, 250.0),
    "nose_tip":           (60.0, 220.0),
    "nose_dorsum":        (40.0, 180.0),
    "cheek_medial_left":  (20.0, 140.0),
    "cheek_medial_right": (20.0, 140.0),
    "cheek_lateral_left": (10.0, 100.0),
    "cheek_lateral_right":(10.0, 100.0),
    "forehead":           (30.0, 180.0),
    "glabella":           (20.0, 150.0),
    "chin":               (15.0, 120.0),
    "perioral":           (10.0, 100.0),
}


# Ordered zone list (for iteration, UI, JSON).
ALL_ZONES: List[str] = list(ZONE_LANDMARKS.keys())


# =============================================================================
# Coverage thresholds
# =============================================================================
# Minimum fraction of nominal zone area that must survive skin-masking
# for the zone to be considered measurable.
ZONE_MIN_COVERAGE_RATIO = 0.40


# =============================================================================
# Foundation model checkpoints (all pretrained, loaded via HuggingFace/URL)
# =============================================================================
# Version-pinned model identifiers. Change here if upgrading.
MODEL_IDS = {
    "face_parser_facexformer": "kartiknarayan/FaceXFormer",          # hf hub
    "face_parser_bisenet_onnx_url": (
        "https://github.com/yakhyo/face-parsing/releases/download/v0.0.1/"
        "resnet18.onnx"
    ),
    "dinov2": "facebook/dinov2-small",                                # hf hub
    "sam2": "facebook/sam2-hiera-base-plus",                          # hf hub
    "depth_anything_v2": "depth-anything/Depth-Anything-V2-Small-hf", # hf hub
    "mediapipe_facelandmarker_url": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
}
