"""
pore_pipeline — Clinical-grade facial pore analysis from a single RGB selfie.

Public API:
    from pore_pipeline import run_pipeline, PipelineModels
    from pore_pipeline import (
        FaceLandmarkDetector, BiSeNetONNXParser,
        DinoV2Encoder, Sam2Prompter, DepthAnythingV2Encoder,
    )
    from pore_pipeline import render_dashboard
"""
from .config import (
    ALL_ZONES,
    PIPELINE_VERSION,
    SCHEMA_VERSION,
    ZONE_LANDMARKS,
    ita_to_fitzpatrick,
)
from .face import FaceLandmarkDetector, FaceLandmarks
from .illumination import IlluminationResult, normalize_illumination
from .mapping import (
    Anchor,
    AnchorCalibration,
    AnchorSet,
    calibrate_flament_from_anchors,
    calibrate_ibsa_from_anchors,
    fit_monotone_piecewise_linear,
)
from .metrology import (
    PoreRecord,
    WholeFace,
    ZoneAggregate,
    aggregate_zones,
    build_pore_records,
    whole_face_composite,
)
from .parsing import BiSeNetONNXParser, SkinMaskResult, build_skin_mask
from .pipeline import PipelineModels, PipelineResult, run_pipeline
from .quality import QualityReport, capture_quality_gate
from .suppression import (
    AcceptedPore,
    DepthAnythingV2Encoder,
    DinoV2Encoder,
    Sam2Prompter,
    suppress_false_positives,
)
from .utils import ScaleCalibration, calibrate_from_iod, load_image_bgr
from .visualization import render_dashboard, render_per_pore_overlay
from .zones import Zone, ZoneSet, build_zones

__all__ = [
    "run_pipeline", "PipelineModels", "PipelineResult",
    "FaceLandmarkDetector", "FaceLandmarks",
    "BiSeNetONNXParser", "SkinMaskResult", "build_skin_mask",
    "DinoV2Encoder", "Sam2Prompter", "DepthAnythingV2Encoder",
    "normalize_illumination", "IlluminationResult",
    "build_zones", "ZoneSet", "Zone",
    "capture_quality_gate", "QualityReport",
    "suppress_false_positives", "AcceptedPore",
    "build_pore_records", "aggregate_zones", "whole_face_composite",
    "PoreRecord", "WholeFace", "ZoneAggregate",
    "Anchor", "AnchorSet", "AnchorCalibration",
    "calibrate_flament_from_anchors", "calibrate_ibsa_from_anchors",
    "fit_monotone_piecewise_linear",
    "render_dashboard", "render_per_pore_overlay",
    "ScaleCalibration", "calibrate_from_iod", "load_image_bgr",
    "ita_to_fitzpatrick",
    "SCHEMA_VERSION", "PIPELINE_VERSION", "ALL_ZONES", "ZONE_LANDMARKS",
]
