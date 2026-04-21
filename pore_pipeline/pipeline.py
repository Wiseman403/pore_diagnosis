"""
Main pipeline orchestrator.

Entry point:
    result = run_pipeline(image_path, models=PipelineModels(...))

Wires all stages 1–10, produces PipelineResult (per-pore records, per-zone
aggregates, whole-face composite, JSON, warnings, timings).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from .candidates import detect_pore_candidates
from .config import PIPELINE_VERSION, SCHEMA_VERSION
from .face import FaceLandmarkDetector
from .illumination import normalize_illumination
from .mapping import AnchorCalibration
from .metrology import (
    PoreRecord,
    WholeFace,
    ZoneAggregate,
    aggregate_zones,
    build_pore_records,
    whole_face_composite,
)
from .parsing import BiSeNetONNXParser, build_skin_mask
from .quality import QualityReport, capture_quality_gate
from .suppression import (
    AcceptedPore,
    DepthAnythingV2Encoder,
    DinoV2Encoder,
    Sam2Prompter,
    suppress_false_positives,
)
from .utils import load_image_bgr, sha256_of_file
from .zones import build_zones


@dataclass
class PipelineModels:
    """Bag of optional foundation-model instances.

    None entries mean that signal is skipped; the pipeline runs with
    degraded accuracy but still produces valid outputs with warnings.
    """
    face_detector: Optional[FaceLandmarkDetector] = None
    face_parser: Optional[BiSeNetONNXParser] = None
    dino: Optional[DinoV2Encoder] = None
    sam: Optional[Sam2Prompter] = None
    depth: Optional[DepthAnythingV2Encoder] = None
    flament_calib: Optional[AnchorCalibration] = None
    ibsa_calib: Optional[AnchorCalibration] = None


@dataclass
class PipelineResult:
    quality: QualityReport
    fitzpatrick: str
    skin_backend: str
    records: List[PoreRecord]
    rejected: List[AcceptedPore]
    zone_aggs: Dict[str, ZoneAggregate]
    whole_face: Optional[WholeFace]
    warnings: List[str]
    timings_ms: Dict[str, float]
    json_output: Dict
    debug: Dict = field(default_factory=dict)


def _time() -> float:
    return time.time() * 1000.0


def run_pipeline(
    image_path: str,
    models: PipelineModels,
    output_json_path: Optional[str] = None,
    debug_keep_intermediates: bool = False,
    max_candidates_to_sam: int = 2000,
) -> PipelineResult:
    """Run the full pipeline end-to-end.

    Args:
        image_path: path to a single RGB smartphone selfie.
        models: PipelineModels bag — any None entry = signal skipped.
        output_json_path: if provided, write JSON output to this path.
        debug_keep_intermediates: if True, keep intermediate numpy arrays
            in result.debug for dashboard rendering.
        max_candidates_to_sam: SAM-2 prompt budget per image.

    Returns:
        PipelineResult.
    """
    timings: Dict[str, float] = {}
    warnings: List[str] = []
    debug: Dict = {}

    # -------------------------------------------------------------------
    # Load image
    # -------------------------------------------------------------------
    t0 = _time()
    bgr = load_image_bgr(image_path)
    img_hash = sha256_of_file(image_path)
    h, w = bgr.shape[:2]
    timings["load_ms"] = _time() - t0

    # -------------------------------------------------------------------
    # Stage 2: face detection + landmarks
    # -------------------------------------------------------------------
    t0 = _time()
    if models.face_detector is None:
        raise RuntimeError("PipelineModels.face_detector is required")
    face = models.face_detector.detect(bgr)
    timings["face_ms"] = _time() - t0
    if face is None:
        quality = QualityReport(passed=False, reasons=["no_face_detected"])
        return PipelineResult(
            quality=quality, fitzpatrick="unknown", skin_backend="none",
            records=[], rejected=[], zone_aggs={}, whole_face=None,
            warnings=["no face detected"], timings_ms=timings,
            json_output=_build_json_minimum(image_path, img_hash, quality, bgr.shape),
        )

    # -------------------------------------------------------------------
    # Stage 1: capture quality gate (run AFTER face detection so we have pose/IOD)
    # -------------------------------------------------------------------
    t0 = _time()
    quality = capture_quality_gate(bgr, face, skin_mask=None, fitzpatrick_prior="III")
    timings["quality_ms"] = _time() - t0
    if not quality.passed:
        # Return early with a complete-structure but empty-metrics JSON.
        return PipelineResult(
            quality=quality, fitzpatrick="unknown", skin_backend="none",
            records=[], rejected=[], zone_aggs={}, whole_face=None,
            warnings=[f"quality gate failed: {', '.join(quality.reasons)}"],
            timings_ms=timings,
            json_output=_build_json(
                image_path=image_path, img_hash=img_hash, bgr_shape=bgr.shape,
                quality=quality, fitzpatrick="unknown",
                skin_backend="none", zone_aggs={}, whole_face=None,
                records=[], warnings=[], models=models,
            ),
        )

    # -------------------------------------------------------------------
    # Stage 3: skin isolation
    # -------------------------------------------------------------------
    t0 = _time()
    skin = build_skin_mask(bgr, face.landmarks_px, parser=models.face_parser,
                            hair_edge_band_px=3)
    timings["skin_ms"] = _time() - t0

    # -------------------------------------------------------------------
    # Stage 4: illumination normalization + phototype
    # -------------------------------------------------------------------
    t0 = _time()
    illum = normalize_illumination(bgr, skin.skin_mask)
    timings["illum_ms"] = _time() - t0
    fitzpatrick = illum.fitzpatrick
    if illum.chromatic_imbalance_flag:
        warnings.append("chromatic_imbalance_detected")

    # -------------------------------------------------------------------
    # Stage 5: zone extraction
    # -------------------------------------------------------------------
    t0 = _time()
    zone_set = build_zones((h, w), face.landmarks_px, skin.skin_mask)
    timings["zones_ms"] = _time() - t0

    def zone_of(x: int, y: int) -> Optional[str]:
        for zid, z in zone_set.zones.items():
            if 0 <= y < z.measurable_mask.shape[0] and 0 <= x < z.measurable_mask.shape[1]:
                if z.measurable_mask[y, x]:
                    return zid
        return None

    # -------------------------------------------------------------------
    # Stage 6: classical candidate detection
    # -------------------------------------------------------------------
    t0 = _time()
    candidates = detect_pore_candidates(
        L_normalized=illum.L_normalized,
        lab=illum.lab,
        skin_mask=skin.skin_mask,
        zone_of=zone_of,
        scale=face.scale,
        fitzpatrick=fitzpatrick,
    )
    timings["candidates_ms"] = _time() - t0

    # -------------------------------------------------------------------
    # Stage 7: zero-shot false-positive suppression
    # -------------------------------------------------------------------
    t0 = _time()
    accepted, rejected, supp_warnings = suppress_false_positives(
        candidates=candidates,
        bgr=illum.bgr_color_constant,
        scale=face.scale,
        fitzpatrick=fitzpatrick,
        dino=models.dino,
        sam=models.sam,
        depth_model=models.depth,
        max_candidates_to_sam=max_candidates_to_sam,
    )
    timings["suppression_ms"] = _time() - t0
    warnings.extend(supp_warnings)

    # -------------------------------------------------------------------
    # Stage 8: per-pore records
    # -------------------------------------------------------------------
    t0 = _time()
    records = build_pore_records(
        accepted=accepted,
        bgr=illum.bgr_color_constant,
        scale=face.scale,
        depth_available=(models.depth is not None),
    )
    timings["metrology_ms"] = _time() - t0

    # -------------------------------------------------------------------
    # Stage 9: per-zone aggregation
    # -------------------------------------------------------------------
    t0 = _time()
    zone_aggs = aggregate_zones(records, zone_set, face.scale)
    timings["aggregate_ms"] = _time() - t0

    # Apply anchor-based Flament/IBSA calibration if available.
    if models.flament_calib is not None:
        for zid, agg in zone_aggs.items():
            agg.flament_score = models.flament_calib.map_grade(agg.visibility_mean)
            agg.flament_score_plausibility = (
                max(0.0, agg.flament_score - 0.5),
                min(9.0, agg.flament_score + 0.5),
            )

    # -------------------------------------------------------------------
    # Stage 10: whole-face composite
    # -------------------------------------------------------------------
    t0 = _time()
    whole_face_out = whole_face_composite(zone_aggs)
    timings["whole_face_ms"] = _time() - t0

    if models.ibsa_calib is not None:
        whole_face_out.ibsa_pore_grade_1_5 = int(round(
            models.ibsa_calib.map_grade(whole_face_out.composite_score_0_100)
        ))

    # -------------------------------------------------------------------
    # Build JSON output
    # -------------------------------------------------------------------
    json_out = _build_json(
        image_path=image_path, img_hash=img_hash, bgr_shape=bgr.shape,
        quality=quality, fitzpatrick=fitzpatrick,
        skin_backend=skin.parser_backend,
        zone_aggs=zone_aggs, whole_face=whole_face_out,
        records=records, warnings=warnings, models=models,
        illum=illum, face=face,
    )
    if output_json_path:
        os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
        with open(output_json_path, "w") as f:
            json.dump(json_out, f, indent=2)

    if debug_keep_intermediates:
        debug = {
            "skin": skin,
            "illum": illum,
            "zones": zone_set,
            "face": face,
            "candidates_raw": candidates,
            "accepted": accepted,
        }

    return PipelineResult(
        quality=quality, fitzpatrick=fitzpatrick, skin_backend=skin.parser_backend,
        records=records, rejected=rejected,
        zone_aggs=zone_aggs, whole_face=whole_face_out,
        warnings=warnings, timings_ms=timings,
        json_output=json_out, debug=debug,
    )


# =============================================================================
# JSON schema builders
# =============================================================================
def _build_json_minimum(image_path: str, img_hash: str,
                         quality: QualityReport, bgr_shape) -> Dict:
    h, w = bgr_shape[:2]
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "image_metadata": {
            "source_path": image_path,
            "source_hash_sha256": img_hash,
            "width_px": int(w), "height_px": int(h),
        },
        "capture_quality": {
            "pass": quality.passed,
            "reasons": quality.reasons,
            "warnings": quality.warnings,
        },
        "calibration_basis_global": "anchor_interpolation_no_cohort",
        "limitations": ["quality gate failed, no downstream measurement"],
    }


def _build_json(image_path, img_hash, bgr_shape, quality, fitzpatrick,
                 skin_backend, zone_aggs, whole_face, records, warnings,
                 models, illum=None, face=None) -> Dict:
    h, w = bgr_shape[:2]
    zones_json = []
    for zid, agg in zone_aggs.items():
        zones_json.append({
            "zone_id": zid,
            "measurable": agg.measurable,
            "count": agg.count,
            "area_cm2": round(agg.area_cm2, 4),
            "density_per_cm2": round(agg.density_per_cm2, 2),
            "mean_diameter_mm": round(agg.mean_diameter_mm, 4),
            "median_diameter_mm": round(agg.median_diameter_mm, 4),
            "p90_diameter_mm": round(agg.p90_diameter_mm, 4),
            "area_fraction": round(agg.area_fraction, 4),
            "visibility_mean": round(agg.visibility_mean, 2),
            "visibility_p90": round(agg.visibility_p90, 2),
            "flament_score": round(agg.flament_score, 2),
            "flament_score_plausibility": [
                round(agg.flament_score_plausibility[0], 2),
                round(agg.flament_score_plausibility[1], 2),
            ],
            "coverage_ratio": round(agg.coverage_ratio, 3),
            "density_plausible": agg.density_plausible,
            "flags": agg.flags,
        })

    wf = None
    if whole_face is not None:
        wf = {
            "composite_score_0_100": round(whole_face.composite_score_0_100, 2),
            "composite_score_plausibility_interval": [
                round(whole_face.composite_score_plausibility_interval[0], 2),
                round(whole_face.composite_score_plausibility_interval[1], 2),
            ],
            "ibsa_pore_grade_1_5": whole_face.ibsa_pore_grade_1_5,
            "dominant_zone_id": whole_face.dominant_zone_id,
            "asymmetry_index": round(whole_face.asymmetry_index, 2),
            "all_zones_measurable": whole_face.all_zones_measurable,
            "measurable_zones": whole_face.measurable_zones,
        }

    limitations: List[str] = []
    if fitzpatrick in ("V", "VI"):
        limitations.append("fitzpatrick_V_VI_flament_extrapolated")
    if models.dino is None:
        limitations.append("dino_signal_unavailable")
    if models.sam is None:
        limitations.append("sam2_signal_unavailable_metrology_from_classical_fit")
    if models.depth is None:
        limitations.append("depth_signal_unavailable_scar_discrimination_degraded")
    if models.flament_calib is None:
        limitations.append("flament_mapping_heuristic_no_anchors")

    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "calibration_basis_global": (
            "anchor_interpolation_no_cohort"
            if models.flament_calib is not None else "heuristic_no_anchors"
        ),
        "image_metadata": {
            "source_path": image_path,
            "source_hash_sha256": img_hash,
            "width_px": int(w), "height_px": int(h),
        },
        "capture_quality": {
            "pass": quality.passed,
            "reasons": quality.reasons,
            "warnings": quality.warnings,
            "blur_var_laplacian": round(quality.blur_var_laplacian, 2),
            "clipped_fraction": round(quality.clipped_fraction, 4),
            "l_iqr": round(quality.l_iqr, 2),
            "pose_deg": {"yaw": round(quality.pose_deg[0], 2),
                          "pitch": round(quality.pose_deg[1], 2),
                          "roll": round(quality.pose_deg[2], 2)},
            "ppmm": round(quality.ppmm, 3),
            "capture_distance_mm_estimate": round(quality.capture_distance_mm, 1),
            "texture_energy_rms": round(quality.texture_energy_rms, 3),
        },
        "phototype": {
            "fitzpatrick_estimate": fitzpatrick,
            "ita_deg_mean": round(illum.ita_deg_mean, 2) if illum else None,
            "ita_deg_std": round(illum.ita_deg_std, 2) if illum else None,
            "extrapolated_calibration": fitzpatrick in ("V", "VI"),
            "chromatic_imbalance_flag": illum.chromatic_imbalance_flag if illum else None,
        },
        "skin_isolation": {"parser_backend": skin_backend},
        "zones": zones_json,
        "whole_face": wf,
        "pore_count_total": len(records),
        "rejected_count_total": None,   # filled in by caller if desired
        "model_versions": {
            "mediapipe_facelandmarker": "v2 (Tasks API)",
            "face_parser": "bisenet_onnx_v1" if models.face_parser else "landmarks_only",
            "dinov2": models.dino.model_id if models.dino else None,
            "sam2": models.sam.model_id if models.sam else None,
            "depth_anything_v2": models.depth.model_id if models.depth else None,
            "flament_mapper": (
                f"anchor_n={models.flament_calib.n_anchors}"
                if models.flament_calib else "heuristic"
            ),
            "ibsa_mapper": (
                f"anchor_n={models.ibsa_calib.n_anchors}"
                if models.ibsa_calib else "heuristic"
            ),
        },
        "warnings": warnings,
        "limitations": limitations,
    }
