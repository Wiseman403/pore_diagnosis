"""
End-to-end sanity test with a mock face detector and synthetic image.
Verifies that every stage wires correctly through run_pipeline().
Foundation models (DINO, SAM2, Depth) are all None — the pipeline should
run with warnings and produce a valid JSON.
"""
from __future__ import annotations

import json
import os
import sys
import numpy as np
import cv2

# Ensure local package import.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pore_pipeline import (
    PipelineModels, FaceLandmarkDetector, FaceLandmarks,
)
from pore_pipeline.pipeline import run_pipeline
from pore_pipeline.utils import calibrate_from_iod


# =============================================================================
# Build a synthetic 'face' image with realistic landmark positions
# =============================================================================
def make_synthetic_face(size=(1800, 1400)):
    h_out, w_out = size
    # Base skin tone with per-pixel noise (realistic texture).
    rng = np.random.default_rng(42)
    img = np.full((h_out, w_out, 3), (140, 170, 200), dtype=np.uint8)
    noise = rng.normal(0, 8, (h_out, w_out, 3))
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Soft shading gradient (light from top-left) — stronger gradient so L_IQR > 10.
    yy, xx = np.mgrid[0:h_out, 0:w_out]
    shade = 1.0 - 0.35 * (yy / h_out) - 0.15 * (xx / w_out)
    img = np.clip(img * shade[..., None], 0, 255).astype(np.uint8)

    # Simulate pores on the T-zone and cheeks (realistic density).
    pore_positions = []
    for _ in range(300):
        x = int(rng.normal(w_out/2, 140))
        y = int(rng.normal(h_out*0.35, 140))
        if 0 < x < w_out and 0 < y < h_out:
            cv2.circle(img, (x, y), int(rng.integers(3, 6)), (85, 105, 125), -1)
            pore_positions.append((x, y))
    for _ in range(150):
        side = rng.choice([-1, 1])
        x = int(w_out/2 + side * rng.normal(280, 60))
        y = int(rng.normal(h_out*0.55, 80))
        if 0 < x < w_out and 0 < y < h_out:
            cv2.circle(img, (x, y), int(rng.integers(3, 5)), (90, 110, 130), -1)
            pore_positions.append((x, y))

    return img, pore_positions


def make_synthetic_landmarks(img_shape, iod_px=300):
    """Construct a plausible 478-landmark set for a front-facing synthetic face.

    This does NOT try to be anatomically perfect — just provides landmarks
    within the image bounds that satisfy the pipeline's needs (iris indices,
    face oval, zone-defining indices).
    """
    h, w = img_shape[:2]
    cx, cy = w / 2, h * 0.5
    landmarks = np.zeros((478, 3), dtype=np.float32)

    # Helper: place landmark at relative (dx, dy) from center.
    def place(idx, dx, dy, dz=0.0):
        landmarks[idx] = (cx + dx, cy + dy, dz)

    # Face oval — an ellipse with aspect 0.7.
    face_w = iod_px * 2.8
    face_h = face_w * 1.3
    from pore_pipeline.parsing import FACE_OVAL_IDX
    for i, idx in enumerate(FACE_OVAL_IDX):
        theta = 2 * np.pi * i / len(FACE_OVAL_IDX) - np.pi/2
        place(idx, (face_w/2) * np.cos(theta), (face_h/2) * np.sin(theta))

    # Eyes: iris centers at ±iod_px/2 from midline.
    place(468, -iod_px/2, -face_h*0.08)    # left iris
    place(473,  iod_px/2, -face_h*0.08)    # right iris
    # Eye corners (needed as IOD fallback and pose).
    place(33, -iod_px*0.7, -face_h*0.08)
    place(263, iod_px*0.7, -face_h*0.08)
    # Upper/lower eye contour landmarks (some used for eye masking).
    for i, idx in enumerate([7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]):
        theta = 2 * np.pi * i / 16
        place(idx, -iod_px/2 + 20 * np.cos(theta), -face_h*0.08 + 10 * np.sin(theta))
    for i, idx in enumerate([362, 382, 381, 380, 374, 373, 390, 249, 466, 388, 387, 386, 385, 384, 398]):
        theta = 2 * np.pi * i / 16
        place(idx, iod_px/2 + 20 * np.cos(theta), -face_h*0.08 + 10 * np.sin(theta))

    # Eyebrows
    for i, idx in enumerate([70, 63, 105, 66, 107, 55, 65, 52, 53, 46]):
        place(idx, -iod_px*0.7 + i*8, -face_h*0.18)
    for i, idx in enumerate([336, 296, 334, 293, 300, 285, 295, 282, 283, 276]):
        place(idx, iod_px*0.7 - i*8, -face_h*0.18)

    # Nose
    place(1, 0, face_h*0.05)           # nose tip
    place(4, 0, face_h*0.04)
    place(6, 0, -face_h*0.04)          # nasion
    place(168, 0, -face_h*0.02)
    place(197, 0, 0.0)
    place(195, 0, face_h*0.02)
    place(5, 0, face_h*0.03)
    place(152, 0, face_h*0.45)         # chin
    # Nostrils (approximate)
    for i, idx in enumerate([64, 102, 49, 131, 134, 51, 45, 48]):
        place(idx, -20 + i*3, face_h*0.08)
    for i, idx in enumerate([294, 331, 279, 360, 363, 281, 275, 278]):
        place(idx, 20 - i*3, face_h*0.08)

    # Lips (outer contour)
    for i, idx in enumerate([61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]):
        theta = 2 * np.pi * i / 20
        place(idx, 60 * np.cos(theta), face_h*0.28 + 20 * np.sin(theta))

    # Fill remaining unset landmarks with centroid (safe default).
    unset_mask = np.all(landmarks == 0, axis=1)
    for idx in np.where(unset_mask)[0]:
        landmarks[idx] = (cx, cy, 0.0)

    return landmarks


class MockFaceDetector:
    """Mock face detector that returns hand-crafted landmarks."""
    def __init__(self, iod_px=300):
        self.iod_px = iod_px
    def detect(self, bgr):
        h, w = bgr.shape[:2]
        landmarks = make_synthetic_landmarks(bgr.shape, iod_px=self.iod_px)
        bbox_x = int(landmarks[:, 0].min()); bbox_y = int(landmarks[:, 1].min())
        bbox_w = int(landmarks[:, 0].max() - landmarks[:, 0].min())
        bbox_h = int(landmarks[:, 1].max() - landmarks[:, 1].min())
        return FaceLandmarks(
            landmarks_px=landmarks,
            landmarks_normalized=np.stack([landmarks[:,0]/w, landmarks[:,1]/h, landmarks[:,2]], axis=1),
            bbox_xywh=(bbox_x, bbox_y, bbox_w, bbox_h),
            head_pose_deg=(0.0, 0.0, 0.0),
            iod_px=float(self.iod_px),
            scale=calibrate_from_iod(float(self.iod_px)),
        )


if __name__ == "__main__":
    print("=== End-to-end synthetic mock test ===")
    img, truth = make_synthetic_face(size=(1800, 1400))
    path = "/tmp/synth_selfie.png"
    cv2.imwrite(path, img)
    print(f"Synthetic image: {img.shape}, {len(truth)} true pore positions, saved to {path}")

    models = PipelineModels(
        face_detector=MockFaceDetector(iod_px=700),
        face_parser=None,
        dino=None, sam=None, depth=None,
        flament_calib=None, ibsa_calib=None,
    )
    detected_face = models.face_detector.detect(img)
    print(f"ppmm = {detected_face.scale.ppmm:.2f}")

    result = run_pipeline(path, models, output_json_path="/tmp/synth_result.json",
                          max_candidates_to_sam=0)
    print(f"\n--- Quality gate ---")
    print(f"  passed: {result.quality.passed}")
    print(f"  reasons: {result.quality.reasons}")
    print(f"  warnings: {result.quality.warnings}")
    print(f"  ppmm: {result.quality.ppmm:.2f}, blur: {result.quality.blur_var_laplacian:.1f}")
    print(f"\n--- Results ---")
    print(f"  Fitzpatrick: {result.fitzpatrick}")
    print(f"  Skin backend: {result.skin_backend}")
    print(f"  Detected pores: {len(result.records)}")
    print(f"  Rejected: {len(result.rejected)}")
    print(f"  Warnings: {result.warnings}")
    print(f"  Timings (ms): {result.timings_ms}")
    if result.whole_face:
        print(f"  Composite: {result.whole_face.composite_score_0_100:.1f}/100")
        print(f"  IBSA: {result.whole_face.ibsa_pore_grade_1_5}/5")
        print(f"  Dominant: {result.whole_face.dominant_zone_id}")
    print(f"\n--- Top 3 zones by count ---")
    zs = sorted(result.zone_aggs.items(), key=lambda x: -x[1].count)
    for zid, agg in zs[:3]:
        print(f"  {zid}: n={agg.count}, density={agg.density_per_cm2:.1f}/cm², "
              f"mean_d={agg.mean_diameter_mm*1000:.0f}µm, "
              f"flament={agg.flament_score:.1f}, plausible={agg.density_plausible}")
    print(f"\nJSON keys: {list(result.json_output.keys())}")
    print(f"JSON written to /tmp/synth_result.json")
