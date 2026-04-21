"""
Stage 2: face detection + 478-landmark mesh.

Uses MediaPipe's FaceLandmarker (Tasks API, 2024+) which returns 478
landmarks including iris when refine_landmarks=True. Iris landmarks
(indices 468-477) give us the inter-ocular distance for mm calibration.

Falls back gracefully to the legacy mediapipe.solutions.face_mesh API
if the Tasks API is unavailable.
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import ADULT_IOD_MM, IOD_UNCERTAINTY_PCT, MODEL_IDS
from .utils import ScaleCalibration, calibrate_from_iod


# MediaPipe iris landmark indices (when refine_landmarks=True, 478 total).
# Left eye iris center ≈ 468, right eye iris center ≈ 473.
LEFT_IRIS_CENTER_IDX = 468
RIGHT_IRIS_CENTER_IDX = 473


@dataclass
class FaceLandmarks:
    """Result of Stage 2: face detection + dense landmark mesh."""
    landmarks_px: np.ndarray             # (478, 3) in pixel coords, z = normalized depth
    landmarks_normalized: np.ndarray     # (478, 3) in normalized [0,1]
    bbox_xywh: Tuple[int, int, int, int] # bounding box from landmarks
    head_pose_deg: Tuple[float, float, float]  # yaw, pitch, roll
    iod_px: float
    scale: ScaleCalibration


def _download_facelandmarker_task(dest_path: str) -> None:
    """Download the face_landmarker.task file if missing (Colab use)."""
    if os.path.exists(dest_path):
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    url = MODEL_IDS["mediapipe_facelandmarker_url"]
    urllib.request.urlretrieve(url, dest_path)


class FaceLandmarkDetector:
    """Wrapper around MediaPipe FaceLandmarker with graceful fallbacks.

    Usage:
        det = FaceLandmarkDetector()
        result = det.detect(bgr_image)   # FaceLandmarks or None

    The detector tries the Tasks API first (mediapipe ≥ 0.10), and falls back
    to the legacy solutions API if unavailable.
    """

    def __init__(self, task_file_path: Optional[str] = None):
        self._task_file_path = task_file_path or "/root/.cache/mediapipe/face_landmarker.task"
        self._tasks_detector = None
        self._legacy_detector = None
        self._init_backend()

    def _init_backend(self) -> None:
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            _download_facelandmarker_task(self._task_file_path)
            base_options = mp_python.BaseOptions(model_asset_path=self._task_file_path)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=True,
                num_faces=1,
            )
            self._tasks_detector = mp_vision.FaceLandmarker.create_from_options(options)
            return
        except Exception as e:
            self._tasks_detector = None
            self._tasks_error = str(e)

        # Fallback: legacy solutions API (older mediapipe).
        try:
            import mediapipe as mp
            self._legacy_detector = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5,
            )
        except Exception as e:
            self._legacy_detector = None
            self._legacy_error = str(e)
            raise RuntimeError(
                f"Could not initialize MediaPipe FaceLandmarker. "
                f"Tasks API error: {getattr(self, '_tasks_error', 'n/a')}; "
                f"Legacy API error: {getattr(self, '_legacy_error', 'n/a')}"
            )

    def detect(self, bgr: np.ndarray) -> Optional[FaceLandmarks]:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self._tasks_detector is not None:
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._tasks_detector.detect(mp_image)
            if not result.face_landmarks:
                return None
            lms = result.face_landmarks[0]   # list of NormalizedLandmark
            landmarks_norm = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            transform = None
            if result.facial_transformation_matrixes:
                transform = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float32)
        else:
            result = self._legacy_detector.process(rgb)
            if not result.multi_face_landmarks:
                return None
            lms = result.multi_face_landmarks[0].landmark
            landmarks_norm = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            transform = None

        if landmarks_norm.shape[0] < 478:
            # Without refine_landmarks / iris, we only get 468. We still run,
            # but IOD comes from eye-corner landmarks with a larger uncertainty.
            pass

        landmarks_px = landmarks_norm.copy()
        landmarks_px[:, 0] *= w
        landmarks_px[:, 1] *= h
        # z kept in normalized units (unitless depth).

        bbox = _bbox_from_landmarks(landmarks_px[:, :2], w, h)
        iod_px = _compute_iod_px(landmarks_px)
        scale = calibrate_from_iod(iod_px, ADULT_IOD_MM, IOD_UNCERTAINTY_PCT)
        yaw, pitch, roll = _head_pose_from_transform(transform) if transform is not None \
            else _head_pose_from_landmarks(landmarks_px)

        return FaceLandmarks(
            landmarks_px=landmarks_px,
            landmarks_normalized=landmarks_norm,
            bbox_xywh=bbox,
            head_pose_deg=(yaw, pitch, roll),
            iod_px=iod_px,
            scale=scale,
        )


def _bbox_from_landmarks(landmarks_xy: np.ndarray, w: int, h: int) -> Tuple[int, int, int, int]:
    x_min = int(np.clip(landmarks_xy[:, 0].min(), 0, w - 1))
    x_max = int(np.clip(landmarks_xy[:, 0].max(), 0, w - 1))
    y_min = int(np.clip(landmarks_xy[:, 1].min(), 0, h - 1))
    y_max = int(np.clip(landmarks_xy[:, 1].max(), 0, h - 1))
    return (x_min, y_min, x_max - x_min, y_max - y_min)


def _compute_iod_px(landmarks_px: np.ndarray) -> float:
    """Inter-ocular distance in pixels.

    Preferred: iris centers (indices 468 and 473) — most accurate when
    refine_landmarks=True.
    Fallback: outer eye corners (33, 263) — looser.
    """
    n = landmarks_px.shape[0]
    if n >= 478:
        lp = landmarks_px[LEFT_IRIS_CENTER_IDX, :2]
        rp = landmarks_px[RIGHT_IRIS_CENTER_IDX, :2]
    else:
        # Legacy fallback: outer eye corners.
        lp = landmarks_px[33, :2]    # left-eye outer corner (subject's right)
        rp = landmarks_px[263, :2]   # right-eye outer corner (subject's left)
    return float(np.linalg.norm(lp - rp))


def _head_pose_from_transform(transform: np.ndarray) -> Tuple[float, float, float]:
    """Extract yaw/pitch/roll from MediaPipe's 4×4 transformation matrix."""
    R = transform[:3, :3]
    # ZYX Tait-Bryan; standard decomposition.
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    singular = sy < 1e-6
    if not singular:
        pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        yaw   = np.degrees(np.arctan2(-R[2, 0], sy))
        roll  = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        yaw   = np.degrees(np.arctan2(-R[2, 0], sy))
        roll  = 0.0
    return float(yaw), float(pitch), float(roll)


def _head_pose_from_landmarks(landmarks_px: np.ndarray) -> Tuple[float, float, float]:
    """Approximate head pose from landmarks via solvePnP (fallback path)."""
    # Canonical 3D model points (mm, approximate).
    model_points = np.array([
        [0.0,  0.0,    0.0],     # nose tip (MP idx 1)
        [0.0, -63.6,  -12.5],    # chin     (MP idx 152)
        [-43.3, 32.7, -26.0],    # left eye outer (MP idx 33)
        [43.3,  32.7, -26.0],    # right eye outer (MP idx 263)
        [-28.9,-28.9, -24.1],    # left mouth (MP idx 61)
        [28.9, -28.9, -24.1],    # right mouth (MP idx 291)
    ], dtype=np.float64)
    image_points = np.array([
        landmarks_px[1,   :2],
        landmarks_px[152, :2],
        landmarks_px[33,  :2],
        landmarks_px[263, :2],
        landmarks_px[61,  :2],
        landmarks_px[291, :2],
    ], dtype=np.float64)
    w = float(landmarks_px[:, 0].max() - landmarks_px[:, 0].min())
    focal = w
    cx = float(landmarks_px[:, 0].mean())
    cy = float(landmarks_px[:, 1].mean())
    cam = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1))
    ok, rvec, tvec = cv2.solvePnP(model_points, image_points, cam, dist,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    R, _ = cv2.Rodrigues(rvec)
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    pitch = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
    yaw   = float(np.degrees(np.arctan2(-R[2, 0], sy)))
    roll  = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return yaw, pitch, roll
