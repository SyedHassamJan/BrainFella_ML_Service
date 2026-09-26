"""Facial signal extraction using PRETRAINED models only.

Nothing in this module trains, fine-tunes or saves a model. Two pretrained
components are used for inference:

  * MediaPipe Face Mesh (Google) - face detection, 468 landmarks + iris.
  * DeepFace's pretrained facial-expression CNN - emotion label + scores.

The outputs are raw signals for the screening pipeline's rule-based fusion
layer. They are not a diagnosis.
"""

import base64
import binascii
import threading
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace

MAX_IMAGE_BYTES = 5 * 1024 * 1024  # decoded size cap
MAX_B64_CHARS = MAX_IMAGE_BYTES * 4 // 3 + 16  # reject before decoding
MAX_DIMENSION = 1024  # downscale larger images for speed

MODELS = {
    "emotion": "DeepFace facial-expression CNN (pretrained, inference only)",
    "landmarks": "MediaPipe Face Mesh (pretrained, inference only)",
}
NOTE = "Signals from pretrained models for a screening result; not a diagnosis."


class AnalysisError(Exception):
    """Expected failure that maps to a `{ "error": code }` response."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


# --------------------------------------------------------------------------
# Model lifecycle: loaded once at startup, reused per request.
# --------------------------------------------------------------------------
_face_mesh = None
# MediaPipe's FaceMesh graph and the Keras model are not safe to call from
# several threads at once, so inference is serialized.
_lock = threading.Lock()


def load_models() -> None:
    global _face_mesh
    _face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,  # adds the iris landmarks (468-477)
        min_detection_confidence=0.5,
    )
    # Warm DeepFace's emotion model (downloads the pretrained weights on first run).
    DeepFace.analyze(
        np.zeros((224, 224, 3), dtype=np.uint8),
        actions=["emotion"],
        detector_backend="skip",
        enforce_detection=False,
        silent=True,
    )


def models_loaded() -> bool:
    return _face_mesh is not None


# --------------------------------------------------------------------------
# Image decoding
# --------------------------------------------------------------------------
def decode_image(b64: str) -> np.ndarray:
    if "," in b64[:100] and b64.lstrip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    b64 = b64.strip()
    if len(b64) > MAX_B64_CHARS:
        raise AnalysisError("image_too_large")
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise AnalysisError("invalid_image")
    if len(raw) > MAX_IMAGE_BYTES:
        raise AnalysisError("image_too_large")
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise AnalysisError("invalid_image")
    h, w = img.shape[:2]
    scale = MAX_DIMENSION / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


# --------------------------------------------------------------------------
# Head pose (from Face Mesh's own 3D landmarks)
# --------------------------------------------------------------------------
# Face Mesh predicts a 3D position for every landmark (x, y in image pixels; z
# is depth on roughly the same scale as x, smaller = closer to the camera). We
# build the face's own axes from three of them and read the rotation off:
#   x axis: image-left outer eye corner (33) -> image-right outer eye corner (263)
#   y axis: chin (152) -> midpoint of the eye corners, made orthogonal to x
#   z axis: x cross y (points out of the face, toward the camera when frontal)
_POSE_IDS = (33, 263, 152)  # left eye corner, right eye corner, chin

# Generic anatomical proportions (mm; x right, y up, z toward camera) for the
# same three points. Used ONLY to cancel the small built-in tilt between the
# chin->eye axis and true vertical, so a frontal face reads as (0, 0, 0). This
# is a fixed constant, not fitted to any images.
_REF_LEFT_EYE = np.array([-43.3, 32.7, -26.0])
_REF_RIGHT_EYE = np.array([43.3, 32.7, -26.0])
_REF_CHIN = np.array([0.0, -63.6, -12.5])


def _face_axes(left_eye: np.ndarray, right_eye: np.ndarray, chin: np.ndarray) -> np.ndarray:
    """Rotation matrix whose columns are the face's x/y/z axes (y-up, z-toward-camera frame)."""
    ex = right_eye - left_eye
    ex = ex / np.linalg.norm(ex)
    up = (left_eye + right_eye) / 2 - chin
    ey = up - np.dot(up, ex) * ex
    ey = ey / np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    return np.column_stack([ex, ey, ez])


_R_REF = _face_axes(_REF_LEFT_EYE, _REF_RIGHT_EYE, _REF_CHIN)


def estimate_head_pose(landmarks_3d: np.ndarray) -> dict:
    """landmarks_3d: (N, 3) Face Mesh landmarks as (x_px, y_px, z_px), image y pointing down.

    Sign convention (right-handed rotation of the face about its own axes):
      yaw   > 0  nose points toward the image's right (viewer's right)
      pitch > 0  nose points down
      roll  > 0  head tilted counter-clockwise as seen by the camera
    """
    # Face Mesh -> y-up, z-toward-camera frame.
    pts = landmarks_3d[list(_POSE_IDS)] * np.array([1.0, -1.0, -1.0])
    if np.linalg.norm(pts[1] - pts[0]) < 1e-6:
        raise AnalysisError("no_face_detected")
    rotation = _face_axes(*pts) @ _R_REF.T
    angles = cv2.RQDecomp3x3(rotation)[0]  # degrees: (x=pitch, y=yaw, z=roll)
    return {
        "yaw": round(float(angles[1]), 1),
        "pitch": round(float(angles[0]), 1),
        "roll": round(float(angles[2]), 1),
    }


# --------------------------------------------------------------------------
# Eye contact heuristic
# --------------------------------------------------------------------------
# Eye corners (outer, inner) and iris centre per eye, image-left eye first.
_EYES = [(33, 133, 468), (263, 362, 473)]
_GAZE_CENTER_RANGE = (0.35, 0.65)  # iris position across the eye (0 outer .. 1 inner)
_MAX_POSE_DEG = 20.0


def compute_eye_contact(landmarks_px: np.ndarray, head_pose: dict) -> bool:
    """True when the head faces the camera and both irises sit near eye centre.

    A simple geometric heuristic on pretrained Face Mesh landmarks - not a
    learned gaze model. Thresholds are deliberately coarse.
    """
    if abs(head_pose["yaw"]) > _MAX_POSE_DEG or abs(head_pose["pitch"]) > _MAX_POSE_DEG:
        return False
    for outer, inner, iris in _EYES:
        span = landmarks_px[inner][0] - landmarks_px[outer][0]
        if abs(span) < 1e-6:
            return False
        ratio = (landmarks_px[iris][0] - landmarks_px[outer][0]) / span
        if not (_GAZE_CENTER_RANGE[0] <= ratio <= _GAZE_CENTER_RANGE[1]):
            return False
    return True


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def _face_crop(img: np.ndarray, landmarks_px: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0 = landmarks_px[:468, :2].min(axis=0)
    x1, y1 = landmarks_px[:468, :2].max(axis=0)
    pad_x, pad_y = (x1 - x0) * 0.1, (y1 - y0) * 0.1
    x0, x1 = int(max(0, x0 - pad_x)), int(min(w, x1 + pad_x))
    y0, y1 = int(max(0, y0 - pad_y)), int(min(h, y1 + pad_y))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        raise AnalysisError("no_face_detected")
    return crop


def analyze_face(b64_image: str, decoded: Optional[np.ndarray] = None) -> dict:
    img = decoded if decoded is not None else decode_image(b64_image)
    h, w = img.shape[:2]

    with _lock:
        result = _face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            raise AnalysisError("no_face_detected")
        lm = result.multi_face_landmarks[0].landmark
        # z is on roughly the same scale as x, so it scales with the width.
        landmarks_px = np.array([(p.x * w, p.y * h, p.z * w) for p in lm], dtype=np.float64)

        head_pose = estimate_head_pose(landmarks_px)
        eye_contact = compute_eye_contact(landmarks_px, head_pose)

        # Reuse Face Mesh's detection: crop the face and skip DeepFace's own detector.
        analysis = DeepFace.analyze(
            _face_crop(img, landmarks_px),
            actions=["emotion"],
            detector_backend="skip",
            enforce_detection=False,
            silent=True,
        )

    first = analysis[0] if isinstance(analysis, list) else analysis
    scores = {k: round(float(v) / 100.0, 4) for k, v in first["emotion"].items()}
    return {
        "emotion": first["dominant_emotion"],
        "emotionScores": scores,
        "eyeContact": eye_contact,
        "blinkRate": None,  # single frame: not measurable
        "headPose": head_pose,
        "models": MODELS,
        "note": NOTE,
    }
