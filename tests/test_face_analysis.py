"""Unit tests that need no model weights."""

import base64

import cv2
import numpy as np
import pytest

from app import face_analysis as fa


def _b64_png(shape=(64, 64, 3)) -> str:
    ok, buf = cv2.imencode(".png", np.full(shape, 128, dtype=np.uint8))
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


class TestDecodeImage:
    def test_decodes_plain_and_data_uri(self):
        b64 = _b64_png()
        assert fa.decode_image(b64).shape == (64, 64, 3)
        assert fa.decode_image("data:image/png;base64," + b64).shape == (64, 64, 3)

    def test_rejects_bad_base64(self):
        with pytest.raises(fa.AnalysisError) as e:
            fa.decode_image("not base64 !!!")
        assert e.value.code == "invalid_image"

    def test_rejects_non_image_bytes(self):
        with pytest.raises(fa.AnalysisError) as e:
            fa.decode_image(base64.b64encode(b"hello world").decode())
        assert e.value.code == "invalid_image"

    def test_rejects_oversized(self):
        with pytest.raises(fa.AnalysisError) as e:
            fa.decode_image("A" * (fa.MAX_B64_CHARS + 1))
        assert e.value.code == "image_too_large"

    def test_downscales_large_images(self):
        big = fa.decode_image(_b64_png((2000, 3000, 3)))
        assert max(big.shape[:2]) == fa.MAX_DIMENSION


# ---- head pose: build synthetic Face Mesh landmarks from a known rotation, recover it ----
def _rot(axis, deg):
    v = np.zeros(3)
    v["xyz".index(axis)] = np.radians(deg)
    return cv2.Rodrigues(v)[0]


def _mesh(object_rotation, offset=(0.0, 0.0, 600.0)):
    """(478, 3) landmarks in Face Mesh convention (image y down, smaller z = closer)."""
    pts = np.zeros((478, 3))
    for idx, ref in (
        (33, fa._REF_LEFT_EYE),
        (263, fa._REF_RIGHT_EYE),
        (152, fa._REF_CHIN),
    ):
        y_up = object_rotation @ ref + np.array(offset)
        pts[idx] = y_up * np.array([1.0, -1.0, -1.0])
    return pts


class TestHeadPose:
    def test_frontal_face_is_exactly_zero(self):
        pose = fa.estimate_head_pose(_mesh(np.eye(3)))
        assert all(abs(v) < 0.2 for v in pose.values()), pose

    @pytest.mark.parametrize("yaw", [-45, -30, -15, 15, 30, 45])
    def test_recovers_signed_yaw(self, yaw):
        # positive yaw = nose toward image right (see estimate_head_pose)
        pose = fa.estimate_head_pose(_mesh(_rot("y", yaw)))
        assert pose["yaw"] == pytest.approx(yaw, abs=1), pose
        assert abs(pose["pitch"]) < 1 and abs(pose["roll"]) < 1

    @pytest.mark.parametrize("pitch", [-25, -10, 10, 25])
    def test_recovers_signed_pitch(self, pitch):
        pose = fa.estimate_head_pose(_mesh(_rot("x", pitch)))
        assert pose["pitch"] == pytest.approx(pitch, abs=1), pose
        assert abs(pose["yaw"]) < 1 and abs(pose["roll"]) < 1

    @pytest.mark.parametrize("roll", [-20, 20])
    def test_recovers_signed_roll(self, roll):
        pose = fa.estimate_head_pose(_mesh(_rot("z", roll)))
        assert pose["roll"] == pytest.approx(roll, abs=1), pose

    def test_independent_of_position_in_frame(self):
        a = fa.estimate_head_pose(_mesh(_rot("y", 20), offset=(0, 0, 600)))
        b = fa.estimate_head_pose(_mesh(_rot("y", 20), offset=(150, -80, 900)))
        assert a == b


# ---- eye contact heuristic ----
def _landmarks(ratio_left=0.5, ratio_right=0.5):
    pts = np.zeros((478, 2))
    # image-left eye: outer 33 -> inner 133, iris 468
    pts[33], pts[133] = (100, 100), (140, 100)
    pts[468] = (100 + 40 * ratio_left, 100)
    # image-right eye: outer 263 -> inner 362, iris 473 (inner is to the image-left)
    pts[263], pts[362] = (260, 100), (220, 100)
    pts[473] = (260 - 40 * ratio_right, 100)
    return pts


class TestEyeContact:
    FRONTAL = {"yaw": 3.0, "pitch": -2.0, "roll": 1.0}

    def test_centered_irises_facing_camera(self):
        assert fa.compute_eye_contact(_landmarks(), self.FRONTAL) is True

    def test_iris_off_center(self):
        assert fa.compute_eye_contact(_landmarks(0.1, 0.5), self.FRONTAL) is False
        assert fa.compute_eye_contact(_landmarks(0.5, 0.9), self.FRONTAL) is False

    def test_head_turned_away(self):
        assert fa.compute_eye_contact(_landmarks(), {"yaw": 35.0, "pitch": 0.0, "roll": 0.0}) is False
        assert fa.compute_eye_contact(_landmarks(), {"yaw": 0.0, "pitch": -30.0, "roll": 0.0}) is False
