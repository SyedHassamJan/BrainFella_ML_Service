"""API contract tests with the models stubbed out (no weights needed)."""

import pytest
from fastapi.testclient import TestClient

from app import face_analysis, voice_analysis
from app.main import app

KEY = "test-key"
HEADERS = {"X-Internal-Key": KEY}
BODY = {"image": "AAAA"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ML_SERVICE_KEY", KEY)
    monkeypatch.setattr(face_analysis, "load_models", lambda: None)
    monkeypatch.setattr(voice_analysis, "load_model", lambda: None)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_health_is_open(client):
    assert client.get("/health").status_code == 200


def test_requires_key(client):
    assert client.post("/analyze-face", json=BODY).status_code == 401
    assert client.post("/analyze-face", json=BODY, headers={"X-Internal-Key": "nope"}).status_code == 401


def test_fails_closed_when_key_not_configured(client, monkeypatch):
    monkeypatch.delenv("ML_SERVICE_KEY")
    assert client.post("/analyze-face", json=BODY, headers=HEADERS).status_code == 503


def test_expected_failure_returns_error_body(client, monkeypatch):
    def boom(_):
        raise face_analysis.AnalysisError("no_face_detected")

    monkeypatch.setattr(face_analysis, "analyze_face", boom)
    r = client.post("/analyze-face", json=BODY, headers=HEADERS)
    assert r.status_code == 200
    assert r.json() == {"error": "no_face_detected"}


def test_unexpected_exception_never_leaks(client, monkeypatch):
    def boom(_):
        raise RuntimeError("secret internals")

    monkeypatch.setattr(face_analysis, "analyze_face", boom)
    r = client.post("/analyze-face", json=BODY, headers=HEADERS)
    assert r.status_code == 500
    assert r.json() == {"error": "analysis_failed"}


def test_success_shape(client, monkeypatch):
    ok = {
        "emotion": "neutral",
        "emotionScores": {"neutral": 0.9, "sad": 0.1},
        "eyeContact": True,
        "blinkRate": None,
        "headPose": {"yaw": 1.0, "pitch": 2.0, "roll": 3.0},
        "models": face_analysis.MODELS,
        "note": face_analysis.NOTE,
    }
    monkeypatch.setattr(face_analysis, "analyze_face", lambda _: ok)
    r = client.post("/analyze-face", json=BODY, headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["emotion"] == "neutral"
    assert r.json()["blinkRate"] is None


# ---- /analyze-voice ----
VOICE_BODY = {"audio": "AAAA"}


def test_voice_requires_key(client):
    assert client.post("/analyze-voice", json=VOICE_BODY).status_code == 401


def test_voice_expected_failure_returns_error_body(client, monkeypatch):
    def boom(*_):
        raise voice_analysis.VoiceError("no_speech_detected")

    monkeypatch.setattr(voice_analysis, "analyze_voice", boom)
    r = client.post("/analyze-voice", json=VOICE_BODY, headers=HEADERS)
    assert r.status_code == 200
    assert r.json() == {"error": "no_speech_detected"}


def test_voice_unexpected_exception_never_leaks(client, monkeypatch):
    def boom(*_):
        raise RuntimeError("secret transcript text")

    monkeypatch.setattr(voice_analysis, "analyze_voice", boom)
    r = client.post("/analyze-voice", json=VOICE_BODY, headers=HEADERS)
    assert r.status_code == 500
    assert r.json() == {"error": "analysis_failed"}
    assert "secret" not in r.text


def test_voice_rejects_bad_language_code(client):
    r = client.post("/analyze-voice", json={"audio": "AAAA", "language": "English!"}, headers=HEADERS)
    assert r.status_code == 422


def test_voice_passes_language_and_returns_success_shape(client, monkeypatch):
    seen = {}

    def fake(audio, language):
        seen["language"] = language
        return {
            "transcript": "hello there",
            "duration": 2.5,
            "language": "en",
            "languageProbability": 0.99,
            "speakingRate": 96.0,
            "pauseCount": 0,
            "models": voice_analysis.model_description(),
            "note": voice_analysis.NOTE,
        }

    monkeypatch.setattr(voice_analysis, "analyze_voice", fake)
    r = client.post("/analyze-voice", json={"audio": "AAAA", "language": "en"}, headers=HEADERS)
    assert r.status_code == 200 and seen["language"] == "en"
    assert r.json()["transcript"] == "hello there"
