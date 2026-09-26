from typing import Optional

from pydantic import BaseModel, Field


class AnalyzeFaceRequest(BaseModel):
    # Base64-encoded image (JPEG/PNG/WebP). A `data:image/...;base64,` prefix is accepted.
    image: str


class HeadPose(BaseModel):
    yaw: float  # degrees, left/right turn
    pitch: float  # degrees, up/down tilt
    roll: float  # degrees, sideways tilt


class AnalyzeFaceResponse(BaseModel):
    emotion: str  # dominant label from DeepFace's pretrained emotion model
    emotionScores: dict[str, float]  # per-emotion confidence, 0-1
    eyeContact: bool  # heuristic from Face Mesh iris + head pose (see face_analysis.py)
    blinkRate: Optional[float] = None  # needs a multi-frame capture; always null for a single image
    headPose: HeadPose
    models: dict[str, str]  # which pretrained models produced these signals
    note: str


class ErrorResponse(BaseModel):
    error: str


class AnalyzeVoiceRequest(BaseModel):
    # Base64-encoded audio (m4a/aac, wav, mp3, webm... anything PyAV can decode).
    audio: str
    # Optional ISO 639-1 code to force the spoken language (e.g. "en"); omit to auto-detect.
    language: Optional[str] = Field(default=None, pattern=r"^[a-z]{2,3}$")


class AnalyzeVoiceResponse(BaseModel):
    transcript: str
    duration: float  # seconds
    language: str  # detected (or forced) ISO code; downstream text models are English-only
    languageProbability: float
    speakingRate: float  # words per minute
    pauseCount: int  # gaps >= 0.7 s between words
    models: dict[str, str]
    note: str
