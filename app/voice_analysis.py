"""Speech-to-text and simple speech-timing signals using a PRETRAINED model.

Whisper (via faster-whisper / CTranslate2) is used for inference only —
nothing here trains, fine-tunes or saves a model. The transcript is returned
to the caller and is never logged or stored by this service. Emotion analysis
of the transcript is NOT done here; it belongs to the text-analysis path.
"""

import base64
import binascii
import io
import os
import threading
from typing import Optional

from faster_whisper import WhisperModel

MAX_AUDIO_BYTES = 10 * 1024 * 1024  # decoded size cap
MAX_B64_CHARS = MAX_AUDIO_BYTES * 4 // 3 + 16  # reject before decoding
MAX_DURATION_SECONDS = 120.0  # bounds CPU time per request
PAUSE_GAP_SECONDS = 0.7  # silence between two words that counts as a pause

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")  # multilingual, not "base.en"

NOTE = (
    "Transcript and timing signals from a pretrained speech model, for a screening "
    "result; not a diagnosis. Non-English speech is transcribed but not emotion-scored."
)


class VoiceError(Exception):
    """Expected failure that maps to a `{ "error": code }` response."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def model_description() -> dict:
    return {"speechToText": f"OpenAI Whisper '{WHISPER_MODEL}' via faster-whisper (pretrained, inference only)"}


# --------------------------------------------------------------------------
# Model lifecycle: loaded once at startup, reused per request.
# --------------------------------------------------------------------------
_model: Optional[WhisperModel] = None
_lock = threading.Lock()  # one transcription at a time


def load_model() -> None:
    global _model
    _model = WhisperModel(
        WHISPER_MODEL,
        device="cpu",
        compute_type="int8",
        download_root=os.environ.get("WHISPER_CACHE_DIR") or None,
    )


def model_loaded() -> bool:
    return _model is not None


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------
def decode_audio(b64: str) -> bytes:
    if b64.lstrip().startswith("data:") and "," in b64[:100]:
        b64 = b64.split(",", 1)[1]
    b64 = b64.strip()
    if len(b64) > MAX_B64_CHARS:
        raise VoiceError("audio_too_large")
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        raise VoiceError("invalid_audio")
    if len(raw) > MAX_AUDIO_BYTES:
        raise VoiceError("audio_too_large")
    if not raw:
        raise VoiceError("invalid_audio")
    return raw


# --------------------------------------------------------------------------
# Speech-timing signals (pure functions, no model needed)
# --------------------------------------------------------------------------
def compute_speech_metrics(words: list[tuple[float, float]], duration: float) -> dict:
    """words: (start, end) seconds per spoken word, in order.

    speakingRate: words per minute over the span from the first word's start to
    the last word's end (so leading/trailing silence doesn't dilute it); falls
    back to the whole clip when that span is under one second.
    pauseCount: gaps of at least PAUSE_GAP_SECONDS between consecutive words.
    """
    if not words:
        return {"speakingRate": 0.0, "pauseCount": 0}
    span = words[-1][1] - words[0][0]
    if span < 1.0:
        span = max(duration, 1.0)
    pauses = sum(
        1 for prev, nxt in zip(words, words[1:]) if nxt[0] - prev[1] >= PAUSE_GAP_SECONDS
    )
    return {"speakingRate": round(len(words) / span * 60.0, 1), "pauseCount": pauses}


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def analyze_voice(b64_audio: str, language: Optional[str] = None) -> dict:
    raw = decode_audio(b64_audio)

    with _lock:
        try:
            segments, info = _model.transcribe(
                io.BytesIO(raw),
                language=language,
                word_timestamps=True,
                vad_filter=True,  # drop silence/noise so it isn't hallucinated into text
                beam_size=1,
            )
        except Exception:  # PyAV raises assorted errors for undecodable audio
            raise VoiceError("invalid_audio")

        if info.duration > MAX_DURATION_SECONDS:
            raise VoiceError("audio_too_long")

        try:
            words = []
            texts = []
            for seg in segments:  # the generator does the actual decoding
                texts.append(seg.text.strip())
                words.extend((w.start, w.end) for w in (seg.words or []))
        except Exception:
            raise VoiceError("invalid_audio")

    transcript = " ".join(t for t in texts if t).strip()
    if not transcript or not words:
        raise VoiceError("no_speech_detected")

    return {
        "transcript": transcript,
        "duration": round(float(info.duration), 2),
        "language": info.language,
        "languageProbability": round(float(info.language_probability), 3),
        **compute_speech_metrics(words, float(info.duration)),
        "models": model_description(),
        "note": NOTE,
    }
