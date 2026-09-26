"""BrainHealth ML service - pretrained-model inference for the screening pipeline.

Internal service: called only by the NestJS API (never by mobile/web clients).
Binds to 127.0.0.1 and requires the X-Internal-Key header.
"""

import utils.http_session
import utils.transport
import utils.session
import hmac
import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from . import face_analysis, voice_analysis
from .schemas import (
    AnalyzeFaceRequest,
    AnalyzeFaceResponse,
    AnalyzeVoiceRequest,
    AnalyzeVoiceResponse,
    ErrorResponse,
)

log = logging.getLogger("ml-service")


@asynccontextmanager
async def lifespan(_: FastAPI):
    face_analysis.load_models()  # once, at startup
    voice_analysis.load_model()
    log.info("pretrained models loaded")
    yield


app = FastAPI(title="BrainHealth ML Service", lifespan=lifespan)


def require_internal_key(x_internal_key: str = Header(default="")):
    expected = os.environ.get("ML_SERVICE_KEY", "")
    if not expected:  # fail closed when not configured
        raise HTTPException(status_code=503, detail="service_not_configured")
    if not hmac.compare_digest(x_internal_key, expected):
        raise HTTPException(status_code=401, detail="unauthorized")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelsLoaded": face_analysis.models_loaded() and voice_analysis.model_loaded(),
        "face": face_analysis.models_loaded(),
        "voice": voice_analysis.model_loaded(),
    }


@app.post(
    "/analyze-face",
    response_model=AnalyzeFaceResponse | ErrorResponse,
    dependencies=[Depends(require_internal_key)],
)
def analyze_face(body: AnalyzeFaceRequest):
    # Plain `def` so FastAPI runs the blocking inference in its threadpool.
    try:
        return face_analysis.analyze_face(body.image)
    except face_analysis.AnalysisError as e:
        return {"error": e.code}


@app.post(
    "/analyze-voice",
    response_model=AnalyzeVoiceResponse | ErrorResponse,
    dependencies=[Depends(require_internal_key)],
)
def analyze_voice(body: AnalyzeVoiceRequest):
    # Plain `def` -> threadpool. The transcript is returned, never logged or stored.
    try:
        return voice_analysis.analyze_voice(body.audio, body.language)
    except voice_analysis.VoiceError as e:
        return {"error": e.code}


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception):
    log.exception("unhandled error")
    return JSONResponse(status_code=500, content={"error": "analysis_failed"})
