# BrainHealth ML Service

Internal FastAPI service for the multimodal screening pipeline (facial analysis + speech-to-text). **Pretrained-model inference only** — nothing here trains, fine-tunes or saves a model. Outputs are signals for the rule-based fusion layer in the NestJS API (a screening result, not a diagnosis).

Called only by the NestJS API, never by the mobile/web clients. Binds to `127.0.0.1` and requires the `X-Internal-Key` header.

## Setup (Python 3.12 — MediaPipe wheels stop at 3.12)

```powershell
cd brainhealth-ml-service
D:\Python312\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt
```

## Run

```powershell
$env:ML_SERVICE_KEY = "dev-only-change-me"   # must match the NestJS side later
$env:WHISPER_CACHE_DIR = "D:\whisper_models"  # optional: where the Whisper weights are cached
$env:WHISPER_MODEL = "base"                   # optional: base (default) or tiny
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8002
```

The first start downloads DeepFace's pretrained emotion weights (about 6 MB, into `~/.deepface`) and the Whisper `base` weights (about 145 MB, into `WHISPER_CACHE_DIR`). Both models load once at startup.

## Endpoints

- `GET /health` — `{ status, modelsLoaded, face, voice }`
- `POST /analyze-face` — body `{ "image": "<base64>" }`

Success:

```json
{
  "emotion": "neutral",
  "emotionScores": { "angry": 0.01, "happy": 0.05, "neutral": 0.9 },
  "eyeContact": true,
  "blinkRate": null,
  "headPose": { "yaw": 2.1, "pitch": -3.4, "roll": 0.8 },
  "models": { "emotion": "...", "landmarks": "..." },
  "note": "Signals from pretrained models for a screening result; not a diagnosis."
}
```

Expected failures return `{ "error": "no_face_detected" | "invalid_image" | "image_too_large" }` (HTTP 200); anything unexpected returns `{ "error": "analysis_failed" }` (HTTP 500). The service does not crash on bad input.

### `POST /analyze-voice`

Body `{ "audio": "<base64>", "language": "en" }` (`language` optional, 2–3 letter ISO code; omit to auto-detect). Any format PyAV can decode works (m4a/aac from `expo-av`, wav, mp3, webm) — no system ffmpeg needed.

```json
{
  "transcript": "I have been feeling very tired and anxious this week.",
  "duration": 7.95,
  "language": "en",
  "languageProbability": 0.997,
  "speakingRate": 182.1,
  "pauseCount": 0,
  "models": { "speechToText": "OpenAI Whisper 'base' via faster-whisper (pretrained, inference only)" },
  "note": "..."
}
```

- `speakingRate` = words per minute over the span from the first word to the last (leading/trailing silence excluded). `pauseCount` = gaps of at least 0.7 s between consecutive words (the threshold is borderline-sensitive: lossy codecs can shift a 0.6–0.7 s gap across it).
- Errors (HTTP 200): `invalid_audio`, `audio_too_large` (> 10 MB), `audio_too_long` (> 120 s), `no_speech_detected`; unexpected failures return `analysis_failed` (HTTP 500).
- The transcript is returned to the caller and is **never logged or stored** by this service. No emotion analysis happens here; the transcript goes to the text-analysis path.
- **Known limitation (scoping decision, for the FYP report):** the model is multilingual, but the downstream emotion models are English-only. Non-English speech (e.g. Urdu) is transcribed but not emotion-scored: callers must check `language` and skip or mark text analysis unavailable when it isn't `en`. Don't force `language` to work around this — forcing the wrong language produces a garbled transcript.

Face notes: `blinkRate` is always `null` (not measurable from one frame). `eyeContact` is a geometric heuristic on Face Mesh iris landmarks + head pose, not a learned gaze model. Head-pose signs: yaw > 0 = nose toward the image's right, pitch > 0 = nose down, roll > 0 = head tilted counter-clockwise as seen by the camera. These are verified on synthetic projections only, not on real turned-head photos (no such test image was available).

On Windows set `PYTHONUTF8=1` (DeepFace prints emoji that the default cp1252 console can't encode); `TF_ENABLE_ONEDNN_OPTS=0` and `TF_CPP_MIN_LOG_LEVEL=2` quiet TensorFlow's startup logs.

## Open items for item 15 (evaluation write-up) — NOT yet done

- **Voice accuracy is unmeasured.** Real human speech has been smoke-tested (two 45 s excerpts of a noisy NASA press-room recording; transcripts read as coherent) but there is no reference transcript, so no word-error-rate was computed. Item 15 should score Whisper `base` (vs `tiny`/`small`) on labeled real speech.
- **No phone voice memo tested.** Real mobile-mic recordings (AAC/m4a from `expo-av`, with genuine background noise, other accents, Urdu/Roman-Urdu speech) have not been tried.
- **Face:** sideways/turned-head real photos and non-smiling expressions are untested (see Face notes above).
- Measured on this CPU: about 4–5x real time (a 45 s clip took 9–11 s), so a 120 s clip takes roughly 30 s. Callers (NestJS) need a request timeout of at least 60 s.
- The 10 MB size cap is not the binding limit for AAC recordings (120 s at 64–128 kbps is 1–2 MB); the 120 s duration cap is. Uncompressed 44.1 kHz stereo WAV would exceed 10 MB in about a minute, so the mobile app must record AAC/m4a.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests            # no model weights needed
.\.venv\Scripts\python.exe scripts\try_face.py test_images\*.jpg   # against a running service
.\.venv\Scripts\python.exe scripts\try_voice.py test_audio\*.wav [--language en]
```
