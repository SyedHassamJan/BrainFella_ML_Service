"""Manual check: send audio files to a running ml-service and print the result.

    python scripts/try_voice.py test_audio/*.wav [--language en]

Reads ML_SERVICE_KEY (and optional ML_SERVICE_URL) from the environment.
Not part of the running app.
"""

import base64
import json
import os
import sys
import time

import httpx

URL = os.environ.get("ML_SERVICE_URL", "http://127.0.0.1:8002")
KEY = os.environ.get("ML_SERVICE_KEY", "")


def main(args: list[str]) -> None:
    language = None
    if "--language" in args:
        i = args.index("--language")
        language = args[i + 1]
        args = args[:i] + args[i + 2 :]
    for path in args:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        body = {"audio": b64, **({"language": language} if language else {})}
        t = time.time()
        r = httpx.post(f"{URL}/analyze-voice", json=body, headers={"X-Internal-Key": KEY}, timeout=300)
        print(f"\n=== {path} -> HTTP {r.status_code} ({time.time() - t:.1f}s)")
        print(json.dumps(r.json(), indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/try_voice.py <audio> [audio ...] [--language en]")
    main(sys.argv[1:])
