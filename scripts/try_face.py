"""Manual check: send image files to a running ml-service and print the result.

    python scripts/try_face.py test_images/*.jpg

Reads ML_SERVICE_KEY (and optional ML_SERVICE_URL) from the environment.
Not part of the running app.
"""

import base64
import json
import os
import sys

import httpx

URL = os.environ.get("ML_SERVICE_URL", "http://127.0.0.1:8002")
KEY = os.environ.get("ML_SERVICE_KEY", "")


def main(paths: list[str]) -> None:
    for path in paths:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        r = httpx.post(
            f"{URL}/analyze-face",
            json={"image": b64},
            headers={"X-Internal-Key": KEY},
            timeout=120,
        )
        print(f"\n=== {path} -> HTTP {r.status_code}")
        print(json.dumps(r.json(), indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/try_face.py <image> [image ...]")
    main(sys.argv[1:])
