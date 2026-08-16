"""Check the approved external data sources without downloading model bodies."""

from __future__ import annotations

import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_sources import (
    MODEL_DOWNLOAD_URLS,
    OPENCV_FACE_MODEL_URL,
    OPENCV_FACE_PROTO_URL,
    PYTORCH_CUDA_WHEEL_INDEX,
    TRUSTED_DOWNLOAD_HOSTS,
)
from src.scripts.runtime import configure_script_output

print = configure_script_output(__name__)


def source_is_approved(url: str) -> bool:
    parsed = urllib.parse.urlsplit(url)
    return (
        parsed.scheme == "https"
        and parsed.hostname is not None
        and parsed.hostname.lower() in TRUSTED_DOWNLOAD_HOSTS
    )


def check_source(label: str, url: str) -> bool:
    if not source_is_approved(url):
        print.warning(f"{label}: rejected; source is not an approved HTTPS host")
        return False
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "NIRT-ShooterRobot-source-check"},
        method="HEAD",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            final_url = response.geturl()
            if not source_is_approved(final_url):
                print.warning(f"{label}: rejected; redirect target is unapproved")
                return False
            print(f"{label}: online ({response.status})")
            return True
    except urllib.error.HTTPError as exc:
        # Some release providers reject HEAD. An HTTP response still proves the
        # endpoint is reachable, but a non-success status is reported clearly.
        print.warning(f"{label}: HTTP {exc.code}")
    except urllib.error.URLError as exc:
        print.warning(f"{label}: offline ({exc.reason})")
    return False


def main() -> None:
    sources = [("PyTorch CUDA wheel index", PYTORCH_CUDA_WHEEL_INDEX)]
    sources.extend((f"Model {name}", url) for name, url in MODEL_DOWNLOAD_URLS.items())
    sources.extend(
        [
            ("OpenCV face deploy config", OPENCV_FACE_PROTO_URL),
            ("OpenCV face model", OPENCV_FACE_MODEL_URL),
        ]
    )
    results = [check_source(label, url) for label, url in sources]
    print(f"Online approved sources: {sum(results)}/{len(results)}")
    print.warning(
        "Cryptographic integrity is not verified because no publisher-signed "
        "SHA-256 manifest is configured."
    )
    if not all(results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
