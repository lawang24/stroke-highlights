"""Download verified upstream GRU and MoveNet weights for local experiments."""

from pathlib import Path
import hashlib
import urllib.request

ROOT = Path(__file__).resolve().parents[1] / "models/tennis-rnn"
REVISION = "6fdfd938f8d6893e2c926eb12fcc7f9bfc3e792f"
ARTIFACTS = {
    "tennis_rnn.h5": (
        f"https://raw.githubusercontent.com/antoinekeller/tennis_shot_recognition/{REVISION}/tennis_rnn.h5",
        "17dcbc5aba7ac4b88378bc0f8e02ed548de7c749f58c6ccb4969daa39020ade6"),
    "movenet.tflite": (
        "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite",
        "0fac2226112d0371903ca86e3853cec24ef603a0b2f96f589b180f0ebdd135ab"),
}


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_models():
    for name, (_, digest) in ARTIFACTS.items():
        path = ROOT / name
        if not path.exists():
            raise ValueError("Tennis RNN weights are missing. Run: python3 scripts/setup_tennis_rnn.py")
        if checksum(path) != digest:
            raise ValueError(f"Model checksum mismatch: {path}. Remove it and rerun setup.")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in ARTIFACTS.items():
        path = ROOT / name
        if path.exists():
            if checksum(path) != digest:
                raise SystemExit(f"Model checksum mismatch: {path}. Remove it and rerun setup.")
        else:
            temporary = path.with_suffix(".download")
            try:
                print(f"Downloading {name}…", flush=True)
                with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as handle:
                    while block := response.read(1024 * 1024):
                        handle.write(block)
                if checksum(temporary) != digest:
                    raise ValueError(f"Downloaded {name} failed SHA-256 verification")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        print(f"Verified {path}")


if __name__ == "__main__":
    main()
