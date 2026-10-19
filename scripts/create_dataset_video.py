"""Ayrı demo sekansındaki gerçek drone karelerini yapay 10 FPS hızla AVI'ye birleştirir."""

import argparse
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from preprocessing import read_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/aeroscapes_subset.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/videos/aeroscapes_demo.avi")
    args = parser.parse_args()
    records = [item for item in json.loads(args.manifest.read_text(encoding="utf-8"))["samples"] if item["split"] == "demo"]
    if not records:
        parser.error("Manifestte demo sekansı yok.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image = read_image(ROOT / records[0]["image"])
    height, width = image.shape[:2]
    writer = cv2.VideoWriter(str(args.output.resolve()), cv2.VideoWriter_fourcc(*"MJPG"), 10, (width, height))
    if not writer.isOpened():
        raise OSError("Video yazıcısı açılamadı.")
    try:
        for item in records:
            image = read_image(ROOT / item["image"])
            if image.shape[:2] != (height, width):
                raise ValueError("Demo karelerinin boyutları farklı.")
            writer.write(image)
    finally:
        writer.release()
    print(f"{len(records)} gerçek kare; yapay 10 FPS; {args.output}")


if __name__ == "__main__":
    main()
