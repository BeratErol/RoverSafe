"""Veri seti yerine geçmeyen, tekrarlanabilir sentetik bir deneme görüntüsü üretir."""

from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    random = np.random.default_rng(42)
    image = np.full((480, 640, 3), (105, 150, 180), dtype=np.uint8)
    cv2.rectangle(image, (80, 100), (380, 360), (140, 175, 195), -1)
    for _ in range(35):
        x, y = random.integers([410, 30], [610, 450])
        cv2.circle(image, (int(x), int(y)), 8, (70, 85, 95), -1)
    noise = random.normal(0, 12, image.shape)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    cv2.putText(image, "SYNTHETIC DEMO", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    output = Path(__file__).resolve().parents[1] / "data/samples/synthetic_terrain.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise OSError("Sentetik örnek kodlanamadı.")
    output.write_bytes(encoded.tobytes())
    print(output)


if __name__ == "__main__":
    main()
