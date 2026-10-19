"""Sabit geliştirme karelerinde bozulma duyarlılığı; parametre seçimi yapmaz."""

import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from configuration import load_config
from evaluation import aggregate, evaluate_result, load_manifest, read_labels
from obstacle_detection import YOLOObstacleDetector
from pipeline import process_frame
from preprocessing import read_image
from surface_classification import SurfacePredictor
from visualization import save_scoring_results


def perturb(image, method):
    if method == "original":
        return image.copy()
    if method == "blur":
        return cv2.GaussianBlur(image, (11, 11), 3)
    if method == "dark":
        return (image.astype(np.float32) * 0.5).astype(np.uint8)
    if method == "shadow":
        result = image.copy()
        result[:, :image.shape[1] // 2] = (result[:, :image.shape[1] // 2] * 0.4).astype(np.uint8)
        return result
    if method == "noise":
        rng = np.random.default_rng(42)
        return np.clip(image.astype(np.float32) + rng.normal(0, 15, image.shape), 0, 255).astype(np.uint8)
    raise ValueError("Bilinmeyen bozulma.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/aeroscapes_expanded_eval.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/surface.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/robustness")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    # Her sekansın ilk geliştirme karesi; en fazla 20 sekans, teste bakılmaz.
    scenes = {}
    for item in manifest["samples"]:
        if item["split"] == "development":
            scenes.setdefault(item["scene_id"], item)
    samples = [scenes[key] for key in sorted(scenes)[:20]]
    options = load_config(args.config)
    if options["surface_config"] is not None:
        options["surface_predictor"] = SurfacePredictor(options["surface_config"])
    if options["obstacle_config"] is not None:
        options["obstacle_detector"] = YOLOObstacleDetector(options["obstacle_config"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, summaries = [], {}
    for method in ("original", "blur", "dark", "shadow", "noise"):
        results = []
        for index, sample in enumerate(samples):
            image = perturb(read_image(ROOT / sample["image"]), method)
            tick = perf_counter()
            frame = process_frame(image, **options)
            row = {"method": method, "sample_id": sample["id"], "scene_id": sample["scene_id"],
                   "processing_seconds": perf_counter() - tick,
                   **evaluate_result(frame, read_labels(ROOT / sample["mask"]), manifest["class_policy"])}
            results.append(row)
            rows.append(row)
            if index == 0:
                save_scoring_results(image, frame.scoring, args.output_dir / "previews" / method)
        summaries[method] = aggregate(results)
        print(f"Sağlamlık: {method} · {len(results)} kare", flush=True)
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps({"split": "development", "config": json.loads(args.config.read_text()),
                                                          "methods": summaries}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
