"""Sadece development sekanslarında dengeli piksel örnekleriyle ExtraTrees eğitir."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier

from evaluation import load_manifest, read_labels
from preprocessing import read_image
from surface_classification import FEATURE_VERSION, feature_grid


def train(manifest_path, output, per_scene=8, per_class=512):
    if type(per_scene) is not int or type(per_class) is not int or min(per_scene, per_class) < 1:
        raise ValueError("Örnek sayıları pozitif olmalı.")
    manifest = load_manifest(manifest_path)
    groups = defaultdict(list)
    for item in manifest["samples"]:
        if item["split"] == "development":
            groups[item["scene_id"]].append(item)
    if not groups:
        raise ValueError("Eğitim için development verisi gerekli.")
    rng = np.random.default_rng(42)
    xs, ys, selected = [], [], []
    start = perf_counter()
    for scene in sorted(groups):
        unique = {}
        for item in sorted(groups[scene], key=lambda item: item["id"]):
            digest = item.get("files", {}).get("image", {}).get("sha256", item["id"])
            unique.setdefault(digest, item)
        items = list(unique.values())
        count = min(per_scene, len(items))
        for index in np.linspace(0, len(items) - 1, count).round().astype(int):
            sample = items[index]
            image = read_image(ROOT / sample["image"])
            labels = read_labels(ROOT / sample["mask"])
            if labels.shape != image.shape[:2]:
                raise ValueError("Eğitim görüntüsü ve etiket boyutları uyuşmuyor.")
            features = feature_grid(image, 4).reshape(-1, 14)
            labels = labels[::4, ::4].ravel()
            for value, key in ((0, "risk"), (1, "target")):
                indices = np.flatnonzero(np.isin(labels, manifest["class_policy"][key]))
                if len(indices):
                    chosen = rng.choice(indices, min(per_class, len(indices)), replace=False)
                    xs.append(features[chosen])
                    ys.append(np.full(len(chosen), value, dtype=np.uint8))
            selected.append(sample["id"])
        print(f"Eğitim özellikleri: {len(selected)} görüntü · {scene}", flush=True)
    x, y = np.concatenate(xs), np.concatenate(ys)
    if len(np.unique(y)) != 2:
        raise ValueError("Eğitim iki sınıfı da içermeli.")
    model = ExtraTreesClassifier(n_estimators=48, max_depth=14, min_samples_leaf=15,
                                class_weight="balanced", random_state=42, n_jobs=2)
    model.fit(x, y)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"feature_version": FEATURE_VERSION, "model": model}, output, compress=3)
    metadata = {"feature_version": FEATURE_VERSION, "training_split": "development", "seed": 42,
                "scene_ids": sorted(groups), "sample_ids": selected, "image_count": len(selected),
                "pixel_count": len(y), "class_counts": np.bincount(y).tolist(),
                "max_images_per_scene": per_scene, "max_pixels_per_class_per_image": per_class,
                "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
                "model_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "sklearn_version": sklearn.__version__, "parameters": model.get_params(),
                "training_grid_step": 4,
                "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ("surface_classification.py", "scripts/train_surface.py")},
                "seconds": perf_counter() - start,
                "score_note": "Uncalibrated class score, not a probability of physical safety."}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/aeroscapes_full.json")
    parser.add_argument("--output", type=Path, default=ROOT / "models/surface.joblib")
    parser.add_argument("--per-scene", type=int, default=8)
    parser.add_argument("--per-class", type=int, default=512)
    args = parser.parse_args()
    result = train(args.manifest, args.output, args.per_scene, args.per_class)
    print(f"Model: {result['image_count']} görüntü, {result['pixel_count']} piksel · {args.output}")
