"""Manifest özeti, boyut/etiket denetimi ve eğitim-test sekans ayrımını kaydeder."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from evaluation import load_manifest, read_labels


def audit(path):
    path = Path(path).resolve()
    manifest = load_manifest(path)
    counts, shapes, classes = Counter(), Counter(), Counter()
    scenes = {}
    image_hashes = {}
    duplicate_images = []
    for index, item in enumerate(manifest["samples"], 1):
        split = item["split"]
        counts[split] += 1
        scenes.setdefault(split, set()).add(item["scene_id"])
        with Image.open(ROOT / item["image"]) as image:
            width, height = image.size
            image.verify()
        labels = read_labels(ROOT / item["mask"])
        if labels.shape != (height, width):
            raise ValueError(f"Görüntü/maske boyut farkı: {item['id']}")
        keys, pixels = np.unique(labels, return_counts=True)
        allowed = set(sum((manifest["class_policy"][key] for key in ("target", "risk", "unknown")), []))
        if not set(keys).issubset(allowed):
            raise ValueError(f"Politika dışı etiket: {item['id']}")
        classes.update({str(int(key)): int(value) for key, value in zip(keys, pixels)})
        shapes[f"{width}x{height}"] += 1
        digest = hashlib.sha256((ROOT / item["image"]).read_bytes()).hexdigest()
        if digest in image_hashes:
            previous = image_hashes[digest]
            if previous["split"] != split:
                raise ValueError(f"Aynı görüntü farklı bölümlerde: {previous['id']} / {item['id']}")
            duplicate_images.append([previous["id"], item["id"]])
        image_hashes[digest] = item
        if index % 250 == 0:
            print(f"Denetlendi: {index}/{len(manifest['samples'])}", flush=True)
    return {"manifest": str(Path(path).relative_to(ROOT)), "manifest_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "sample_count": len(manifest["samples"]), "split_counts": dict(counts),
            "scene_counts": {key: len(value) for key, value in scenes.items()},
            "shapes": dict(shapes), "class_pixels": dict(classes), "same_split_duplicates": duplicate_images,
            "class_policy": manifest["class_policy"], "status": "passed"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/dataset_audit.json")
    args = parser.parse_args()
    reports = [audit(path) for path in args.manifests]
    args.output.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps([{key: row[key] for key in ('sample_count', 'split_counts', 'scene_counts', 'status')} for row in reports]))
