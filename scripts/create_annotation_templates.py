"""Birkaç farklı geliştirme sekansı için tamamen belirsiz manuel maske şablonları üretir."""

import argparse
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("Sayı pozitif olmalı.")
    source = json.loads((ROOT / "data/manifests/aeroscapes_subset.json").read_text(encoding="utf-8"))
    selected, scenes = [], set()
    for item in source["samples"]:
        if item["split"] != "development" or item["scene_id"] in scenes:
            continue
        mask_path = ROOT / "data/annotations/manual" / (item["id"] + ".png")
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not mask_path.exists():
            with Image.open(ROOT / item["image"]) as image:
                Image.new("L", image.size, 255).save(mask_path)
        selected.append({"id": item["id"], "scene_id": item["scene_id"], "split": "development",
                         "image": item["image"], "mask": mask_path.relative_to(ROOT).as_posix(),
                         "annotation_kind": "manual_visual", "reviewed": False})
        scenes.add(item["scene_id"])
        if len(selected) >= args.count:
            break
    output = ROOT / "data/manifests/manual_visual.json"
    if output.exists():
        parser.exit(1, "Mevcut manuel manifest korunuyor; otomatik üzerine yazılmadı.\n")
    output.write_text(json.dumps({"annotation_kind": "manual_visual", "schema_version": 1,
                                  "class_policy": {"target": [1], "risk": [0], "unknown": [255]},
                                  "samples": selected}, indent=2), encoding="utf-8")
    print(f"{len(selected)} boş/belirsiz maske hazır; bunlar etiketlenmiş gerçek veri değildir.")


if __name__ == "__main__":
    main()
