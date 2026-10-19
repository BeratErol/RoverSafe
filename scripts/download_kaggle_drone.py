"""Kaggle v2 RGB görüntülerini ve özgün 24 sınıflı maskeleri indirir."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASET = "santurini/semantic-segmentation-drone-dataset"
BASE = f"https://www.kaggle.com/api/v1/datasets"


def request(url):
    return urlopen(Request(url, headers={"User-Agent": "RoverSafe research project"}), timeout=90)


def remote_bytes(name):
    with request(f"{BASE}/download/{DATASET}/{quote(name, safe='')}?datasetVersionNumber=2") as response:
        data = response.read()
    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
            if len(names) != 1 or Path(names[0]).name != Path(name).name:
                raise ValueError("Tek dosya yanıtında beklenmeyen arşiv içeriği.")
            return archive.read(names[0])
    return data


def list_files():
    files, token = [], None
    while True:
        url = f"{BASE}/list/{DATASET}?pageSize=200"
        if token:
            url += "&pageToken=" + quote(token, safe="")
        with request(url) as response:
            data = json.load(response)
        files.extend(data["datasetFiles"])
        token = data.get("nextPageToken")
        if not token:
            return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        parser.error("İşçi sayısı 1–12 olmalı.")
    files = list_files()
    images = sorted(item["name"] for item in files if item["name"].startswith("classes_dataset/classes_dataset/original_images/"))
    masks = {Path(item["name"]).stem: item["name"] for item in files
             if item["name"].startswith("semantic_drone_dataset/semantic_drone_dataset/label_images_semantic/")}
    if len(images) != 400 or len(masks) != 400:
        raise ValueError("Beklenen sürüm 2 dosya sayıları değişti.")
    output = ROOT / "data/datasets/kaggle_drone"
    def fetch(name):
        stem = Path(name).stem
        record = {"id": "kaggle_" + stem, "scene_id": "kaggle_external", "split": "test",
                  "dataset": DATASET, "annotation_kind": "semantic_proxy", "files": {}}
        image_size = None
        for kind, remote in (("image", name), ("mask", masks[stem])):
            local = output / ("images" if kind == "image" else "masks") / (stem + ".png")
            sidecar = local.with_suffix(".source.json")
            source_hash = None
            if not local.exists() or not sidecar.exists():
                for attempt in range(5):
                    try:
                        raw = remote_bytes(remote)
                        source_hash = hashlib.sha256(raw).hexdigest()
                        with Image.open(io.BytesIO(raw)) as source:
                            if kind == "image":
                                converted = source.convert("RGB")
                            else:
                                labels = np.array(source)
                                if labels.ndim != 2 or not np.isin(labels, np.arange(24)).all():
                                    raise ValueError("Özgün maske 0–23 sınıf indeksleri içermeli.")
                                converted = Image.fromarray(labels).resize(image_size, Image.Resampling.NEAREST)
                            local.parent.mkdir(parents=True, exist_ok=True)
                            temporary = local.with_suffix(".part")
                            converted.save(temporary, format="PNG")
                            temporary.replace(local)
                        break
                    except Exception:
                        if attempt == 4:
                            raise
                        time.sleep(min(2 ** attempt, 8))
            if kind == "image":
                with Image.open(local) as image:
                    image_size = image.size
            record[kind] = local.relative_to(ROOT).as_posix()
            record["files"][kind] = {"remote": remote, "sha256": hashlib.sha256(local.read_bytes()).hexdigest()}
            if source_hash:
                sidecar.write_text(json.dumps({"source_sha256": source_hash}), encoding="utf-8")
            record["files"][kind].update(json.loads(sidecar.read_text()))
        return record
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, record in enumerate(pool.map(fetch, images), 1):
            records.append(record)
            if index % 25 == 0:
                print(f"Kaggle: {index}/400", flush=True)
    manifest = {"schema_version": 1, "source": DATASET, "version": 2,
                "source_url": f"https://www.kaggle.com/datasets/{DATASET}",
                "annotation_kind": "semantic_proxy",
                "label_names": dict(enumerate(["unlabeled", "paved-area", "dirt", "grass", "gravel", "water", "rocks", "pool", "vegetation", "roof", "wall", "window", "door", "fence", "fence-pole", "person", "dog", "car", "bicycle", "tree", "bald-tree", "ar-marker", "obstacle", "conflicting"])),
                "class_policy": {"target": [1, 2, 4], "risk": [i for i in range(24) if i not in [0, 1, 2, 4, 23]], "unknown": [0, 23, 255]},
                "transformation": "RGB converted losslessly to PNG. Original 24-class masks resized to RGB dimensions with nearest-neighbor. Roof class 9 is risk.",
                "split_note": "All images external test only. Acquisition grouping unavailable; one conservative scene_id; not 400 independent scenes.",
                "evaluation_note": "Semantic ground type is not verified physical landing safety.", "samples": records}
    path = ROOT / "data/manifests/kaggle_drone_external.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Tamamlandı: {len(records)} görüntü/maske", flush=True)


if __name__ == "__main__":
    main()
