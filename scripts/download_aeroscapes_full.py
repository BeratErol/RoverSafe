"""Tam AeroScapes'i indirir; görülmüş küçük test sekanslarını yeni kör testten ayırır."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

if __package__:
    from .download_aeroscapes import ROOT, download, list_files
else:
    from download_aeroscapes import ROOT, download, list_files


def group_duplicate_scenes(records):
    """Birebir aynı kare paylaşan sekansları bağlı grup halinde bir bölümde tutar."""
    parent = {item["scene_id"]: item["scene_id"] for item in records}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    seen = {}
    for item in records:
        digest = item["files"]["image"]["sha256"]
        scene = item["scene_id"]
        if digest in seen:
            a, b = find(scene), find(seen[digest])
            parent[max(a, b)] = min(a, b)
        seen[digest] = scene
    # Görülmüş demo veya geliştirme verisi yeni kör teste taşınmaz.
    priority = {"demo": 0, "development": 1, "validation": 2, "test": 3}
    splits = {}
    for item in records:
        root = find(item["scene_id"])
        current = splits.get(root, item["split"])
        splits[root] = min((current, item["split"]), key=priority.get)
    for item in records:
        item["source_scene_id"] = item["scene_id"]
        item["scene_id"] = find(item["scene_id"])
        item["split"] = splits[item["scene_id"]]
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        parser.error("İndirme işçisi sayısı 1–12 olmalı.")
    old = json.loads((ROOT / "data/manifests/aeroscapes_subset.json").read_text())
    old_scenes = {item["scene_id"]: item["split"] for item in old["samples"]}
    groups = defaultdict(list)
    for source_split in ("train", "val"):
        for path in list_files(f"data/images/{source_split}"):
            groups[Path(path).stem.split("_")[0]].append(path)
    new_scenes = [key for key in sorted(groups) if key not in old_scenes]
    new_splits = {key: ("validation" if i % 4 == 0 else "test" if i % 4 == 1 else "development")
                  for i, key in enumerate(new_scenes)}
    assigned = {}
    for key in sorted(groups):
        assigned[key] = ("development" if old_scenes[key] == "development" else "demo") if key in old_scenes else new_splits[key]
    pairs = [(path, assigned[key]) for key in sorted(groups) for path in sorted(groups[key])]
    print(f"{len(pairs)} görüntü/maske · {len(groups)} sekans · {len(new_scenes)} yeni sekans", flush=True)
    output = ROOT / "data/datasets/aeroscapes"
    def fetch(pair):
        for attempt in range(5):
            try:
                return download(pair, output)
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 8))
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, item in enumerate(pool.map(fetch, pairs), 1):
            records.append(item)
            if index % 100 == 0:
                print(f"İndirildi/doğrulandı: {index}/{len(pairs)}", flush=True)
    records = group_duplicate_scenes(records)
    manifest = {**{key: value for key, value in old.items() if key != "samples"},
                "annotation_kind": "semantic_proxy", "samples": records,
                "split_note": "Old test/demo groups remain demo. Exact duplicate image hashes join source scenes into one group before splitting; priority demo/development/validation/test. New held-out groups exclude seen samples."}
    path = ROOT / "data/manifests/aeroscapes_full.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Hızlı karşılaştırma alt kümesi de sekans ayrımını aynen korur.
    selected = []
    for key in sorted({item["scene_id"] for item in records}):
        unique = {}
        for item in records:
            if item["scene_id"] == key:
                unique.setdefault(item["files"]["image"]["sha256"], item)
        scene = list(unique.values())
        count = min(12, len(scene))
        selected.extend(scene[round(i * (len(scene) - 1) / max(count - 1, 1))] for i in range(count))
    (ROOT / "data/manifests/aeroscapes_expanded_eval.json").write_text(
        json.dumps(dict(manifest, samples=selected), indent=2), encoding="utf-8")
    for split in ("development", "validation", "test", "demo"):
        rows = [item for item in records if item["split"] == split]
        print(split, len(rows), "kare", len({item['scene_id'] for item in rows}), "sekans", flush=True)


if __name__ == "__main__":
    main()
