"""AeroScapes aynasından küçük, sürümü sabit ve sekans bazında ayrılmış alt küme indirir."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

REPO = "dronefreak/Aeroscapes"
REVISION = "7c2e6f4e301882934799753989563d85ecd6157f"
ROOT = Path(__file__).resolve().parents[1]


def request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "RoverSafe/0.3"}), timeout=45)


def list_files(folder):
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/{folder}?limit=1000"
    files = []
    while url:
        with request(url) as response:
            files.extend(json.load(response))
            link = response.headers.get("Link", "")
        url = next((part.split("<")[1].split(">")[0] for part in link.split(",") if 'rel="next"' in part), None)
    return [item["path"] for item in files if item["type"] == "file" and item["path"].endswith(".jpg")]


def download(item, output):
    image_remote, split = item
    stem = Path(image_remote).stem
    mask_remote = image_remote.replace("/images/", "/masks/").replace(".jpg", ".png")
    record = {"id": stem, "scene_id": stem.split("_")[0], "split": split,
              "source_split": image_remote.split("/")[2], "dataset": "AeroScapes",
              "annotation_kind": "semantic_proxy", "files": {}}
    for kind, remote in (("image", image_remote), ("mask", mask_remote)):
        local = output / ("images" if kind == "image" else "masks") / Path(remote).name
        local.parent.mkdir(parents=True, exist_ok=True)
        if not local.exists():
            with request(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{remote}") as response:
                data = response.read()
            temporary = local.with_suffix(local.suffix + ".part")
            temporary.write_bytes(data)
            temporary.replace(local)
        record[kind] = local.relative_to(ROOT).as_posix()
        record["files"][kind] = {"remote": remote, "sha256": hashlib.sha256(local.read_bytes()).hexdigest()}
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=int, default=16, help="Değerlendirme sekansı sayısı")
    parser.add_argument("--per-group", type=int, default=3)
    parser.add_argument("--demo-frames", type=int, default=20)
    args = parser.parse_args()
    if min(args.groups, args.per_group, args.demo_frames) < 1:
        parser.error("Örnek sayıları pozitif olmalı.")
    groups = defaultdict(list)
    for split in ("train", "val"):
        for path in list_files(f"data/images/{split}"):
            groups[Path(path).stem.split("_")[0]].append(path)
    keys = sorted(groups)
    chosen = []
    for index, key in enumerate(keys[:args.groups + 1]):
        paths = sorted(groups[key], key=lambda path: Path(path).name)
        if index == 0:
            chosen.extend((path, "demo") for path in paths[:args.demo_frames])
        else:
            split = "test" if index % 4 == 0 else "development"
            count = min(args.per_group, len(paths))
            indices = [round(i * (len(paths) - 1) / max(count - 1, 1)) for i in range(count)]
            chosen.extend((paths[i], split) for i in indices)
    output = ROOT / "data/datasets/aeroscapes"
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(lambda item: download(item, output), chosen))
    manifest = {"schema_version": 1, "source": REPO, "revision": REVISION,
                "official_source": "https://github.com/ishann/aeroscapes", "mirror_is_unofficial": True,
                "class_policy": {"target": [10], "risk": list(range(1, 10)) + [11], "unknown": [0, 255]},
                "evaluation_note": "Road is a semantic proxy, not verified landing safety.", "samples": records}
    (ROOT / "data/manifests").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/manifests/aeroscapes_subset.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    counts = {split: sum(item["split"] == split for item in records) for split in ("development", "test", "demo")}
    print(json.dumps({"downloaded": len(records), "splits": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
