"""Yalnızca validation sonuçlarıyla yüzey eşiğini seçip test profillerini dondurur."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import run_experiments


def choose(methods):
    # Önceden belirlenen kural: en az %20 karede öneri; sonra düşük risk önceliği.
    feasible = [name for name, value in methods.items() if (value["proposal_rate"] or 0) >= 0.2]
    if feasible:
        return min(feasible, key=lambda name: (methods[name]["risk_intersection_rate"] if methods[name]["risk_intersection_rate"] is not None else 1,
                                               -(methods[name]["footprint_target_only_rate"] or 0),
                                               -(methods[name]["iou"] or 0), name))
    return min(methods, key=lambda name: (-(methods[name]["proposal_rate"] or 0), name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/aeroscapes_expanded_eval.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/experiments/surface_validation")
    args = parser.parse_args()
    base = {"preprocessing": {}, "terrain": {}, "segmentation": {}, "scoring": {}}
    definitions = [{"name": f"surface_{int(threshold * 100)}", "config": dict(base, surface={"threshold": threshold})}
                   for threshold in (0.6, 0.75, 0.9)]
    path = ROOT / "configs/surface_validation.json"
    path.write_text(json.dumps(definitions, indent=2), encoding="utf-8")
    report = run_experiments(args.manifest, path, args.output_dir, split="validation")
    name = choose(report["methods"])
    selected = next(item for item in definitions if item["name"] == name)
    (ROOT / "configs/surface.json").write_text(json.dumps(selected["config"], indent=2), encoding="utf-8")
    final = [{"name": "canny_texture", "config": base},
             {"name": "surface_selected", "config": selected["config"]},
             {"name": "surface_yolo", "config": dict(selected["config"], obstacles={"model_path": "models/yolo11n.pt", "device": "cpu"})}]
    (ROOT / "configs/expanded_experiments.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    decision = {"selection_split": "validation", "selected": name,
                "rule": "proposal_rate>=0.20 then minimum risk_intersection_rate, maximum target_only_rate, maximum IoU; otherwise maximum proposal_rate",
                "model_sha256": hashlib.sha256((ROOT / "models/surface.joblib").read_bytes()).hexdigest(),
                "validation_manifest_sha256": report["manifest_sha256"], "methods": report["methods"],
                "external_data_used_for_selection": False}
    (ROOT / "reports/surface_selection.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(f"Dondurulan profil: {name}")


if __name__ == "__main__":
    main()
