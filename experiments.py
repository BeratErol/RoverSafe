"""Sabit yöntemleri sekans bazında ayrılmış veride çalıştırır; CSV, JSON ve grafik üretir."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
import platform
from pathlib import Path
from time import perf_counter

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from configuration import parse_config
from evaluation import ROOT, aggregate, evaluate_result, load_manifest, read_labels
from obstacle_detection import YOLOObstacleDetector
from pipeline import process_frame
from preprocessing import read_image
from surface_classification import SurfacePredictor
from visualization import save_scoring_results


def run_experiments(manifest_path, definitions_path, output_dir, *, split="development", skip_yolo=False):
    if split not in {"development", "validation", "test"}:
        raise ValueError("Deney development, validation veya test bölümünde çalışır.")
    manifest = load_manifest(manifest_path)
    samples = [item for item in manifest["samples"] if item["split"] == split]
    if not samples:
        raise ValueError("Seçilen bölüm boş.")
    definitions = json.loads(Path(definitions_path).read_text(encoding="utf-8"))
    names = [item["name"] for item in definitions]
    if len(names) != len(set(names)) or any(not name.replace("_", "").isalnum() for name in names):
        raise ValueError("Deney adları benzersiz ve dosya adına uygun olmalı.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "run_status.json"
    status_path.write_text('{"status": "running"}', encoding="utf-8")
    rows, summaries, models, recommendations = [], {}, {}, []
    try:
        for definition in definitions:
            options = parse_config(definition["config"])
            if options["scoring_config"] is None:
                raise ValueError("Deney yapılandırmasında scoring gerekli.")
            if skip_yolo and options["obstacle_config"] is not None:
                continue
            detector = YOLOObstacleDetector(options["obstacle_config"]) if options["obstacle_config"] else None
            if detector is not None:
                options["obstacle_detector"] = detector
                models[definition["name"]] = {"sha256": detector.model_sha256,
                                               "ultralytics_version": detector.ultralytics_version}
            method_rows = []
            if options["surface_config"] is not None:
                predictor = SurfacePredictor(options["surface_config"])
                options["surface_predictor"] = predictor
                models.setdefault(definition["name"], {})["surface_sha256"] = predictor.model_sha256
            for index, sample in enumerate(samples):
                image, labels = read_image(ROOT / sample["image"]), read_labels(ROOT / sample["mask"])
                started = perf_counter()
                frame = process_frame(image, **options)
                elapsed = perf_counter() - started
                recommendations.append({"method": definition["name"], "sample_id": sample["id"],
                                        "status": frame.scoring.status,
                                        "best": asdict(frame.scoring.best) if frame.scoring.best is not None else None})
                row = {"method": definition["name"], "sample_id": sample["id"], "scene_id": sample["scene_id"],
                       "split": split, "processing_seconds": elapsed,
                       **evaluate_result(frame, labels, manifest["class_policy"])}
                rows.append(row)
                method_rows.append(row)
                if index == 0:
                    save_scoring_results(image, frame.scoring, output_dir / "previews" / definition["name"], obstacles=frame.obstacles)
                if (index + 1) % 12 == 0:
                    print(f"{definition['name']}: {index + 1}/{len(samples)}", flush=True)
            summaries[definition["name"]] = aggregate(method_rows)
        if not rows:
            raise ValueError("Çalıştırılabilir deney kalmadı.")
        with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (output_dir / "recommendations.jsonl").open("w", encoding="utf-8") as file:
            for item in recommendations:
                file.write(json.dumps(item, allow_nan=False) + "\n")
        packages = {}
        for name in ("numpy", "opencv-python", "matplotlib", "ultralytics", "torch", "scikit-learn", "joblib"):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        report = {"split": split, "annotation_kind": manifest.get("annotation_kind", "semantic_proxy"), "target_is_verified_landing_safety": False,
                  "class_policy": manifest["class_policy"], "methods": summaries,
                  "environment": {"python": platform.python_version(), "platform": platform.platform(),
                                  "processor": platform.processor(), "packages": packages}, "models": models,
                  "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
                  "definitions_sha256": hashlib.sha256(Path(definitions_path).read_bytes()).hexdigest(),
                  "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(ROOT.glob("*.py"))},
                  "definitions": definitions, "scene_count": len({item['scene_id'] for item in samples})}
        (output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        figure = Figure(figsize=(12, 5), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, 2)
        method_names = list(summaries)
        for axis, key, title in zip(axes, ("iou", "risk_intersection_rate"),
                                    ("Hedef etiketiyle IoU (vekil ölçüt)", "Önerilerde risk etiketiyle kesişme oranı")):
            values = [summaries[name][key] for name in method_names]
            axis.barh(method_names, [value if value is not None else 0 for value in values])
            for index, value in enumerate(values):
                axis.text((value or 0) + 0.01, index, "N/A" if value is None else f"{value:.3f}", va="center")
            axis.set_xlim(0, 1.2)
            axis.set_title(title)
        figure.suptitle(f"{split} · {len(samples)} görüntü · fiziksel güvenlik başarımı değildir")
        figure.savefig(output_dir / "comparison.png", dpi=140)
        status_path.write_text('{"status": "completed"}', encoding="utf-8")
        return report
    except Exception as error:
        status_path.write_text(json.dumps({"status": "failed", "error": str(error)}), encoding="utf-8")
        raise


def main():
    parser = argparse.ArgumentParser(description="RoverSafe yöntem karşılaştırması")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/aeroscapes_subset.json")
    parser.add_argument("--definitions", type=Path, default=ROOT / "configs/experiments.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/experiments/development")
    parser.add_argument("--split", choices=["development", "validation", "test"], default="development")
    parser.add_argument("--skip-yolo", action="store_true")
    args = parser.parse_args()
    try:
        report = run_experiments(args.manifest, args.definitions, args.output_dir, split=args.split, skip_yolo=args.skip_yolo)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Deney başarısız: {error}\n")
    print(f"{len(report['methods'])} yöntem tamamlandı: {args.output_dir}")


if __name__ == "__main__":
    main()
