"""Etiketli maskelerle piksel örtüşmesini ve önerilen ayak izini değerlendirir."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from segmentation import find_valid_centers

ROOT = Path(__file__).resolve().parent


def load_manifest(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data.get("samples"), list) or not data["samples"]:
        raise ValueError("Manifest boş veya geçersiz.")
    ids, scenes = set(), {}
    for item in data["samples"]:
        if item.get("annotation_kind") == "manual_visual" and item.get("reviewed") is not True:
            raise ValueError("Manuel anotasyon incelenmeden değerlendirilemez; reviewed=true kaydı gerekli.")
        if item["id"] in ids:
            raise ValueError("Manifestte aynı örnek iki kez bulunuyor.")
        ids.add(item["id"])
        scene, split = item["scene_id"], item["split"]
        if split not in {"development", "validation", "test", "demo"}:
            raise ValueError("Bilinmeyen veri bölümü.")
        if scene in scenes and scenes[scene] != split:
            raise ValueError("Aynı sekans birden fazla bölüme dağılmış; veri sızıntısı var.")
        scenes[scene] = split
        for kind in ("image", "mask"):
            local = (ROOT / item[kind]).resolve()
            if not local.is_relative_to(ROOT) or not local.is_file():
                raise ValueError(f"Manifest dosyası proje içinde mevcut olmalı: {item[kind]}")
            expected = item.get("files", {}).get(kind, {}).get("sha256")
            if expected and hashlib.sha256(local.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Dosya özeti uyuşmuyor: {local}")
    policy = data["class_policy"]
    sets = [set(policy[key]) for key in ("target", "risk", "unknown")]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Etiket sınıfları çakışıyor.")
    return data


def read_labels(path):
    with Image.open(path) as image:
        labels = np.array(image)
    if labels.ndim != 2:
        raise ValueError("Etiket maskesi tek kanallı veya palet indeksli olmalı.")
    return labels


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def evaluate_result(frame, labels, policy):
    prediction = frame.candidates.candidate_mask
    if labels.shape != prediction.shape:
        raise ValueError("Etiket ve tahmin boyutları farklı; etiketler sessizce ölçeklenmez.")
    allowed = policy["target"] + policy["risk"] + policy["unknown"]
    if not np.isin(labels, allowed).all():
        raise ValueError("Maskede politikada bulunmayan etiket var.")
    target = np.isin(labels, policy["target"])
    risk = np.isin(labels, policy["risk"])
    known = target | risk
    tp = int((prediction & target).sum())
    fp = int((prediction & risk).sum())
    fn = int((~prediction & target).sum())
    best = frame.scoring.best
    metrics = {"tp": tp, "fp": fp, "fn": fn, "known_pixels": int(known.sum()),
               "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
               "iou": ratio(tp, tp + fp + fn), "proposal": int(best is not None),
               "target_footprint_exists": int(find_valid_centers(target, frame.candidates.config).any()),
               "footprint_target_only": None, "footprint_risk_fraction": None, "footprint_unknown_fraction": None}
    if best is not None:
        x, y, width, height = best.margin_box
        if x < 0 or y < 0 or x + width > labels.shape[1] or y + height > labels.shape[0]:
            raise ValueError("Önerilen ayak izi etiket görüntüsünün dışına taşıyor.")
        metrics.update(footprint_target_only=int(target[y:y + height, x:x + width].all()),
                       footprint_risk_fraction=float(risk[y:y + height, x:x + width].mean()),
                       footprint_unknown_fraction=float((~known)[y:y + height, x:x + width].mean()))
    metrics["correct_abstention"] = int(not metrics["target_footprint_exists"] and best is None)
    return metrics


def aggregate(rows):
    tp, fp, fn = (sum(row[key] for row in rows) for key in ("tp", "fp", "fn"))
    proposed = [row for row in rows if row["proposal"]]
    no_target = [row for row in rows if not row["target_footprint_exists"]]
    return {"sample_count": len(rows), "proposal_count": len(proposed),
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn), "iou": ratio(tp, tp + fp + fn),
            "proposal_rate": ratio(len(proposed), len(rows)),
            "footprint_target_only_rate": ratio(sum(row["footprint_target_only"] for row in proposed), len(proposed)),
            "risk_intersection_rate": ratio(sum(row["footprint_risk_fraction"] > 0 for row in proposed), len(proposed)),
            "unknown_intersection_rate": ratio(sum(row["footprint_unknown_fraction"] > 0 for row in proposed), len(proposed)),
            "correct_abstention_rate": ratio(sum(row["correct_abstention"] for row in no_target), len(no_target)),
            "mean_processing_seconds": ratio(sum(row["processing_seconds"] for row in rows), len(rows))}
