"""Öğrenilmiş filtreden önce veri sızıntısı, örnekleme ve aday eleme sözleşmeleri."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from configuration import parse_config
from pipeline import process_frame
from segmentation import SegmentationConfig
from surface_classification import SurfaceConfig, SurfacePredictor, SurfaceResult, feature_grid


class SurfaceTests(unittest.TestCase):
    def test_features_are_finite_and_grid_aligned(self):
        image = np.zeros((17, 19, 3), dtype=np.uint8)
        image[:, :8, 1] = 240
        features = feature_grid(image, 4)
        self.assertEqual(features.shape, (5, 5, 14))
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_array_equal(features[..., :3], image[::4, ::4] / np.float32(255))

    def test_surface_blocks_entire_footprint(self):
        config = SurfaceConfig()
        image = np.full((120, 200, 3), 120, dtype=np.uint8)
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[:, 100:] = True
        predictor = SimpleNamespace(config=config, predict=lambda image: SurfaceResult((~mask).astype(np.float32), mask, "fake"))
        frame = process_frame(image, surface_predictor=predictor,
                              segmentation_config=SegmentationConfig(rover_width=30, rover_height=20, safety_margin=5))
        self.assertIsNotNone(frame.scoring.best)
        x, y, width, height = frame.scoring.best.margin_box
        self.assertFalse(mask[y:y + height, x:x + width].any())
        self.assertFalse(frame.candidates.candidate_mask[:, 100:].any())

    def test_invalid_configuration_and_mismatch(self):
        for kwargs in ({"threshold": 0}, {"threshold": True}, {"threshold": float("nan")}, {"grid_step": 0}, {"grid_step": True}):
            with self.assertRaises(ValueError):
                SurfaceConfig(**kwargs)
        options = parse_config({"surface": {"threshold": 0.9}})
        self.assertEqual(options["surface_config"].threshold, 0.9)
        with self.assertRaises(ValueError):
            process_frame(np.zeros((20, 20, 3), dtype=np.uint8), surface_config=SurfaceConfig(),
                          surface_predictor=SimpleNamespace(config=SurfaceConfig(threshold=0.8)))

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "İsteğe bağlı yüzey bağımlılıkları yok")
    def test_model_prediction_and_version_guard(self):
        import joblib
        from sklearn.ensemble import ExtraTreesClassifier
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            x = np.array([[0] * 14, [1] * 14], dtype=np.float32)
            model = ExtraTreesClassifier(n_estimators=2, random_state=42).fit(x, [0, 1])
            joblib.dump({"feature_version": "color_texture_v1", "model": model}, path)
            predictor = SurfacePredictor(SurfaceConfig(model_path=str(path)))
            result = predictor.predict(np.zeros((17, 19, 3), dtype=np.uint8))
            self.assertEqual(result.probability.shape, (17, 19))
            self.assertEqual(result.exclusion_mask.dtype, bool)
            joblib.dump({"feature_version": "wrong", "model": model}, path)
            with self.assertRaises(ValueError):
                SurfacePredictor(SurfaceConfig(model_path=str(path)))

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "İsteğe bağlı yüzey bağımlılıkları yok")
    def test_training_never_uses_validation_or_test(self):
        from scripts.train_surface import train
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.full((20, 24, 3), 100, dtype=np.uint8)
            labels = np.ones((20, 24), dtype=np.uint8)
            labels[:, 12:] = 10
            Image.fromarray(image).save(root / "image.png")
            Image.fromarray(labels).save(root / "mask.png")
            samples = [{"id": split, "scene_id": split, "split": split, "image": "image.png", "mask": "mask.png"}
                       for split in ("development", "validation", "test")]
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"samples": samples, "class_policy": {"target": [10], "risk": [1], "unknown": [0]}}))
            with patch("evaluation.ROOT", root), patch("scripts.train_surface.ROOT", root):
                # Kod özetleri için gerçek kaynakların küçük test kopyaları.
                (root / "surface_classification.py").write_text("fixture")
                (root / "scripts").mkdir()
                (root / "scripts/train_surface.py").write_text("fixture")
                result = train(manifest, root / "model.joblib", per_scene=1, per_class=10)
            self.assertEqual(result["sample_ids"], ["development"])
            self.assertEqual(result["scene_ids"], ["development"])
            self.assertEqual(result["image_count"], 1)

    def test_profile_selection_requires_coverage_before_risk(self):
        from scripts.select_surface_profile import choose
        methods = {
            "no_proposals": {"proposal_rate": 0, "risk_intersection_rate": None, "footprint_target_only_rate": None, "iou": 0},
            "risky": {"proposal_rate": 1, "risk_intersection_rate": 0.8, "footprint_target_only_rate": 0.2, "iou": 0.5},
            "selective": {"proposal_rate": 0.3, "risk_intersection_rate": 0.1, "footprint_target_only_rate": 0.9, "iou": 0.4}}
        self.assertEqual(choose(methods), "selective")


class DatasetGroupingTests(unittest.TestCase):
    def test_duplicate_connections_never_cross_splits(self):
        from scripts.download_aeroscapes_full import group_duplicate_scenes
        def row(scene, split, digest):
            return {"scene_id": scene, "split": split, "files": {"image": {"sha256": digest}}}
        records = [row("a", "development", "one"), row("b", "test", "one"),
                   row("b", "test", "two"), row("c", "validation", "two"),
                   row("d", "validation", "three"), row("e", "demo", "three")]
        result = group_duplicate_scenes(records)
        self.assertEqual({item["split"] for item in result[:4]}, {"development"})
        self.assertEqual({item["scene_id"] for item in result[:4]}, {"a"})
        self.assertEqual({item["split"] for item in result[4:]}, {"demo"})
        self.assertEqual(result[3]["source_scene_id"], "c")


if __name__ == "__main__":
    unittest.main()
