"""YOLO arabirimini, kutu maskesini ve engellerin öneriden dışlanmasını doğrular."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from obstacle_detection import (
    ObstacleConfig, ObstacleDetectionError, YOLOObstacleDetector, boxes_to_mask,
)
from pipeline import process_image_file
from segmentation import SegmentationConfig
from visualization import save_image


class TensorStub:
    def __init__(self, values):
        self.values = np.array(values, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def prediction(shape, boxes, classes, confidences):
    return SimpleNamespace(orig_shape=shape, boxes=SimpleNamespace(
        xyxy=TensorStub(np.array(boxes).reshape(-1, 4)), cls=TensorStub(classes), conf=TensorStub(confidences),
    ))


class ObstacleDetectionTests(unittest.TestCase):
    def test_fractional_boxes_round_outward_and_margins_clip_to_image(self):
        config = ObstacleConfig(margin=2)
        mask, obstacles = boxes_to_mask((12, 16), np.array([[-1.2, 2.4, 3.1, 5.9]]),
                                         np.array([0]), np.array([0.8]), {0: "person"}, config)
        expected = np.zeros((12, 16), dtype=bool)
        expected[0:8, 0:6] = True
        np.testing.assert_array_equal(mask, expected)
        self.assertEqual(obstacles[0].bounding_box, (0, 2, 4, 4))
        self.assertEqual(obstacles[0].expanded_box, (0, 0, 6, 8))

    def test_confidence_class_filtering_overlaps_and_empty_output(self):
        config = ObstacleConfig(margin=0, classes=("person",))
        boxes = np.array([[1, 1, 5, 5], [3, 3, 7, 7], [8, 8, 10, 10], [9, 1, 11, 4]])
        mask, items = boxes_to_mask((12, 12), boxes, np.array([0, 0, 0, 1]),
                                    np.array([0.9, 0.7, 0.1, 0.9]), {0: "person", 1: "car"}, config)
        self.assertEqual(len(items), 2)
        self.assertEqual(int(mask.sum()), 28)  # 16 + 16 - 4 örtüşme.
        empty, items = boxes_to_mask((12, 12), np.empty((0, 4)), np.empty(0), np.empty(0), {}, config)
        self.assertEqual(items, [])
        self.assertFalse(empty.any())

    def test_bad_predictions_do_not_silently_become_no_obstacles(self):
        for box, class_id, confidence in (
            ([5, 1, 2, 4], 0, 0.9), ([0, 0, float("nan"), 4], 0, 0.9),
            ([0, 0, 2, 4], 0, 1.2), ([0, 0, 2, 4], 0.5, 0.9), ([0, 0, 2, 4], 9, 0.9),
        ):
            with self.subTest(box=box), self.assertRaises(ObstacleDetectionError):
                boxes_to_mask((12, 12), np.array([box]), np.array([class_id]), np.array([confidence]),
                              {0: "person"}, ObstacleConfig())

    def test_invalid_settings_are_rejected(self):
        for options in (
            {"confidence": 0}, {"confidence": float("nan")}, {"iou": 2}, {"margin": -1},
            {"margin": True}, {"image_size": 33}, {"device": ""}, {"classes": ()},
            {"classes": ("person", "person")}, {"model_path": ""},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ObstacleConfig(**options)

    def test_missing_model_fails_before_import_or_download(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ObstacleDetectionError, "bulunamadı"):
                YOLOObstacleDetector(ObstacleConfig(model_path=str(Path(directory) / "missing.pt")))

    def test_detector_loads_once_uses_original_bgr_and_resolves_names_from_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            path.write_bytes(b"test fixture, not model weights")
            model = SimpleNamespace(task="detect", names={4: "person", 9: "car"}, predict=Mock())
            model.predict.return_value = [prediction((32, 40), [[3, 4, 8, 10]], [4], [0.9])]
            factory = Mock(return_value=model)
            with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=factory)}), patch("obstacle_detection.version", return_value="test"):
                detector = YOLOObstacleDetector(ObstacleConfig(model_path=str(path), classes=("person",)))
                image = np.full((32, 40, 3), (5, 50, 150), dtype=np.uint8)
                first = detector.detect(image)
                detector.detect(image)
                self.assertEqual(factory.call_count, 1)
                self.assertEqual(model.predict.call_args.kwargs["classes"], [4])
                self.assertIs(model.predict.call_args.kwargs["source"], image)
                self.assertEqual(first.detections[0].class_name, "person")
                self.assertEqual(len(first.model_sha256), 64)
                model.predict.side_effect = RuntimeError("test inference failure")
                with self.assertRaisesRegex(ObstacleDetectionError, "öneri üretilmedi"):
                    detector.detect(image)

    def test_missing_optional_package_raises_installation_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            path.write_bytes(b"test fixture")
            with patch.dict(sys.modules, {"ultralytics": None}):
                with self.assertRaisesRegex(ObstacleDetectionError, "requirements-yolo.txt"):
                    YOLOObstacleDetector(ObstacleConfig(model_path=str(path)))

    def test_unsupported_class_and_wrong_result_shape_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            path.write_bytes(b"test fixture")
            model = SimpleNamespace(task="detect", names={0: "person"}, predict=Mock())
            with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=Mock(return_value=model))}), patch("obstacle_detection.version", return_value="test"):
                with self.assertRaisesRegex(ObstacleDetectionError, "bulunmayan sınıf"):
                    YOLOObstacleDetector(ObstacleConfig(model_path=str(path), classes=("rock",)))
                detector = YOLOObstacleDetector(ObstacleConfig(model_path=str(path)))
                model.predict.return_value = [prediction((10, 10), [], [], [])]
                with self.assertRaises(ObstacleDetectionError):
                    detector.detect(np.zeros((32, 40, 3), dtype=np.uint8))
                model.task = "classify"
                with self.assertRaisesRegex(ObstacleDetectionError, "Nesne tespit modeli gerekli"):
                    YOLOObstacleDetector(ObstacleConfig(model_path=str(path)))

    def test_pipeline_excludes_obstacles_and_records_completed_empty_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "engel çalışması"
            image_path = root / "örnek.png"
            save_image(image_path, np.full((40, 60, 3), 120, dtype=np.uint8))
            model_path = root / "model.pt"
            model_path.write_bytes(b"test fixture")
            config = ObstacleConfig(model_path=str(model_path), margin=2)
            model = SimpleNamespace(task="detect", names={0: "person"}, predict=Mock())
            seg = SegmentationConfig(min_area=1, opening_size=3, rover_width=5, rover_height=3, safety_margin=1)
            with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=Mock(return_value=model))}), patch("obstacle_detection.version", return_value="test"):
                detector = YOLOObstacleDetector(config)
                for scenario in ("one", "empty", "all"):
                    boxes = [[25, 10, 35, 30]] if scenario == "one" else [[0, 0, 60, 40]] if scenario == "all" else []
                    model.predict.return_value = [prediction((40, 60), boxes, [0] * len(boxes), [0.9] * len(boxes))]
                    output = root / scenario
                    process_image_file(image_path, output, segmentation_config=seg, obstacle_detector=detector)
                    summary = json.loads((output / "landing_recommendation.json").read_text(encoding="utf-8"))
                    self.assertTrue(summary["obstacle_detection_applied"])
                    obstacle_summary = json.loads((output / "obstacle_detections.json").read_text(encoding="utf-8"))
                    self.assertEqual(obstacle_summary["status"], "completed")
                    self.assertEqual(obstacle_summary["detection_count"], len(boxes))
                    with np.load(output / "candidate_regions.npz") as regions, np.load(output / "obstacles.npz") as obstacles:
                        self.assertFalse((regions["threshold_mask"] & obstacles["exclusion_mask"]).any())
                        self.assertFalse((regions["candidate_mask"] & obstacles["exclusion_mask"]).any())
                        if summary["best"]:
                            x, y, width, height = summary["best"]["margin_box"]
                            self.assertFalse(obstacles["exclusion_mask"][y:y + height, x:x + width].any())
                    self.assertEqual(summary["status"], "no_candidate" if scenario == "all" else "selected")

    def test_detector_failure_stops_pipeline_before_output_and_config_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "image.png"
            save_image(path, np.zeros((32, 40, 3), dtype=np.uint8))
            config = ObstacleConfig()
            detector = SimpleNamespace(config=config, detect=Mock(side_effect=ObstacleDetectionError("failed")))
            with self.assertRaises(ObstacleDetectionError):
                process_image_file(path, root / "output", obstacle_detector=detector)
            self.assertFalse((root / "output").exists())
            with self.assertRaises(ValueError):
                process_image_file(path, root / "other", obstacle_detector=detector,
                                   obstacle_config=replace(config, margin=50))


if __name__ == "__main__":
    unittest.main()
