"""Etiket ölçümleri, veri ayrımı, yapılandırma ve gerçek video dosyası akışını doğrular."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from configuration import parse_config, cli_defaults
from experiments import run_experiments
from evaluation import aggregate, evaluate_result, load_manifest
from landing_score import ScoringConfig
from pipeline import process_frame
from segmentation import SegmentationConfig
from terrain_analysis import TerrainConfig
from video import process_video
from visualization import save_image
from PIL import Image


OPTIONS = {"terrain_config": TerrainConfig(window_size=3),
           "segmentation_config": SegmentationConfig(min_area=1, opening_size=1,
                                                      rover_width=3, rover_height=3, safety_margin=0),
           "scoring_config": ScoringConfig()}
POLICY = {"target": [10], "risk": [1], "unknown": [0, 255]}


class EvaluationTests(unittest.TestCase):
    def test_target_and_unknown_are_not_conflated_and_empty_metrics_are_not_perfect(self):
        frame = process_frame(np.full((24, 32, 3), 120, dtype=np.uint8), **OPTIONS)
        labels = np.full((24, 32), 10, dtype=np.uint8)
        good = evaluate_result(frame, labels, POLICY)
        self.assertEqual(good["precision"], 1)
        self.assertEqual(good["footprint_target_only"], 1)
        labels[:] = 255
        unknown = evaluate_result(frame, labels, POLICY)
        self.assertIsNone(unknown["iou"])
        self.assertEqual(unknown["footprint_unknown_fraction"], 1)
        self.assertEqual(unknown["footprint_target_only"], 0)
        labels[:] = 1
        risk = evaluate_result(frame, labels, POLICY)
        self.assertEqual(risk["precision"], 0)
        self.assertEqual(risk["footprint_risk_fraction"], 1)

    def test_aggregate_tracks_coverage_and_abstention_separately(self):
        frame = process_frame(np.full((24, 32, 3), 120, dtype=np.uint8), **OPTIONS)
        row = {**evaluate_result(frame, np.full((24, 32), 10, dtype=np.uint8), POLICY), "processing_seconds": 0.1}
        options = dict(OPTIONS, segmentation_config=replace(OPTIONS["segmentation_config"], rover_width=100))
        none = process_frame(np.full((24, 32, 3), 120, dtype=np.uint8), **options)
        second = {**evaluate_result(none, np.full((24, 32), 1, dtype=np.uint8), POLICY), "processing_seconds": 0.2}
        result = aggregate([row, second])
        self.assertEqual(result["proposal_rate"], 0.5)
        self.assertEqual(result["footprint_target_only_rate"], 1)
        self.assertEqual(result["correct_abstention_rate"], 1)
        self.assertAlmostEqual(result["mean_processing_seconds"], 0.15)

    def test_scene_leakage_duplicates_and_policy_overlap_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "image").write_bytes(b"fixture")
            (root / "mask").write_bytes(b"fixture")
            path = root / "manifest.json"
            data = {"samples": [{"id": "a", "scene_id": "scene", "split": "development", "image": "image", "mask": "mask"},
                                 {"id": "b", "scene_id": "scene", "split": "test", "image": "image", "mask": "mask"}],
                    "class_policy": POLICY}
            path.write_text(json.dumps(data))
            with patch("evaluation.ROOT", root):
                with self.assertRaisesRegex(ValueError, "sızıntısı"):
                    load_manifest(path)
                data["samples"][1]["split"] = "development"
                data["samples"][1]["id"] = "a"
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    load_manifest(path)
                data["samples"] = [dict(data["samples"][0], annotation_kind="manual_visual", reviewed=False)]
                path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, "incelenmeden"):
                    load_manifest(path)

    def test_config_unknown_keys_invalid_types_and_cli_conversion(self):
        options = parse_config({"scoring": {}, "preprocessing": {"filter_type": "median"}})
        self.assertEqual(options["filter_type"], "median")
        self.assertIsNotNone(options["scoring_config"])
        for data in ({"typo": {}}, {"terrain": {"typo": 3}}, {"preprocessing": {"use_clahe": "yes"}},
                     {"obstacles": {"classes": "person"}}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                parse_config(data)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"terrain": {}, "scoring": {}, "obstacles": {"classes": ["person"]}}))
            defaults = cli_defaults(path)
            self.assertTrue(defaults["obstacles"])
            self.assertEqual(defaults["obstacle_classes"], ("person",))

    def test_experiment_writes_metrics_and_complete_status_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_image(root / "image.png", np.full((32, 40, 3), 120, dtype=np.uint8))
            Image.new("L", (40, 32), 10).save(root / "mask.png")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"class_policy": POLICY, "samples": [
                {"id": "a", "scene_id": "s1", "split": "development", "image": "image.png", "mask": "mask.png"}]}))
            definitions = root / "methods.json"
            definitions.write_text(json.dumps([{"name": "fixture", "config": {
                "terrain": {"window_size": 3}, "segmentation": {"min_area": 1, "rover_width": 3,
                "rover_height": 3, "safety_margin": 0}, "scoring": {}}}]))
            with patch("evaluation.ROOT", root), patch("experiments.ROOT", root):
                report = run_experiments(manifest, definitions, root / "results", skip_yolo=True)
            self.assertEqual(report["methods"]["fixture"]["iou"], 1)
            self.assertEqual(json.loads((root / "results/run_status.json").read_text())["status"], "completed")
            self.assertTrue((root / "results/results.csv").is_file())
            self.assertTrue((root / "results/comparison.png").is_file())


class VideoTests(unittest.TestCase):
    def create_video(self, path):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12, (64, 48))
        self.assertTrue(writer.isOpened())
        for index in range(6):
            image = np.full((48, 64, 3), 100 + index, dtype=np.uint8)
            writer.write(image)
        writer.release()

    def test_video_stride_timestamps_output_count_and_no_frame_artifact_explosion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.avi"
            self.create_video(source)
            result = process_video(source, root / "output", options=OPTIONS, stride=2)
            self.assertEqual(result["processed_frames"], 3)
            self.assertEqual(result["output_fps"], 6)
            records = [json.loads(line) for line in (root / "output/frames.jsonl").read_text().splitlines()]
            self.assertEqual([row["frame_index"] for row in records], [0, 2, 4])
            self.assertAlmostEqual(records[2]["timestamp_seconds"], 4 / 12)
            capture = cv2.VideoCapture(result["output"])
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
            capture.release()
            self.assertFalse(list((root / "output").glob("*.partial.avi")))
            self.assertEqual(len(list((root / "output").glob("*.png"))), 1)

    def test_video_failure_does_not_publish_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.avi"
            self.create_video(source)
            with patch("video.process_frame", side_effect=ValueError("fixture failure")):
                with self.assertRaisesRegex(ValueError, "fixture failure"):
                    process_video(source, root / "output", options=OPTIONS)
            self.assertFalse((root / "output/annotated.avi").exists())
            self.assertEqual(json.loads((root / "output/video_summary.json").read_text())["status"], "failed")
            with self.assertRaises(ValueError):
                process_video(source, root / "other", stride=0)


if __name__ == "__main__":
    unittest.main()
