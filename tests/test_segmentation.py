"""Aday eleme ve tam sığma kontrolünü bilinen geometrilerle doğrular."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pipeline import process_image_file
from segmentation import SegmentationConfig, extract_candidates, find_valid_centers
from terrain_analysis import TerrainConfig, TerrainResult
from visualization import save_image


def terrain_from_mask(mask):
    """Yalnızca True piksellerin yoğunluk eşiğinden geçeceği özellikler üretir."""
    zeros = np.zeros(mask.shape, dtype=np.float32)
    edges = np.zeros(mask.shape, dtype=np.uint8)
    return TerrainResult(zeros, edges.copy(), edges.copy(), (~mask).astype(np.float32),
                         zeros.copy(), TerrainConfig(window_size=3))


class SegmentationTests(unittest.TestCase):
    def test_fit_matches_brute_force_for_odd_even_sizes_margins_and_holes(self):
        masks = [np.ones((9, 11), dtype=bool), np.random.default_rng(42).random((9, 11)) > 0.15]
        holed = masks[0].copy()
        holed[4, 5] = False
        masks.append(holed)
        for mask in masks:
            for width, height, margin in ((1, 1, 0), (4, 2, 0), (3, 5, 0), (2, 3, 1), (20, 2, 0)):
                with self.subTest(width=width, height=height, margin=margin):
                    config = SegmentationConfig(rover_width=width, rover_height=height, safety_margin=margin)
                    expected = np.zeros_like(mask)
                    full_width, full_height = width + 2 * margin, height + 2 * margin
                    for y in range(mask.shape[0] - full_height + 1):
                        for x in range(mask.shape[1] - full_width + 1):
                            if mask[y:y + full_height, x:x + full_width].all():
                                expected[y + full_height // 2, x + full_width // 2] = True
                    np.testing.assert_array_equal(find_valid_centers(mask, config), expected)

    def test_exact_fit_has_one_center_and_extra_margin_rejects_it(self):
        mask = np.zeros((8, 12), dtype=bool)
        mask[1:5, 2:8] = True  # Pay dahil 6×4 alan.
        config = SegmentationConfig(rover_width=4, rover_height=2, safety_margin=1)
        centers = find_valid_centers(mask, config)
        self.assertEqual(int(centers.sum()), 1)
        self.assertTrue(centers[3, 5])
        self.assertFalse(find_valid_centers(mask, replace(config, safety_margin=2)).any())

    def test_large_thin_region_is_rejected_even_with_sufficient_area(self):
        mask = np.zeros((30, 60), dtype=bool)
        mask[10:12, 5:55] = True
        config = SegmentationConfig(min_area=50, opening_size=1, rover_width=5,
                                    rover_height=5, safety_margin=0)
        result = extract_candidates(terrain_from_mask(mask), config)
        self.assertEqual(result.regions, [])
        self.assertEqual(result.rejected_no_fit, 1)
        self.assertFalse(result.valid_centers.any())

    def test_regions_are_labeled_separately_and_small_region_is_removed(self):
        mask = np.zeros((35, 45), dtype=bool)
        mask[2:10, 2:10] = True
        mask[15:25, 25:35] = True
        mask[30:32, 1:3] = True
        config = SegmentationConfig(min_area=20, opening_size=1, rover_width=4,
                                    rover_height=3, safety_margin=0)
        result = extract_candidates(terrain_from_mask(mask), config)
        self.assertEqual(len(result.regions), 2)
        self.assertEqual(sorted(region.area_px for region in result.regions), [64, 100])
        self.assertEqual(sorted(region.valid_center_count for region in result.regions), [30, 56])
        self.assertEqual(result.rejected_small, 1)
        self.assertEqual(set(np.unique(result.labels)), {0, 1, 2})
        self.assertEqual(int(result.valid_centers.sum()), 86)
        self.assertFalse(result.candidate_mask[30, 1])

    def test_opening_removes_speckles_without_filling_rejected_hole(self):
        mask = np.zeros((30, 30), dtype=bool)
        mask[5:25, 5:25] = True
        mask[15, 15] = False
        mask[1, 1] = True
        config = SegmentationConfig(min_area=1, opening_size=3, rover_width=3,
                                    rover_height=3, safety_margin=0)
        result = extract_candidates(terrain_from_mask(mask), config)
        self.assertFalse(result.cleaned_mask[1, 1])
        self.assertFalse(result.cleaned_mask[15, 15])
        self.assertTrue(np.all(~result.cleaned_mask | result.threshold_mask))
        self.assertFalse(result.valid_centers[14:17, 14:17].any())

    def test_edge_pixels_and_high_variance_are_always_excluded(self):
        terrain = terrain_from_mask(np.ones((25, 25), dtype=bool))
        terrain.canny_edges[10, 10] = 255
        terrain.local_variance[15, 15] = 300
        config = SegmentationConfig(min_area=1, opening_size=1, rover_width=1,
                                    rover_height=1, safety_margin=0)
        result = extract_candidates(terrain, config)
        self.assertFalse(result.threshold_mask[10, 10])
        self.assertFalse(result.threshold_mask[15, 15])
        sobel = extract_candidates(replace(terrain, config=TerrainConfig(edge_method="sobel")), config)
        self.assertTrue(sobel.threshold_mask[10, 10])  # Seçilen kenar yöntemi önemlidir.

    def test_empty_scene_and_oversized_footprint_return_no_candidate(self):
        config = SegmentationConfig(min_area=1, opening_size=1)
        for mask in (np.zeros((32, 40), dtype=bool), np.ones((32, 40), dtype=bool)):
            result = extract_candidates(terrain_from_mask(mask), config)
            self.assertEqual(result.regions, [])
            self.assertFalse(result.candidate_mask.any())
            self.assertFalse(result.valid_centers.any())

    def test_invalid_parameters_and_feature_values_are_rejected(self):
        for options in (
            {"max_edge_density": -1}, {"max_edge_density": 2}, {"max_variance": float("nan")},
            {"max_variance": float("inf")}, {"min_area": 0}, {"opening_size": 2},
            {"rover_width": 0}, {"rover_height": True}, {"safety_margin": -1},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SegmentationConfig(**options)
        terrain = terrain_from_mask(np.ones((10, 10), dtype=bool))
        terrain.edge_density[3, 3] = float("nan")
        with self.assertRaises(ValueError):
            extract_candidates(terrain)
        with self.assertRaises(ValueError):
            find_valid_centers(np.ones((10, 10), dtype=np.uint8), SegmentationConfig())

    def test_pipeline_saves_candidates_and_explicit_no_candidate_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bölge çalışması"
            path = root / "görüntü.png"
            save_image(path, np.full((32, 40, 3), 120, dtype=np.uint8))
            config = SegmentationConfig(min_area=20, rover_width=4, rover_height=3, safety_margin=1)
            for oversized in (False, True):
                output = root / ("büyük rover" if oversized else "adaylar")
                current = replace(config, rover_width=100) if oversized else config
                process_image_file(path, output, segmentation_config=current)
                summary = json.loads((output / "candidates.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["status"], "no_candidate" if oversized else "candidates_found")
                self.assertEqual(summary["candidate_count"], 0 if oversized else 1)
                settings = json.loads((output / "settings.json").read_text(encoding="utf-8"))
                self.assertIsNotNone(settings["terrain"])
                self.assertEqual(settings["segmentation"]["rover_width"], current.rover_width)
                with np.load(output / "candidate_regions.npz", allow_pickle=False) as arrays:
                    self.assertEqual(arrays["labels"].dtype, np.int32)
                    self.assertEqual(arrays["valid_centers"].dtype, bool)
                    self.assertEqual(int(arrays["valid_centers"].sum()), summary["valid_center_count"])
                self.assertGreater((output / "candidate_comparison.png").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
