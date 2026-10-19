"""Puanlama, açıklanabilir bileşenler, geometri ve öneriden kaçınmayı doğrular."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from landing_score import ScoringConfig, score_landing
from pipeline import process_image_file
from segmentation import SegmentationConfig, extract_candidates
from terrain_analysis import TerrainConfig, TerrainResult
from visualization import save_image


def make_terrain(mask):
    zeros = np.zeros(mask.shape, dtype=np.float32)
    edges = np.zeros(mask.shape, dtype=np.uint8)
    return TerrainResult(zeros, edges.copy(), edges.copy(), (~mask).astype(np.float32),
                         zeros.copy(), TerrainConfig(window_size=3))


SMALL_ROVER = SegmentationConfig(min_area=1, opening_size=1, rover_width=3,
                                  rover_height=3, safety_margin=0)


class LandingScoreTests(unittest.TestCase):
    def test_clearance_selects_middle_and_includes_image_boundary(self):
        terrain = make_terrain(np.ones((21, 21), dtype=bool))
        candidates = extract_candidates(terrain, SMALL_ROVER)
        config = ScoringConfig(edge_weight=0, texture_weight=0, clearance_weight=1,
                                area_weight=0, clearance_target=100, min_score=0)
        result = score_landing(terrain, candidates, config)
        self.assertEqual(result.best.center, (10, 10))
        self.assertAlmostEqual(result.best.measurements["clearance_px"], 10)
        self.assertAlmostEqual(result.best.score, 0.1, places=6)
        self.assertTrue(np.isnan(result.score_map[0, 0]))
        self.assertTrue(np.isfinite(result.score_map[candidates.valid_centers]).all())

    def test_weights_can_change_selected_region(self):
        mask = np.zeros((50, 90), dtype=bool)
        mask[5:45, 5:35] = True
        mask[5:45, 55:85] = True
        terrain = make_terrain(mask)
        terrain.edge_density[5:45, 5:35] = 0.005
        terrain.edge_density[5:45, 55:85] = 0.02
        terrain.local_variance[5:45, 5:35] = 200
        terrain.local_variance[5:45, 55:85] = 20
        candidates = extract_candidates(terrain, SMALL_ROVER)
        config = ScoringConfig(edge_weight=1, texture_weight=0, clearance_weight=0, area_weight=0, min_score=0)
        edge_best = score_landing(terrain, candidates, config).best
        texture_best = score_landing(terrain, candidates, replace(config, edge_weight=0, texture_weight=1)).best
        self.assertLess(edge_best.center[0], 35)
        self.assertGreater(texture_best.center[0], 55)
        self.assertAlmostEqual(edge_best.score, 1 - 0.005 / 0.03, places=6)
        self.assertAlmostEqual(texture_best.score, 1 - 20 / 250, places=6)

    def test_components_reconstruct_weighted_score_and_record_full_footprint_means(self):
        terrain = make_terrain(np.ones((21, 25), dtype=bool))
        terrain.edge_density[:] = np.linspace(0.002, 0.02, 25)
        terrain.local_variance[:] = np.linspace(0, 100, 21)[:, None]
        candidates = extract_candidates(terrain, replace(SMALL_ROVER, rover_width=4, rover_height=2, safety_margin=1))
        config = ScoringConfig(edge_weight=3, texture_weight=2, clearance_weight=4, area_weight=1, min_score=0)
        point = score_landing(terrain, candidates, config).best
        expected = sum(point.components[name] * weight for name, weight in
                       (("edge", 3), ("texture", 2), ("clearance", 4), ("area", 1))) / 10
        self.assertAlmostEqual(point.score, expected, places=6)
        x, y, width, height = point.margin_box
        self.assertAlmostEqual(point.measurements["mean_edge_density"],
                               float(terrain.edge_density[y:y + height, x:x + width].mean()), places=6)
        self.assertAlmostEqual(point.measurements["mean_variance"],
                               float(terrain.local_variance[y:y + height, x:x + width].mean()), places=6)
        self.assertTrue(all(0 <= value <= 1 for value in point.components.values()))

    def test_clearance_matches_brute_force_distance_over_entire_even_footprint(self):
        mask = np.ones((9, 11), dtype=bool)
        mask[4, 5] = False
        terrain = make_terrain(mask)
        seg = replace(SMALL_ROVER, rover_width=2, rover_height=4)
        candidates = extract_candidates(terrain, seg)
        result = score_landing(terrain, candidates, ScoringConfig(min_score=0))
        outside = np.argwhere(~np.pad(candidates.candidate_mask, 1)) - 1
        for cy, cx in np.argwhere(candidates.valid_centers):
            x, y = int(cx) - 1, int(cy) - 2
            footprint = np.array([(row, column) for row in range(y, y + 4) for column in range(x, x + 2)])
            expected = np.sqrt(((footprint[:, None, :] - outside[None, :, :]) ** 2).sum(axis=2)).min()
            self.assertAlmostEqual(float(result.clearance_map[cy, cx]), float(expected), places=5)

    def test_invalid_centers_are_never_rescued_by_score(self):
        mask = np.ones((15, 19), dtype=bool)
        mask[7, 9] = False
        terrain = make_terrain(mask)
        candidates = extract_candidates(terrain, SMALL_ROVER)
        altered = replace(candidates, valid_centers=np.ones(mask.shape, dtype=bool))
        result = score_landing(terrain, altered, ScoringConfig(min_score=0))
        self.assertTrue(np.isnan(result.score_map[~candidates.valid_centers]).all())
        x, y, width, height = result.best.margin_box
        self.assertTrue(mask[y:y + height, x:x + width].all())

    def test_below_threshold_and_no_candidate_are_distinct(self):
        terrain = make_terrain(np.ones((21, 21), dtype=bool))
        terrain.edge_density[:] = 0.02
        candidates = extract_candidates(terrain, SMALL_ROVER)
        config = ScoringConfig(edge_weight=1, texture_weight=0, clearance_weight=0, area_weight=0, min_score=0.9)
        result = score_landing(terrain, candidates, config)
        self.assertEqual(result.status, "below_min_score")
        self.assertIsNone(result.best)
        self.assertEqual(len(result.ranked_regions), 1)
        no_fit = extract_candidates(terrain, replace(SMALL_ROVER, rover_width=100))
        empty = score_landing(terrain, no_fit, config)
        self.assertEqual(empty.status, "no_candidate")
        self.assertTrue(np.isnan(empty.score_map).all())

    def test_zero_feature_thresholds_and_deterministic_tie_break(self):
        terrain = make_terrain(np.ones((20, 20), dtype=bool))
        candidates = extract_candidates(terrain, replace(SMALL_ROVER, max_edge_density=0, max_variance=0))
        config = ScoringConfig(edge_weight=1, texture_weight=1, clearance_weight=0, area_weight=0, min_score=0)
        result = score_landing(terrain, candidates, config)
        self.assertEqual(result.best.center, (9, 9))
        self.assertEqual(result.best.score, 1)

    def test_invalid_weights_thresholds_and_maps_raise_errors(self):
        for options in (
            {"edge_weight": -1}, {"texture_weight": True}, {"clearance_target": 0},
            {"area_target_multiple": 0.5}, {"min_score": 1.1}, {"min_score": float("nan")},
            {"edge_weight": 0, "texture_weight": 0, "clearance_weight": 0, "area_weight": 0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ScoringConfig(**options)
        terrain = make_terrain(np.ones((21, 21), dtype=bool))
        candidates = extract_candidates(terrain, SMALL_ROVER)
        terrain.local_variance[0, 0] = float("inf")
        with self.assertRaises(ValueError):
            score_landing(terrain, candidates)

    def test_pipeline_saves_recommendation_settings_and_score_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "puan çalışması"
            path = root / "örnek.png"
            save_image(path, np.full((32, 40, 3), 120, dtype=np.uint8))
            for oversized in (False, True):
                output = root / ("aday yok" if oversized else "öneri")
                seg = replace(SMALL_ROVER, rover_width=100) if oversized else SMALL_ROVER
                process_image_file(path, output, segmentation_config=seg, scoring_config=ScoringConfig())
                summary = json.loads((output / "landing_recommendation.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["status"], "no_candidate" if oversized else "selected")
                self.assertFalse(summary["obstacle_detection_applied"])
                self.assertEqual(summary["best"] is None, oversized)
                settings = json.loads((output / "settings.json").read_text(encoding="utf-8"))
                self.assertEqual(settings["scoring"]["min_score"], 0.5)
                with np.load(output / "landing_scores.npz", allow_pickle=False) as arrays:
                    self.assertEqual(arrays["score_map"].shape, (32, 40))
                    if oversized:
                        self.assertTrue(np.isnan(arrays["score_map"]).all())
                self.assertGreater((output / "landing_comparison.png").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
