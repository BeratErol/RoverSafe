"""Arazi özelliklerini bilinen yüzeyler, sınırlar ve kayıt akışıyla doğrular."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pipeline import process_image_file
from terrain_analysis import TerrainConfig, analyze_terrain
from visualization import save_image


class TerrainAnalysisTests(unittest.TestCase):
    def test_uniform_surface_has_no_texture_or_edges_including_borders(self):
        for level in (0, 127, 255):
            with self.subTest(level=level):
                image = np.full((32, 40), level, dtype=np.uint8)
                original = image.copy()
                result = analyze_terrain(image)
                for feature in (
                    result.sobel_magnitude, result.sobel_edges, result.canny_edges,
                    result.edge_density, result.local_variance,
                ):
                    self.assertEqual(feature.shape, image.shape)
                    np.testing.assert_array_equal(feature, np.zeros_like(feature))
                np.testing.assert_array_equal(image, original)

    def test_step_edge_is_localized_and_sobel_does_not_overflow(self):
        image = np.zeros((41, 61), dtype=np.uint8)
        image[:, 30:] = 255
        result = analyze_terrain(image, TerrainConfig(window_size=5, edge_method="sobel"))
        self.assertAlmostEqual(float(result.sobel_magnitude.max()), 1020.0)
        self.assertTrue(np.any(result.sobel_edges[:, 29:31]))
        self.assertTrue(np.any(result.canny_edges[:, 28:32]))
        self.assertEqual(float(result.edge_density[:, :20].max()), 0)
        self.assertGreater(float(result.edge_density[:, 30].mean()), 0)
        self.assertLessEqual(float(result.edge_density.max()), 1)

    def test_texture_has_more_variance_than_uniform_patch(self):
        image = np.full((64, 128), 120, dtype=np.uint8)
        image[:, 64:] = np.random.default_rng(42).integers(0, 256, (64, 64), dtype=np.uint8)
        result = analyze_terrain(image, TerrainConfig(window_size=9))
        self.assertEqual(float(result.local_variance[10:54, 10:50].mean()), 0)
        self.assertGreater(float(result.local_variance[10:54, 78:118].mean()), 1000)

    def test_variance_matches_manual_population_variance_at_border_and_center(self):
        image = np.arange(63, dtype=np.uint8).reshape(7, 9) * 4
        result = analyze_terrain(image, TerrainConfig(window_size=3))
        padded = np.pad(image.astype(np.float64), 1, mode="reflect")
        for row, column in ((0, 0), (3, 4), (6, 8)):
            expected = padded[row:row + 3, column:column + 3].var()
            self.assertAlmostEqual(float(result.local_variance[row, column]), expected, places=3)

    def test_brightness_inversion_preserves_sobel_strength_and_variance(self):
        image = np.random.default_rng(7).integers(0, 256, (32, 32), dtype=np.uint8)
        config = TerrainConfig(window_size=7)
        first = analyze_terrain(image, config)
        inverted = analyze_terrain(255 - image, config)
        np.testing.assert_allclose(first.sobel_magnitude, inverted.sobel_magnitude, atol=1e-4)
        np.testing.assert_allclose(first.local_variance, inverted.local_variance, atol=1e-3)

    def test_density_uses_selected_method_and_fraction_not_byte_values(self):
        image = np.zeros((31, 31), dtype=np.uint8)
        image[:, 15:] = 255
        for method in ("canny", "sobel"):
            with self.subTest(method=method):
                result = analyze_terrain(image, TerrainConfig(window_size=5, edge_method=method))
                edges = result.canny_edges if method == "canny" else result.sobel_edges
                expected = np.count_nonzero(edges[13:18, 13:18]) / 25
                self.assertAlmostEqual(float(result.edge_density[15, 15]), expected, places=6)

    def test_invalid_configs_and_images_have_clear_errors(self):
        invalid = [
            {"window_size": 4}, {"window_size": True}, {"window_size": 1},
            {"edge_method": "other"}, {"canny_low": 150, "canny_high": 50},
            {"canny_low": -1}, {"canny_high": 256}, {"canny_low": 2.5},
            {"sobel_threshold": 0}, {"sobel_threshold": float("nan")},
            {"sobel_threshold": float("inf")}, {"sobel_threshold": True},
        ]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                TerrainConfig(**options)
        for image in (None, np.zeros((32, 32)), np.zeros((32, 32, 3), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                analyze_terrain(image)
        with self.assertRaisesRegex(ValueError, "kısa kenar"):
            analyze_terrain(np.zeros((8, 8), dtype=np.uint8))

    def test_pipeline_records_config_numeric_features_and_unicode_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "arazi ölçümü"
            path = root / "örnek.png"
            image = np.random.default_rng(1).integers(0, 256, (32, 40, 3), dtype=np.uint8)
            save_image(path, image)
            output = root / "çıktı"
            config = TerrainConfig(window_size=7, edge_method="sobel", sobel_threshold=120)
            process_image_file(path, output, terrain_config=config)
            settings = json.loads((output / "settings.json").read_text(encoding="utf-8"))
            self.assertEqual(settings["terrain"], asdict(config))
            with np.load(output / "terrain_features.npz", allow_pickle=False) as data:
                self.assertEqual(data["edge_density"].shape, (32, 40))
                self.assertEqual(data["local_variance"].dtype, np.float32)
                self.assertTrue(np.isfinite(data["local_variance"]).all())
                summary = json.loads((output / "terrain_summary.json").read_text(encoding="utf-8"))
                self.assertAlmostEqual(summary["edge_density_mean"], float(data["edge_density"].mean()))
            self.assertGreater((output / "terrain_comparison.png").stat().st_size, 0)
            # Ön işleme tek başına hâlâ aynı arayüzle çalışır.
            preprocess_only = root / "ön işleme"
            process_image_file(path, preprocess_only)
            self.assertFalse((preprocess_only / "terrain_features.npz").exists())


if __name__ == "__main__":
    unittest.main()
