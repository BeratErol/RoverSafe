"""Ön işlemenin veri bütünlüğünü, gürültü etkisini ve dosya hatalarını kontrol eder."""

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from pipeline import process_image_file
from preprocessing import preprocess_image, read_image
from visualization import save_image


class PreprocessingTests(unittest.TestCase):
    def test_preserves_input_and_dimensions(self):
        image = np.random.default_rng(4).integers(0, 256, (32, 48, 3), dtype=np.uint8)
        original = image.copy()
        for filter_type in ("gaussian", "median", "none"):
            with self.subTest(filter=filter_type):
                result = preprocess_image(image, filter_type, use_clahe=True)
                self.assertEqual(result.processed.shape, (32, 48))
                self.assertEqual(result.processed.dtype, np.uint8)
                np.testing.assert_array_equal(image, original)
                np.testing.assert_array_equal(result.original_bgr, original)

    def test_bgr_channel_order(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[:, :, 2] = 255  # BGR'de üçüncü kanal kırmızıdır.
        result = preprocess_image(image, "none")
        self.assertEqual(int(result.grayscale[0, 0]), 76)
        np.testing.assert_array_equal(result.grayscale, result.processed)

    def test_gaussian_reduces_noise_on_uniform_surface(self):
        noise = np.random.default_rng(42).normal(0, 20, (64, 64))
        gray = np.clip(128 + noise, 0, 255).astype(np.uint8)
        result = preprocess_image(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        self.assertLess(result.denoised.var(), gray.var() / 2)

    def test_median_removes_isolated_impulse(self):
        image = np.full((16, 16, 3), 100, dtype=np.uint8)
        image[8, 8] = 255
        result = preprocess_image(image, "median", 3)
        self.assertEqual(int(result.denoised[8, 8]), 100)

    def test_rejects_invalid_parameters_and_arrays(self):
        valid = np.zeros((8, 8, 3), dtype=np.uint8)
        for size in (0, 1, 4, -3, 3.5, True):
            with self.subTest(kernel=size), self.assertRaises(ValueError):
                preprocess_image(valid, kernel_size=size)
        with self.assertRaises(ValueError):
            preprocess_image(valid, "invalid")
        for image in (None, np.zeros((8, 8)), np.empty((0, 8, 3), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                preprocess_image(image)

    def test_unicode_file_round_trip_and_pipeline_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "Türkçe görüntüler"
            path = root / "örnek.png"
            image = np.full((24, 32, 3), (30, 80, 180), dtype=np.uint8)
            save_image(path, image)
            np.testing.assert_array_equal(read_image(path), image)
            output = root / "çıktılar"
            comparison = process_image_file(path, output, "median", 3, True)
            self.assertGreater(comparison.stat().st_size, 0)
            for name in ("original.png", "grayscale.png", "denoised.png", "processed.png"):
                self.assertEqual(read_image(output / name).shape, image.shape)
            self.assertTrue((output / "settings.json").is_file())

    def test_missing_empty_and_corrupt_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.png"
            with self.assertRaises(FileNotFoundError):
                read_image(path)
            for content in (b"", b"not an image"):
                path.write_bytes(content)
                with self.assertRaises(ValueError):
                    read_image(path)


if __name__ == "__main__":
    unittest.main()
