"""Renk ve yerel doku özelliklerinden öğrenilmiş, isteğe bağlı yüzey filtresi."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import cv2
import numpy as np

FEATURE_VERSION = "color_texture_v1"


@dataclass(frozen=True)
class SurfaceConfig:
    model_path: str = "models/surface.joblib"
    threshold: float = 0.75
    grid_step: int = 4

    def __post_init__(self):
        if not isinstance(self.model_path, str) or not self.model_path.strip():
            raise ValueError("Yüzey modeli yolu boş olamaz.")
        if type(self.threshold) not in (float, int) or not np.isfinite(self.threshold) or not 0 < self.threshold <= 1:
            raise ValueError("Yüzey eşiği (0, 1] aralığında olmalı.")
        if type(self.grid_step) is not int or not 1 <= self.grid_step <= 16:
            raise ValueError("Özellik örnekleme adımı 1–16 olmalı.")


def feature_grid(image, step=4):
    """Konum kullanmadan BGR, HSV, Lab, yeşillik, varyans ve gradyan çıkarır."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Özellik girdisi uint8 BGR görüntü olmalı.")
    if type(step) is not int or step < 1:
        raise ValueError("Örnekleme adımı pozitif tam sayı olmalı.")
    bgr = image.astype(np.float32) / 255
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32) / 255
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    mean = cv2.boxFilter(gray, -1, (9, 9), borderType=cv2.BORDER_REFLECT_101)
    variance = np.maximum(cv2.boxFilter(gray * gray, -1, (9, 9), borderType=cv2.BORDER_REFLECT_101) - mean * mean, 0)
    dx, dy = (cv2.Sobel(gray, cv2.CV_32F, x, y, ksize=3) for x, y in ((1, 0), (0, 1)))
    hue = hsv[..., 0] * (2 * np.pi / 180)
    features = np.dstack((bgr, np.sin(hue), np.cos(hue), hsv[..., 1:] / 255, lab,
                          2 * bgr[..., 1] - bgr[..., 0] - bgr[..., 2], mean,
                          np.sqrt(variance), cv2.magnitude(dx, dy)))
    return np.ascontiguousarray(features[::step, ::step], dtype=np.float32)


@dataclass
class SurfaceResult:
    probability: np.ndarray
    exclusion_mask: np.ndarray
    model_sha256: str


class SurfacePredictor:
    def __init__(self, config: SurfaceConfig):
        # joblib yalnızca bu projede eğitilmiş güvenilir yerel model için kullanılır.
        try:
            import joblib
        except ImportError as error:
            raise ValueError("Yüzey filtresi için requirements-surface.txt bağımlılıklarını kurun.") from error
        path = Path(config.model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Yüzey modeli yok: {path}. Önce scripts/train_surface.py çalıştırın.")
        bundle = joblib.load(path)
        if bundle.get("feature_version") != FEATURE_VERSION:
            raise ValueError("Yüzey modeli özellik sürümü uyuşmuyor.")
        self.model = bundle["model"]
        if list(self.model.classes_) != [0, 1] or self.model.n_features_in_ != 14:
            raise ValueError("Yüzey modeli iki sınıflı 14 özellikli model olmalı.")
        self.config = config
        self.model_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    def predict(self, image):
        features = feature_grid(image, self.config.grid_step)
        values = self.model.predict_proba(features.reshape(-1, features.shape[-1]))[:, 1]
        coarse = values.reshape(features.shape[:2]).astype(np.float32)
        # Örnekleme sol üst köşeye hizalıdır; resize merkez kayması oluşturmaz.
        probability = coarse.repeat(self.config.grid_step, 0).repeat(self.config.grid_step, 1)[:image.shape[0], :image.shape[1]]
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError("Model geçersiz yüzey puanı üretti.")
        return SurfaceResult(probability, probability < self.config.threshold, self.model_sha256)
