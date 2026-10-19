"""Gri görüntüden kenar yoğunluğu ve yerel doku değişimi haritaları çıkarır.

Bu ölçüler görsel karmaşıklığı anlatır; gerçek zemin eğimini ölçmez.
"""

from dataclasses import dataclass
from math import isfinite

import cv2
import numpy as np


# 3x3 Sobel için her eksende 4*255 üst sınırından türetilen sabit gösterim sınırı.
SOBEL_DISPLAY_MAX = float(np.sqrt(2) * 4 * 255)
VARIANCE_MAX = 255.0**2 / 4


@dataclass(frozen=True)
class TerrainConfig:
    """Eşikler tüm görüntülerde aynı birimle uygulanır; görüntüye göre ölçeklenmez."""

    window_size: int = 25
    edge_method: str = "canny"
    canny_low: int = 50
    canny_high: int = 150
    sobel_threshold: float = 80.0

    def __post_init__(self) -> None:
        if type(self.window_size) is not int or self.window_size < 3 or self.window_size % 2 == 0:
            raise ValueError("Analiz penceresi 3 veya daha büyük tek bir tam sayı olmalı.")
        if self.edge_method not in {"canny", "sobel"}:
            raise ValueError("Kenar yöntemi canny veya sobel olmalı.")
        if (
            type(self.canny_low) is not int
            or type(self.canny_high) is not int
            or not 0 <= self.canny_low < self.canny_high <= 255
        ):
            raise ValueError("Canny eşikleri 0 <= alt < üst <= 255 koşulunu sağlamalı.")
        if (
            isinstance(self.sobel_threshold, bool)
            or not isinstance(self.sobel_threshold, (int, float))
            or not isfinite(self.sobel_threshold)
            or not 0 < self.sobel_threshold <= SOBEL_DISPLAY_MAX
        ):
            raise ValueError(f"Sobel eşiği 0'dan büyük ve en fazla {SOBEL_DISPLAY_MAX:.2f} olmalı.")


@dataclass(frozen=True)
class TerrainResult:
    """İkili kenarlar 0/255; yoğunluk 0–1; varyans gri seviye karesi birimindedir."""

    sobel_magnitude: np.ndarray
    sobel_edges: np.ndarray
    canny_edges: np.ndarray
    edge_density: np.ndarray
    local_variance: np.ndarray
    config: TerrainConfig


def analyze_terrain(gray: np.ndarray, config: TerrainConfig | None = None) -> TerrainResult:
    """Ön işlenmiş uint8 gri kareyi analiz eder; girdi ve boyutlar korunur.

    Yerel istatistikler pencere içinde hesaplanır. Sınırda BORDER_REFLECT_101
    kullanılır; sıfır dolguyla yapay kenar veya düşük ortalama oluşturulmaz.
    """
    if (
        not isinstance(gray, np.ndarray)
        or gray.dtype != np.uint8
        or gray.ndim != 2
        or gray.size == 0
    ):
        raise ValueError("Arazi analizi boş olmayan uint8 gri seviye görüntü bekler.")
    config = config if config is not None else TerrainConfig()
    if config.window_size > min(gray.shape):
        raise ValueError("Analiz penceresi görüntünün kısa kenarından büyük olamaz.")

    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    magnitude = cv2.magnitude(dx, dy)
    sobel_edges = (magnitude >= config.sobel_threshold).astype(np.uint8) * 255
    canny_edges = cv2.Canny(
        gray, config.canny_low, config.canny_high, apertureSize=3, L2gradient=True
    )
    selected_edges = canny_edges if config.edge_method == "canny" else sobel_edges
    kernel = (config.window_size, config.window_size)
    density = cv2.boxFilter(
        selected_edges.astype(np.float32) / 255.0,
        -1, kernel, normalize=True, borderType=cv2.BORDER_REFLECT_101,
    )

    # E[X²] - E[X]²; karesini uint8 üzerinde almak taşmaya yol açardı.
    values = gray.astype(np.float64)
    mean = cv2.boxFilter(values, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    mean_square = cv2.boxFilter(values * values, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    variance = np.clip(mean_square - mean * mean, 0, VARIANCE_MAX).astype(np.float32)
    return TerrainResult(
        magnitude, sobel_edges, canny_edges,
        np.clip(density, 0, 1), variance, config,
    )
