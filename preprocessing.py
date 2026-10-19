"""Görüntüyü okur; gri seviye, gürültü azaltma ve isteğe bağlı CLAHE uygular."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessingResult:
    """Ara çıktılar, sonraki modüller ve karşılaştırma için korunur."""

    original_bgr: np.ndarray
    grayscale: np.ndarray
    denoised: np.ndarray
    processed: np.ndarray


def read_image(path: str | Path) -> np.ndarray:
    """Windows'ta Türkçe karakterli yolları da destekleyerek BGR görüntü okur."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Görüntü dosyası bulunamadı: {path}")
    encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    if encoded.size == 0:
        raise ValueError(f"Görüntü dosyası boş: {path}")
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Dosya geçerli veya desteklenen bir görüntü değil: {path}")
    return image


def preprocess_image(
    image_bgr: np.ndarray,
    filter_type: str = "gaussian",
    kernel_size: int = 5,
    use_clahe: bool = False,
) -> PreprocessingResult:
    """8 bit BGR kareyi işler; boyutları ve giriş görüntüsünü değiştirmez.

    Gauss genel yumuşatma, medyan tuz-biber gürültüsü karşılaştırması içindir.
    CLAHE, filtrelemeden sonra yerel kontrastı iyileştirmek için açılabilir.
    """
    if (
        not isinstance(image_bgr, np.ndarray)
        or image_bgr.dtype != np.uint8
        or image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
        or image_bgr.size == 0
    ):
        raise ValueError("Girdi boş olmayan, uint8 türünde üç kanallı BGR görüntü olmalı.")
    if filter_type not in {"gaussian", "median", "none"}:
        raise ValueError("Filtre gaussian, median veya none olmalı.")
    if type(kernel_size) is not int or kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("Filtre boyutu 3 veya daha büyük tek bir tam sayı olmalı.")

    original = image_bgr.copy()
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    if filter_type == "gaussian":
        denoised = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    elif filter_type == "median":
        denoised = cv2.medianBlur(gray, kernel_size)
    else:
        denoised = gray.copy()

    processed = denoised.copy()
    if use_clahe:
        processed = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
    return PreprocessingResult(original, gray, denoised, processed)
