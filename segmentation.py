"""Görsel özelliklerden aday bölgeler çıkarır ve roverın tam sığmasını kontrol eder."""

from dataclasses import dataclass
from math import isfinite

import cv2
import numpy as np

from terrain_analysis import VARIANCE_MAX, TerrainResult


@dataclass(frozen=True)
class SegmentationConfig:
    """Bütün boyutlar pikseldir; ayak izi görüntü eksenlerine paralel kabul edilir."""

    max_edge_density: float = 0.03
    max_variance: float = 250.0
    min_area: int = 2500
    opening_size: int = 3
    rover_width: int = 80
    rover_height: int = 60
    safety_margin: int = 10

    def __post_init__(self) -> None:
        for name, maximum in (("max_edge_density", 1), ("max_variance", VARIANCE_MAX)):
            value = getattr(self, name)
            if (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not isfinite(value) or not 0 <= value <= maximum
            ):
                raise ValueError(f"{name}, 0–{maximum} aralığında sonlu bir sayı olmalı.")
        for name in ("min_area", "opening_size", "rover_width", "rover_height", "safety_margin"):
            value = getattr(self, name)
            minimum = 0 if name == "safety_margin" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name}, en az {minimum} olan bir tam sayı olmalı.")
        if self.opening_size % 2 == 0:
            raise ValueError("Morfolojik açma boyutu tek sayı olmalı; kapatmak için 1 kullanın.")


@dataclass(frozen=True)
class CandidateRegion:
    region_id: int
    area_px: int
    bounding_box: tuple[int, int, int, int]  # x, y, genişlik, yükseklik
    valid_center_count: int


@dataclass(frozen=True)
class SegmentationResult:
    threshold_mask: np.ndarray
    cleaned_mask: np.ndarray
    candidate_mask: np.ndarray
    valid_centers: np.ndarray
    labels: np.ndarray
    regions: list[CandidateRegion]
    rejected_small: int
    rejected_no_fit: int
    config: SegmentationConfig


def find_valid_centers(mask: np.ndarray, config: SegmentationConfig) -> np.ndarray:
    """Rover + çevresindeki payın tamamen True maskeye sığdığı merkezleri bulur.

    İntegral görüntüyle her olası dikdörtgendeki uygun piksel sayısı hesaplanır.
    Görüntü dışı hiçbir konum kabul edilmez. Çift boyutta merkez, iki orta
    pikselden sağdaki/alttakidir; sol üst köşe merkez - boyut//2'dir.
    """
    if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.ndim != 2 or not mask.size:
        raise ValueError("Sığma kontrolü boş olmayan iki boyutlu bool maske bekler.")
    width = config.rover_width + 2 * config.safety_margin
    height = config.rover_height + 2 * config.safety_margin
    rows, columns = mask.shape
    centers = np.zeros_like(mask)
    if width > columns or height > rows:
        return centers

    integral = cv2.integral(mask.astype(np.uint8), sdepth=cv2.CV_64F)
    sums = (
        integral[height:, width:] - integral[:-height, width:]
        - integral[height:, :-width] + integral[:-height, :-width]
    )
    centers[height // 2:height // 2 + rows - height + 1,
            width // 2:width // 2 + columns - width + 1] = sums == width * height
    return centers


def extract_candidates(
    terrain: TerrainResult, config: SegmentationConfig | None = None,
    *, exclusion_mask: np.ndarray | None = None,
) -> SegmentationResult:
    """Eşikle, açma uygula, bağlı bileşenleri alan ve tam sığma koşuluyla ele."""
    config = config if config is not None else SegmentationConfig()
    density, variance = terrain.edge_density, terrain.local_variance
    edges = terrain.canny_edges if terrain.config.edge_method == "canny" else terrain.sobel_edges
    if (
        not isinstance(density, np.ndarray) or density.ndim != 2 or not density.size
        or not isinstance(variance, np.ndarray) or variance.shape != density.shape
        or not isinstance(edges, np.ndarray) or edges.shape != density.shape
    ):
        raise ValueError("Yoğunluk, varyans ve kenar haritaları aynı boyutta iki boyutlu diziler olmalı.")
    if (
        not np.isfinite(density).all() or not np.isfinite(variance).all()
        or np.any((density < 0) | (density > 1))
        or np.any((variance < 0) | (variance > VARIANCE_MAX))
        or not np.isin(edges, (0, 255)).all()
    ):
        raise ValueError("Arazi haritaları geçerli aralıkta sonlu değerler ve 0/255 kenarlar içermeli.")

    # Kenar pikseli, çevresindeki ortalama yoğunluk düşük olsa bile elenir.
    threshold = (density <= config.max_edge_density) & (variance <= config.max_variance) & (edges == 0)
    if exclusion_mask is not None:
        if not isinstance(exclusion_mask, np.ndarray) or exclusion_mask.dtype != bool or exclusion_mask.shape != density.shape:
            raise ValueError("Engel maskesi araziyle aynı boyutta bool bir dizi olmalı.")
        threshold &= ~exclusion_mask
    cleaned = threshold.copy()
    if config.opening_size > min(threshold.shape):
        cleaned[:] = False
    elif config.opening_size > 1:
        kernel = np.ones((config.opening_size, config.opening_size), dtype=np.uint8)
        cleaned = cv2.morphologyEx(
            threshold.astype(np.uint8), cv2.MORPH_OPEN, kernel,
            borderType=cv2.BORDER_CONSTANT, borderValue=0,
        ).astype(bool)
        # Hiçbir temizlik işlemi elenen pikseli yeniden aday yapmamalı.
        cleaned &= threshold

    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned.astype(np.uint8), connectivity=8)
    centers = find_valid_centers(cleaned, config)
    center_counts = np.bincount(labels[centers], minlength=count)
    mapping = np.zeros(count, dtype=np.int32)
    regions = []
    rejected_small = rejected_no_fit = 0
    for label in range(1, count):
        x, y, width, height, area = map(int, stats[label])
        if area < config.min_area:
            rejected_small += 1
        elif center_counts[label] == 0:
            rejected_no_fit += 1
        else:
            region_id = len(regions) + 1
            mapping[label] = region_id
            regions.append(CandidateRegion(region_id, area, (x, y, width, height), int(center_counts[label])))

    candidate_labels = mapping[labels]
    candidate_mask = candidate_labels > 0
    return SegmentationResult(
        threshold, cleaned, candidate_mask, centers & candidate_mask,
        candidate_labels, regions, rejected_small, rejected_no_fit, config,
    )
