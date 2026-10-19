"""Geçerli rover merkezlerini açıklanabilir ölçütlerle puanlar ve öneri seçer."""

from dataclasses import dataclass
from math import isfinite

import cv2
import numpy as np

from segmentation import SegmentationResult, find_valid_centers
from terrain_analysis import TerrainResult, VARIANCE_MAX


@dataclass(frozen=True)
class ScoringConfig:
    edge_weight: float = 0.3
    texture_weight: float = 0.3
    clearance_weight: float = 0.3
    area_weight: float = 0.1
    clearance_target: float = 40.0  # Pay dahil ayak izinden sonraki boşluk, px.
    area_target_multiple: float = 4.0
    min_score: float = 0.5

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError(f"{name} sonlu bir sayı olmalı.")
        weights = (self.edge_weight, self.texture_weight, self.clearance_weight, self.area_weight)
        if any(value < 0 for value in weights) or not isfinite(sum(weights)) or sum(weights) <= 0:
            raise ValueError("Puan ağırlıkları negatif olamaz ve toplamları pozitif, sonlu olmalı.")
        if self.clearance_target <= 0 or self.area_target_multiple < 1:
            raise ValueError("Boşluk hedefi pozitif, alan hedefi çarpanı en az 1 olmalı.")
        if not 0 <= self.min_score <= 1:
            raise ValueError("Asgari puan 0–1 aralığında olmalı.")


@dataclass(frozen=True)
class LandingPoint:
    region_id: int
    center: tuple[int, int]  # x, y
    footprint_box: tuple[int, int, int, int]
    margin_box: tuple[int, int, int, int]
    score: float
    components: dict[str, float]
    measurements: dict[str, float]


@dataclass(frozen=True)
class ScoringResult:
    score_map: np.ndarray  # Geçersiz konumlar NaN; puanlar 0–1.
    clearance_map: np.ndarray  # Pay dahil ayak izindeki en küçük uzaklık, px.
    ranked_regions: list[LandingPoint]
    best: LandingPoint | None
    status: str
    config: ScoringConfig


def score_landing(
    terrain: TerrainResult, candidates: SegmentationResult, config: ScoringConfig | None = None
) -> ScoringResult:
    """Yalnızca tam sığan merkezleri puanlar; eşik altındaysa öneri üretmez.

    Uzaklık, bilinen bir nesneye değil aday maskesinin dışına ve görüntü dışına
    göredir. Puan güvenlik olasılığı değildir. Ölçütler sabit hedeflerle 0–1'e
    dönüştürülür; sahneye göre min–max normalizasyonu uygulanmaz.
    """
    config = config if config is not None else ScoringConfig()
    shape = candidates.candidate_mask.shape
    if (
        len(shape) != 2 or not candidates.candidate_mask.size
        or candidates.candidate_mask.dtype != bool
        or candidates.valid_centers.shape != shape or candidates.valid_centers.dtype != bool
        or candidates.labels.shape != shape
        or terrain.edge_density.shape != shape or terrain.local_variance.shape != shape
    ):
        raise ValueError("Puanlama haritaları aynı boyutta olmalı; aday ve merkez maskeleri bool olmalı.")
    if (
        not np.isfinite(terrain.edge_density).all() or not np.isfinite(terrain.local_variance).all()
        or np.any((terrain.edge_density < 0) | (terrain.edge_density > 1))
        or np.any((terrain.local_variance < 0) | (terrain.local_variance > VARIANCE_MAX))
    ):
        raise ValueError("Puanlama geçerli aralıkta sonlu arazi özellikleri bekler.")

    seg = candidates.config
    # Puanlama, sığma koşulunu bozan bir merkezi yüksek puanla kabul edemez.
    valid = candidates.valid_centers & find_valid_centers(candidates.candidate_mask, seg)
    scores = np.full(shape, np.nan, dtype=np.float32)
    clearance_map = np.full(shape, np.nan, dtype=np.float32)
    if not valid.any():
        return ScoringResult(scores, clearance_map, [], None, "no_candidate", config)

    width = seg.rover_width + 2 * seg.safety_margin
    height = seg.rover_height + 2 * seg.safety_margin
    kernel = (width, height)
    mean_density = cv2.boxFilter(terrain.edge_density.astype(np.float64), -1, kernel,
                                borderType=cv2.BORDER_CONSTANT)
    mean_variance = cv2.boxFilter(terrain.local_variance.astype(np.float64), -1, kernel,
                                 borderType=cv2.BORDER_CONSTANT)
    # Sıfır dolgulu çerçeve, tüm görüntü uygun olsa bile sonlu sınır mesafesi sağlar.
    padded = np.pad(candidates.candidate_mask.astype(np.uint8), 1)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    clearance = cv2.erode(distance, np.ones((height, width), dtype=np.uint8),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0)
    clearance_map[valid] = clearance[valid]

    # Eşik sıfırsa geçerli ayak izindeki özellik de sıfırdır; bölme yapılmaz.
    edge_score = np.clip(1 - mean_density / seg.max_edge_density, 0, 1) if seg.max_edge_density else np.ones(shape)
    texture_score = np.clip(1 - mean_variance / seg.max_variance, 0, 1) if seg.max_variance else np.ones(shape)
    clearance_score = np.clip(clearance / config.clearance_target, 0, 1)
    weights = np.array([config.edge_weight, config.texture_weight, config.clearance_weight, config.area_weight], dtype=np.float64)
    weights /= weights.sum()

    ranked = []
    for region in candidates.regions:
        region_centers = valid & (candidates.labels == region.region_id)
        if not region_centers.any():
            continue
        area_score = min(region.area_px / (width * height * config.area_target_multiple), 1.0)
        values = (weights[0] * edge_score + weights[1] * texture_score
                  + weights[2] * clearance_score + weights[3] * area_score)
        scores[region_centers] = np.clip(values[region_centers], 0, 1)
        # Eşit puanda daha geniş boşluk, sonra yukarıdan aşağı/ soldan sağa sıra.
        eligible = region_centers & (scores == scores[region_centers].max())
        eligible &= clearance == clearance[eligible].max()
        y, x = map(int, np.argwhere(eligible)[0])
        point = LandingPoint(
            region.region_id, (x, y),
            (x - seg.rover_width // 2, y - seg.rover_height // 2, seg.rover_width, seg.rover_height),
            (x - width // 2, y - height // 2, width, height), float(scores[y, x]),
            {"edge": float(edge_score[y, x]), "texture": float(texture_score[y, x]),
             "clearance": float(clearance_score[y, x]), "area": float(area_score)},
            {"mean_edge_density": float(mean_density[y, x]), "mean_variance": float(mean_variance[y, x]),
             "clearance_px": float(clearance[y, x]), "region_area_px": float(region.area_px)},
        )
        ranked.append(point)
    ranked.sort(key=lambda point: (-point.score, -point.measurements["clearance_px"], point.center[1], point.center[0]))
    best = ranked[0] if ranked and ranked[0].score >= config.min_score else None
    status = "selected" if best is not None else "below_min_score" if ranked else "no_candidate"
    return ScoringResult(scores, clearance_map, ranked, best, status, config)
