"""Yerel YOLO modeliyle nesneleri bulur ve paylı bir engel maskesi oluşturur."""

from dataclasses import dataclass
import hashlib
from importlib.metadata import version
from math import ceil, floor, isfinite
from pathlib import Path
from time import perf_counter

import numpy as np


class ObstacleDetectionError(RuntimeError):
    """Algılayıcı hatası engel yokmuş gibi yorumlanmadan üst katmana aktarılır."""


@dataclass(frozen=True)
class ObstacleConfig:
    model_path: str = "models/yolo11n.pt"
    confidence: float = 0.25
    iou: float = 0.7
    margin: int = 15
    image_size: int = 640
    device: str = "cpu"
    classes: tuple[str, ...] | None = None  # None: modelin desteklediği tüm sınıflar.

    def __post_init__(self) -> None:
        if not isinstance(self.model_path, str) or not self.model_path.strip():
            raise ValueError("Yerel model dosyası yolu boş olamaz.")
        for name in ("confidence", "iou"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name}, 0'dan büyük ve en fazla 1 olan sonlu bir sayı olmalı.")
        if type(self.margin) is not int or self.margin < 0:
            raise ValueError("Engel payı negatif olmayan bir tam sayı olmalı.")
        if type(self.image_size) is not int or self.image_size < 32 or self.image_size % 32:
            raise ValueError("YOLO görüntü boyutu en az 32 ve 32'nin katı olmalı.")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("YOLO cihazı boş olamaz; örneğin cpu veya 0 kullanın.")
        if self.classes is not None and (
            not isinstance(self.classes, tuple) or not self.classes
            or any(not isinstance(name, str) or not name.strip() for name in self.classes)
            or len(set(self.classes)) != len(self.classes)
        ):
            raise ValueError("Engel sınıfları benzersiz ve boş olmayan sınıf adlarından oluşan tuple olmalı.")


@dataclass(frozen=True)
class Obstacle:
    class_id: int
    class_name: str
    confidence: float
    bounding_box: tuple[int, int, int, int]  # x, y, genişlik, yükseklik
    expanded_box: tuple[int, int, int, int]


@dataclass(frozen=True)
class ObstacleResult:
    exclusion_mask: np.ndarray  # True: aday üretiminde dışlanacak piksel.
    detections: list[Obstacle]
    inference_seconds: float
    model_path: str
    model_sha256: str
    ultralytics_version: str
    config: ObstacleConfig


def boxes_to_mask(
    image_shape: tuple[int, int], boxes: np.ndarray, class_ids: np.ndarray,
    confidences: np.ndarray, names: dict[int, str], config: ObstacleConfig,
) -> tuple[np.ndarray, list[Obstacle]]:
    """Orijinal görüntüdeki xyxy kutularını dışa yuvarlar, pay ekler ve sınırlar."""
    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError("Görüntü boyutları pozitif olmalı.")
    if boxes.ndim != 2 or boxes.shape[1] != 4 or class_ids.shape != (len(boxes),) or confidences.shape != (len(boxes),):
        raise ObstacleDetectionError("YOLO kutu, sınıf ve güven dizileri tutarsız.")
    if (
        not np.isfinite(boxes).all() or not np.isfinite(class_ids).all() or not np.isfinite(confidences).all()
        or np.any((confidences < 0) | (confidences > 1)) or np.any(class_ids != np.floor(class_ids))
        or np.any(boxes[:, 2:] <= boxes[:, :2])
    ):
        raise ObstacleDetectionError("YOLO geçersiz koordinat, sınıf veya güven değeri döndürdü.")
    mask = np.zeros((height, width), dtype=bool)
    obstacles = []
    for box, raw_id, confidence in zip(boxes, class_ids, confidences):
        class_id = int(raw_id)
        if class_id not in names:
            raise ObstacleDetectionError(f"Model çıktısındaki sınıf kimliği bilinmiyor: {class_id}")
        name = names[class_id]
        if confidence < config.confidence or (config.classes is not None and name not in config.classes):
            continue
        x1, y1 = max(0, min(width, floor(box[0]))), max(0, min(height, floor(box[1])))
        x2, y2 = max(0, min(width, ceil(box[2]))), max(0, min(height, ceil(box[3])))
        if x2 <= x1 or y2 <= y1:
            continue  # Tamamen görüntü dışındaki kutu görüntüde bir engel oluşturmaz.
        left, top = max(0, x1 - config.margin), max(0, y1 - config.margin)
        right, bottom = min(width, x2 + config.margin), min(height, y2 + config.margin)
        mask[top:bottom, left:right] = True
        obstacles.append(Obstacle(class_id, name, float(confidence), (x1, y1, x2 - x1, y2 - y1),
                                  (left, top, right - left, bottom - top)))
    return mask, obstacles


class YOLOObstacleDetector:
    """Modeli bir kez yükler; aynı nesneyle sonraki video kareleri de işlenebilir."""

    def __init__(self, config: ObstacleConfig | None = None):
        self.config = config if config is not None else ObstacleConfig()
        model_path = Path(self.config.model_path)
        if not model_path.is_file():
            raise ObstacleDetectionError(
                f"YOLO modeli bulunamadı: {model_path}. Önce scripts/download_yolo_model.py çalıştırın."
            )
        try:
            from ultralytics import YOLO  # Temel akış bu isteğe bağlı paketi yüklemez.
        except Exception as error:
            raise ObstacleDetectionError(
                f"YOLO bağımlılıkları yüklenemedi; pip install -r requirements-yolo.txt ile kurulumu kontrol edin: {error}"
            ) from error
        try:
            self.model = YOLO(str(model_path.resolve()), task="detect")
            if self.model.task != "detect":
                raise ValueError("Nesne tespit modeli gerekli; sınıflandırma/segmentasyon modeli verilmiş.")
            self.names = dict(self.model.names)
            if self.config.classes is not None:
                missing = set(self.config.classes) - set(self.names.values())
                if missing:
                    raise ValueError(f"Modelde bulunmayan sınıf adları: {', '.join(sorted(missing))}")
            self.class_ids = [key for key, name in self.names.items() if name in self.config.classes] if self.config.classes else None
            self.model_path = str(model_path.resolve())
            self.model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
            self.ultralytics_version = version("ultralytics")
        except Exception as error:
            raise ObstacleDetectionError(f"YOLO modeli yüklenemedi: {error}") from error

    def detect(self, image_bgr: np.ndarray) -> ObstacleResult:
        """Özgün BGR kareyi işler; kutular orijinal görüntü koordinatlarındadır."""
        if (
            not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8
            or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or not image_bgr.size
        ):
            raise ValueError("YOLO boş olmayan uint8 BGR görüntü bekler.")
        started = perf_counter()
        try:
            results = self.model.predict(
                source=image_bgr, conf=self.config.confidence, iou=self.config.iou,
                imgsz=self.config.image_size, device=self.config.device,
                classes=self.class_ids, verbose=False, save=False,
            )
            if len(results) != 1 or tuple(results[0].orig_shape) != image_bgr.shape[:2] or results[0].boxes is None:
                raise ValueError("YOLO bir kare için geçerli kutu sonucu döndürmedi.")
            boxes = results[0].boxes
            mask, detections = boxes_to_mask(
                image_bgr.shape[:2], boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(),
                boxes.conf.cpu().numpy(), self.names, self.config,
            )
        except Exception as error:
            raise ObstacleDetectionError(f"YOLO tahmini başarısız; öneri üretilmedi: {error}") from error
        return ObstacleResult(mask, detections, perf_counter() - started, self.model_path,
                              self.model_sha256, self.ultralytics_version, self.config)
