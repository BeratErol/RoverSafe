"""JSON deney ayarlarını doğrular ve ortak kare işleme seçeneklerine çevirir."""

from dataclasses import asdict
import json
from pathlib import Path

from landing_score import ScoringConfig
from obstacle_detection import ObstacleConfig
from segmentation import SegmentationConfig
from terrain_analysis import TerrainConfig
from surface_classification import SurfaceConfig

ROOT = Path(__file__).resolve().parent


def load_config(path: str | Path) -> dict:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return parse_config(data)


def parse_config(data: dict) -> dict:
    if not isinstance(data, dict) or set(data) - {"preprocessing", "terrain", "segmentation", "scoring", "obstacles", "surface"}:
        raise ValueError("Yapılandırma yalnızca bilinen işlem aşamalarını içeren bir nesne olmalı.")
    prep = data.get("preprocessing", {})
    if not isinstance(prep, dict) or set(prep) - {"filter_type", "kernel_size", "use_clahe"}:
        raise ValueError("Bilinmeyen ön işleme ayarı.")
    options = {"filter_type": "gaussian", "kernel_size": 5, "use_clahe": False, **prep}
    if options["filter_type"] not in ("gaussian", "median", "none"):
        raise ValueError("Geçersiz filtre türü.")
    size = options["kernel_size"]
    if type(size) is not int or size < 3 or size % 2 == 0 or type(options["use_clahe"]) is not bool:
        raise ValueError("Filtre boyutu tek ve en az 3; CLAHE boolean olmalı.")
    for section, constructor in (("terrain", TerrainConfig), ("segmentation", SegmentationConfig),
                                  ("scoring", ScoringConfig), ("obstacles", ObstacleConfig), ("surface", SurfaceConfig)):
        values = data.get(section)
        if values is not None and not isinstance(values, dict):
            raise ValueError(f"{section} bir nesne veya null olmalı.")
        if section == "obstacles" and values is not None and "classes" in values:
            if values["classes"] is not None and not isinstance(values["classes"], list):
                raise ValueError("classes bir sınıf adı listesi olmalı.")
            values = dict(values, classes=tuple(values["classes"]) if values["classes"] is not None else None)
        try:
            options[{"terrain": "terrain_config", "segmentation": "segmentation_config",
                     "scoring": "scoring_config", "obstacles": "obstacle_config", "surface": "surface_config"}[section]] = (
                constructor(**values) if values is not None else None)
        except TypeError as error:
            raise ValueError(f"{section} ayarları geçersiz: {error}") from error
    return options


def cli_defaults(path: str | Path) -> dict:
    """Mevcut komut satırı seçeneklerini koruyarak JSON değerlerini varsayılan yapar."""
    options = load_config(path)
    defaults = {"filter": options["filter_type"], "kernel_size": options["kernel_size"], "clahe": options["use_clahe"]}
    for section, flag in (("terrain", "terrain"), ("segmentation", "candidates"), ("scoring", "score")):
        config = options[section + "_config"]
        if config is not None:
            defaults.update(asdict(config))
            defaults[flag] = True
    config = options["obstacle_config"]
    if config is not None:
        defaults["obstacles"] = True
        mapping = {"model_path": "yolo_model", "confidence": "yolo_confidence", "iou": "yolo_iou",
                   "margin": "obstacle_margin", "image_size": "yolo_image_size", "device": "yolo_device", "classes": "obstacle_classes"}
        defaults.update({mapping[key]: value for key, value in asdict(config).items()})
    config = options["surface_config"]
    if config is not None:
        defaults.update(surface=True, surface_model=config.model_path, surface_threshold=config.threshold,
                        surface_grid_step=config.grid_step)
    return defaults
