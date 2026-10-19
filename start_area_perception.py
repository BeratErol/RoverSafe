"""Başlangıç analizine otomatik zemin ve isteğe bağlı nesne maskesi sağlar."""

from dataclasses import asdict, dataclass
from math import ceil, isfinite
from pathlib import Path
import cv2
import numpy as np
from obstacle_detection import ObstacleConfig, ObstacleDetectionError, YOLOObstacleDetector
from surface_classification import SurfaceConfig, SurfacePredictor


@dataclass
class StartPerception:
    ground: np.ndarray
    ground_score: np.ndarray
    obstacles: np.ndarray
    metadata: dict


def estimate_surfaces(data, manifest_path):
    spec = data.manifest
    base = Path(manifest_path).resolve().parent
    bgr = cv2.cvtColor(data.rgb, cv2.COLOR_RGB2BGR)
    mode = spec.get('ground_mode', 'reference')
    if mode == 'reference':
        if data.ground is None:
            raise ValueError('Referans zemin maskesi gerekli.')
        ground = data.ground.copy()
        score = np.full(ground.shape, np.nan, dtype='float32')
        metadata = {'ground_mode':'reference', 'source':spec['ground_mask_source']}
    else:
        settings = dict(spec['surface'])
        if 'model_path' not in settings:
            raise ValueError('Tahmin modunda yerel model_path açıkça verilmeli.')
        settings['model_path'] = str(base / settings['model_path'])
        config = SurfaceConfig(**settings)
        predictor = SurfacePredictor(config)
        selection = spec.get('surface_selection')
        if selection is not None and not isinstance(selection,dict):
            raise ValueError('surface_selection bir JSON nesnesi olmalı.')
        if selection is not None and (selection.get('split') != 'validation'
                or selection.get('model_sha256') != predictor.model_sha256
                or selection.get('threshold') != config.threshold):
            raise ValueError('Model veya eşik dondurulmuş doğrulama seçimiyle uyuşmuyor.')
        prediction = predictor.predict(bgr)
        score = prediction.probability
        if score.shape != data.rgb.shape[:2] or not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
            raise ValueError('Yüzey puan haritası geçersiz; öneri üretilmedi.')
        ground = (score >= config.threshold) & data.rgb_valid
        metadata = {'ground_mode':'predicted', 'source':spec['ground_mask_source'],
                    'surface':asdict(config), 'model_sha256':predictor.model_sha256,
                    'score_note':'Uncalibrated visual class score, not physical safety probability.'}
        if selection is not None:
            metadata['surface_selection'] = selection
    blocked = np.zeros(data.rgb.shape[:2], dtype=bool)
    metadata['obstacles'] = {'enabled':False, 'note':'No object detector executed; absence of objects is not established.'}
    if spec.get('obstacles') is not None:
        settings = dict(spec['obstacles'])
        if 'model_path' not in settings or 'margin' in settings:
            raise ValueError('Nesne modeli açıkça verilmeli; piksel margin yerine margin_m kullanın.')
        margin = settings.pop('margin_m', .2)
        if type(margin) not in (int,float) or not isfinite(margin) or margin < 0:
            raise ValueError('Nesne çevre payı metre cinsinden negatif olmayan sonlu sayı olmalı.')
        settings['model_path'] = str(base / settings['model_path'])
        settings['margin'] = ceil(margin / min(data.dx, data.dy))
        if isinstance(settings.get('classes'), list):
            settings['classes'] = tuple(settings['classes'])
        config = ObstacleConfig(**settings)
        detection = YOLOObstacleDetector(config).detect(bgr)
        blocked = detection.exclusion_mask
        if blocked.shape != ground.shape or blocked.dtype != bool:
            raise ObstacleDetectionError('Nesne maskesi görüntü ızgarasıyla uyuşmuyor.')
        metadata['obstacles'] = {'enabled':True, 'config':asdict(config), 'margin_m':margin,
                                 'model_sha256':detection.model_sha256,
                                 'ultralytics_version':detection.ultralytics_version,
                                 'inference_seconds':detection.inference_seconds,
                                 'detections':[asdict(item) for item in detection.detections],
                                 'note':'Generic detections may miss aerial objects; no detection is not proof of clearance.'}
    return StartPerception(ground, score, blocked, metadata)
