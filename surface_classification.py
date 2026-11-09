"""Renk ve yerel doku özelliklerinden öğrenilmiş, isteğe bağlı yüzey filtresi."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import cv2
import numpy as np

FEATURE_VERSION = "color_texture_v1"
MULTISCALE_VERSION = "color_texture_multiscale_v2"
FEATURE_COUNTS = {FEATURE_VERSION:14, MULTISCALE_VERSION:35}
FEATURE_BORDERS = {FEATURE_VERSION:8, MULTISCALE_VERSION:20}
FEATURE_COUNTS['resnet18_context_v3'] = 419
FEATURE_BORDERS['resnet18_context_v3'] = 20


@dataclass(frozen=True)
class SurfaceConfig:
    model_path: str = "models/surface.joblib"
    threshold: float = 0.75
    grid_step: int = 4
    expected_viewpoint: str = 'any'
    encoder_path: str | None = None

    def __post_init__(self):
        if not isinstance(self.model_path, str) or not self.model_path.strip():
            raise ValueError("Yüzey modeli yolu boş olamaz.")
        if type(self.threshold) not in (float, int) or not np.isfinite(self.threshold) or not 0 < self.threshold <= 1:
            raise ValueError("Yüzey eşiği (0, 1] aralığında olmalı.")
        if type(self.grid_step) is not int or not 1 <= self.grid_step <= 16:
            raise ValueError("Özellik örnekleme adımı 1–16 olmalı.")
        if self.expected_viewpoint not in ('any','aerial','ground'):
            raise ValueError('Beklenen bakış açısı any, aerial veya ground olmalı.')
        if self.encoder_path is not None and (not isinstance(self.encoder_path,str) or not self.encoder_path.strip()):
            raise ValueError('Kodlayıcı yolu boş olamaz.')


def feature_grid(image, step=4, version=FEATURE_VERSION):
    """Konum kullanmadan BGR, HSV, Lab, yeşillik, varyans ve gradyan çıkarır."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Özellik girdisi uint8 BGR görüntü olmalı.")
    if type(step) is not int or step < 1:
        raise ValueError("Örnekleme adımı pozitif tam sayı olmalı.")
    if version not in FEATURE_COUNTS:
        raise ValueError('Bilinmeyen yüzey özellik sürümü.')
    if version == 'resnet18_context_v3':
        raise ValueError('Bağlam özellikleri ContextEncoder ile hesaplanmalıdır.')
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
    sampled = np.ascontiguousarray(features[::step, ::step], dtype=np.float32)
    if version == FEATURE_VERSION:
        return sampled
    # Üç ölçek renk bağlamı ve doku: koordinat veya referans etiketi kullanılmaz.
    extra = [sampled]
    gradient = cv2.magnitude(dx,dy)
    for size in (5,17,33):
        mean_color = cv2.boxFilter(bgr,-1,(size,size),borderType=cv2.BORDER_REFLECT_101)
        variance_color = np.maximum(cv2.boxFilter(bgr*bgr,-1,(size,size),borderType=cv2.BORDER_REFLECT_101)-mean_color*mean_color,0)
        extra += [mean_color[::step,::step].copy(),np.sqrt(variance_color[::step,::step]),
                  cv2.boxFilter(gradient,-1,(size,size),borderType=cv2.BORDER_REFLECT_101)[::step,::step,None].copy()]
    return np.ascontiguousarray(np.concatenate(extra,axis=2),dtype=np.float32)


@dataclass
class SurfaceResult:
    probability: np.ndarray
    exclusion_mask: np.ndarray
    model_sha256: str
    encoder_sha256: str | None = None
    feature_version: str | None = None
    viewpoint: str | None = None


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
        self.delegate = None
        if path.suffix.lower() == '.pth':
            from dense_surface import DenseSurfacePredictor
            self.delegate = DenseSurfacePredictor(config)
            for name in ('feature_version','viewpoint','model_sha256','encoder','config'):
                setattr(self,name,getattr(self.delegate,name))
            return
        bundle = joblib.load(path)
        if bundle.get("feature_version") not in FEATURE_COUNTS:
            raise ValueError("Yüzey modeli özellik sürümü uyuşmuyor.")
        self.feature_version = bundle['feature_version']
        self.viewpoint = bundle.get('viewpoint','unspecified')
        if config.expected_viewpoint != 'any' and self.viewpoint not in (config.expected_viewpoint,'unspecified'):
            raise ValueError('Yüzey modeli bakış açısı bu iş akışıyla uyuşmuyor.')
        self.model = bundle["model"]
        if list(self.model.classes_) != [0, 1] or self.model.n_features_in_ != FEATURE_COUNTS[self.feature_version]:
            raise ValueError("Yüzey modeli sınıfları veya özellik sayısı sürümle uyuşmuyor.")
        self.config = config
        self.model_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.encoder = None
        if self.feature_version == 'resnet18_context_v3':
            if config.encoder_path is None or not isinstance(bundle.get('encoder_sha256'),str):
                raise ValueError('Bağlam modeli için encoder_path ve dondurulmuş kodlayıcı özeti gerekli.')
            from context_features import ContextEncoder
            self.encoder = ContextEncoder(config.encoder_path,bundle['encoder_sha256'])

    def predict(self, image):
        if self.delegate is not None:
            return self.delegate.predict(image)
        features = (self.encoder.features(image,self.config.grid_step) if self.encoder else
                    feature_grid(image, self.config.grid_step,self.feature_version))
        values = self.model.predict_proba(features.reshape(-1, features.shape[-1]))[:, 1]
        coarse = values.reshape(features.shape[:2]).astype(np.float32)
        # Örnekleme sol üst köşeye hizalıdır; resize merkez kayması oluşturmaz.
        probability = coarse.repeat(self.config.grid_step, 0).repeat(self.config.grid_step, 1)[:image.shape[0], :image.shape[1]]
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError("Model geçersiz yüzey puanı üretti.")
        return SurfaceResult(probability, probability < self.config.threshold, self.model_sha256,
                             self.encoder.sha256 if self.encoder else None,self.feature_version,self.viewpoint)
