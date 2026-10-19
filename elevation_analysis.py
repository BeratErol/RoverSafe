"""Metre ölçekli yükseklikten eğim, düzlemden sapma ve yerel basamak çıkarır."""

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class ElevationResult:
    slope_deg: np.ndarray
    roughness_m: np.ndarray
    step_m: np.ndarray
    valid: np.ndarray
    window_size: int
    slope_upper_deg: np.ndarray | None = None
    roughness_upper_m: np.ndarray | None = None
    step_upper_m: np.ndarray | None = None
    error_bound_m: np.ndarray | None = None
    quality_invalid: np.ndarray | None = None

    def decision_maps(self):
        """Sınır varsa üst sınırları, yoksa nominal haritaları döndürür."""
        upper = (self.slope_upper_deg,self.roughness_upper_m,self.step_upper_m)
        if all(value is None for value in upper):
            return self.slope_deg,self.roughness_m,self.step_m
        if any(value is None for value in upper):
            raise ValueError('Üç arazi üst sınırı birlikte verilmelidir.')
        return upper


def analyze_elevation(elevation, dx, dy, window_size=5, *, quality_mask=None, error_bound_m=None):
    if not isinstance(elevation, np.ndarray) or elevation.ndim != 2 or not elevation.size:
        raise ValueError("Yükseklik boş olmayan iki boyutlu dizi olmalı.")
    if type(window_size) is not int or window_size < 3 or window_size % 2 == 0:
        raise ValueError("Yükseklik analiz penceresi tek ve en az 3 olmalı.")
    if not np.isfinite([dx, dy]).all() or min(dx, dy) <= 0:
        raise ValueError("Yatay çözünürlük pozitif metre olmalı.")
    if elevation.dtype.kind not in 'iuf':
        raise ValueError('Yükseklik gerçek sayılardan oluşmalı.')
    quality_known = np.ones(elevation.shape,dtype=bool)
    if quality_mask is not None:
        if not isinstance(quality_mask,np.ndarray) or quality_mask.dtype != bool or quality_mask.shape != elevation.shape:
            raise ValueError('Kalite maskesi yükseklikle aynı boyutta bool dizi olmalı.')
        quality_known &= quality_mask
    error = None
    if error_bound_m is not None:
        if type(error_bound_m) in (int,float):
            if not np.isfinite(error_bound_m) or error_bound_m < 0:
                raise ValueError('Hata sınırı negatif olmayan sonlu metre olmalı.')
            error = np.full(elevation.shape,error_bound_m,dtype='float64')
        elif isinstance(error_bound_m,np.ndarray) and error_bound_m.shape == elevation.shape and error_bound_m.dtype.kind in 'iuf':
            error = error_bound_m.astype('float64')
            if np.any(error[np.isfinite(error)] < 0):
                raise ValueError('Raster hata sınırları negatif olamaz.')
        else:
            raise ValueError('Hata sınırı skaler veya yükseklikle aynı boyutta gerçek sayı dizisi olmalı.')
        quality_known &= np.isfinite(error)
    known = np.isfinite(elevation) & quality_known
    kernel = np.ones((window_size,window_size),np.uint8)
    valid = cv2.erode(known.astype(np.uint8), kernel,
                      borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    quality_invalid = ~cv2.erode(quality_known.astype('uint8'),kernel,
                                 borderType=cv2.BORDER_CONSTANT,borderValue=1).astype(bool)
    # Büyük mutlak rakımlarda varyansın sayısal hassasiyetini korur.
    origin = float(np.median(elevation[known])) if known.any() else 0.0
    z = np.where(known, elevation - origin, 0).astype(np.float64)
    radius = window_size // 2
    x = np.arange(-radius, radius + 1, dtype=np.float64) * dx
    y = np.arange(-radius, radius + 1, dtype=np.float64) * dy
    kx = np.tile(x, (window_size, 1))
    ky = np.tile(y[:, None], (1, window_size))
    a = cv2.filter2D(z, -1, kx / (kx * kx).sum(), borderType=cv2.BORDER_CONSTANT)
    b = cv2.filter2D(z, -1, ky / (ky * ky).sum(), borderType=cv2.BORDER_CONSTANT)
    mean = cv2.boxFilter(z, -1, (window_size, window_size), borderType=cv2.BORDER_CONSTANT)
    square = cv2.boxFilter(z * z, -1, (window_size, window_size), borderType=cv2.BORDER_CONSTANT)
    residual = np.maximum(square - mean * mean - a * a * np.mean(x * x) - b * b * np.mean(y * y), 0)
    slope = np.degrees(np.arctan(np.hypot(a, b)))
    # 3x3 yükseklik açıklığı; engel yüksekliği için muhafazakâr bir gösterge.
    local = z
    step = cv2.dilate(local, np.ones((3, 3), np.uint8)) - cv2.erode(local, np.ones((3, 3), np.uint8))
    maps = [np.where(valid, value, np.nan).astype(np.float32) for value in (slope, np.sqrt(residual), step)]
    result = ElevationResult(*maps, valid, window_size,quality_invalid=quality_invalid)
    if error is not None:
        error = np.where(quality_known,error,0)
        # |da| <= sum |kx| e / sum kx²; korelasyon/normal dağılım varsayılmaz.
        da = cv2.filter2D(error,-1,np.abs(kx)/(kx*kx).sum(),borderType=cv2.BORDER_CONSTANT)
        db = cv2.filter2D(error,-1,np.abs(ky)/(ky*ky).sum(),borderType=cv2.BORDER_CONSTANT)
        slope_upper = np.degrees(np.arctan(np.hypot(np.abs(a)+da,np.abs(b)+db)))
        rms_error = np.sqrt(cv2.boxFilter(error*error,-1,(window_size,window_size),borderType=cv2.BORDER_CONSTANT))
        roughness_upper = np.sqrt(residual)+rms_error
        # OpenCV çift hassasiyette morfoloji destekler; alt/üst yükseklik aralığı.
        step_upper = cv2.dilate(z+error,np.ones((3,3),np.uint8))-cv2.erode(z-error,np.ones((3,3),np.uint8))
        bounds = [np.where(valid,value,np.nan).astype('float32') for value in (slope_upper,roughness_upper,step_upper)]
        if any(not np.isfinite(value[valid]).all() for value in bounds):
            raise ValueError('Hata sınırları sayısal olarak hesaplanabilir aralığın dışında.')
        # Nominal float32 yerel aralık yuvarlamasının altında sınır oluşturma.
        result.slope_upper_deg,result.roughness_upper_m,result.step_upper_m = [np.maximum(upper,nominal) for upper,nominal in zip(bounds,maps)]
        result.error_bound_m = np.where(quality_known,error,np.nan)
    return result
