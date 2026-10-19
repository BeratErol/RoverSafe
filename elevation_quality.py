"""Yükseklik doğruluğunun bilinmediği ve sınırlı hata varsayılan durumları ayırır."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ElevationQualityConfig:
    mode: str = 'nominal'
    vertical_error_bound_m: float | None = None
    error_bound_raster: str | None = None
    validity_mask: str | None = None
    source: str | None = None

    def __post_init__(self):
        if self.mode not in ('nominal','bounded_error'):
            raise ValueError('Yükseklik kalite modu nominal veya bounded_error olmalı.')
        for name in ('error_bound_raster','validity_mask','source'):
            value = getattr(self,name)
            if value is not None and (not isinstance(value,str) or not value.strip()):
                raise ValueError(f'{name} boş olmayan metin olmalı.')
        bound = self.vertical_error_bound_m
        if bound is not None and (type(bound) not in (int,float) or not isfinite(bound) or bound < 0):
            raise ValueError('Mutlak düşey hata sınırı negatif olmayan sonlu metre olmalı.')
        has_bound = bound is not None
        has_raster = self.error_bound_raster is not None
        if self.mode == 'nominal' and (has_bound or has_raster):
            raise ValueError('Nominal modda hata sınırı verilmez; bounded_error seçin.')
        if self.mode == 'bounded_error' and (has_bound == has_raster or self.source is None):
            raise ValueError('Sınırlı hata modu tek bir skaler/raster sınır ve kaynak açıklaması gerektirir.')
        if self.validity_mask is not None and self.source is None:
            raise ValueError('Yükseklik kalite maskesinin kaynak açıklaması gerekli.')
