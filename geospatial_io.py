"""Aynı metre tabanlı ızgaradaki görüntü, yükseklik ve görev maskelerini okur."""

from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np
import rasterio
from elevation_quality import ElevationQualityConfig


@dataclass
class GeoInputs:
    rgb: np.ndarray
    elevation: np.ndarray
    ground: np.ndarray | None
    allowed: np.ndarray
    start_zone: np.ndarray
    transform: object
    crs: object
    dx: float
    dy: float
    manifest: dict
    rgb_valid: np.ndarray
    quality_config: ElevationQualityConfig = field(default_factory=ElevationQualityConfig)
    elevation_validity: np.ndarray | None = None
    elevation_error_bound_m: float | np.ndarray | None = None

    def world(self, point):
        return list(self.transform * (point[0] + 0.5, point[1] + 0.5))


def load_manifest(path):
    path = Path(path)
    spec = json.loads(path.read_text(encoding='utf-8-sig'))
    quality = ElevationQualityConfig(**spec.get('elevation_quality',{}))
    mode = spec.get('ground_mode', 'reference')
    if mode not in ('reference', 'predicted'):
        raise ValueError('ground_mode reference veya predicted olmalı.')
    if mode == 'predicted' and ('ground_mask' in spec or not isinstance(spec.get('surface'), dict)):
        raise ValueError('Tahmin modunda ground_mask verilmez; surface ayarları zorunludur.')
    if mode == 'reference' and 'surface' in spec:
        raise ValueError('Referans modunda surface ayarı kullanılmaz; modu açıkça seçin.')
    if spec.get('elevation_unit') != 'm' or spec.get('elevation_kind') not in ('DSM', 'DTM'):
        raise ValueError('Yükseklik birimi m, türü DSM veya DTM olarak belirtilmeli.')
    for key in ('ground_mask_source', 'mission_source', 'dataset_source'):
        if not isinstance(spec.get(key), str) or not spec[key].strip():
            raise ValueError(f'{key} kaynak açıklaması gerekli.')
    arrays = {}
    reference = None
    keys = ['rgb', 'elevation', 'allowed_mask', 'start_zone']
    if mode == 'reference':
        keys.append('ground_mask')
    paths = {key:spec[key] for key in keys}
    if quality.validity_mask is not None:
        paths['elevation_validity'] = quality.validity_mask
    if quality.error_bound_raster is not None:
        paths['elevation_error_bound'] = quality.error_bound_raster
    for key,filename in paths.items():
        with rasterio.open(path.parent / filename) as src:
            grid = (src.width, src.height, src.crs, src.transform)
            if reference is None:
                reference = grid
                transform, crs = src.transform, src.crs
                if crs is None or not crs.is_projected or crs.linear_units != 'metre':
                    raise ValueError('Yatay koordinatlar projeksiyonlu ve metre biriminde olmalı.')
                if transform.b != 0 or transform.d != 0 or transform.a <= 0 or transform.e >= 0:
                    raise ValueError('Kuzey yönlü, döndürülmemiş raster gerekli; önce yeniden projekte edin.')
            elif grid != reference:
                raise ValueError(f'{key}: CRS, dönüşüm ve boyutlar aynı ızgarada olmalı.')
            if key == 'rgb':
                if src.count < 3 or src.dtypes[0] != 'uint8':
                    raise ValueError('RGB girdisi en az üç uint8 bant içermeli.')
                arrays[key] = src.read([1, 2, 3]).transpose(1, 2, 0)
                rgb_valid = np.all(src.read_masks([1, 2, 3]) > 0, axis=0)
            elif key in ('elevation','elevation_error_bound'):
                if src.count != 1 or src.scales != (1.0,) or src.offsets != (0.0,):
                    raise ValueError('Yükseklik tek bantlı ve fiziksel metre değerlerinde olmalı.')
                arrays[key] = src.read(1, masked=True).astype('float64').filled(np.nan)
                if key == 'elevation_error_bound' and np.any(arrays[key][np.isfinite(arrays[key])] < 0):
                    raise ValueError('Raster hata sınırları negatif olamaz.')
            else:
                if src.count != 1:
                    raise ValueError('Maskeler tek bantlı olmalı.')
                value = src.read(1, masked=True)
                if not np.isin(value.compressed(), [0, 1, 255]).all():
                    raise ValueError('Maskeler yalnızca 0, 1 veya 255 içermeli.')
                arrays[key] = value.filled(0) > 0
    ground = arrays.get('ground_mask')
    if ground is not None:
        ground &= rgb_valid
    return GeoInputs(arrays['rgb'], arrays['elevation'], ground,
                     arrays['allowed_mask'], arrays['start_zone'], transform, crs,
                     transform.a, -transform.e, spec, rgb_valid,quality,
                     arrays.get('elevation_validity'),arrays.get('elevation_error_bound',quality.vertical_error_bound_m))
