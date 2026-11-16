"""Kararı coğrafi dosyalara aktarır ve girdilerin dosya özetlerini kaydeder."""

import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform


EXPORT_ARTIFACTS = ('selection.geojson', 'decision.tif', 'traversability.tif')


def fingerprint_inputs(paths, manifest_path):
    """Büyük rasterları belleğe almadan SHA-256 hesaplar; içerik kopyalanmaz."""
    base = Path(manifest_path).resolve().parent
    records = []
    for path in sorted(paths, key=str):
        path = Path(path)
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        try:
            name = str(path.relative_to(base))
        except ValueError:
            name = str(path)
        records.append({'path': name, 'size_bytes': path.stat().st_size, 'sha256': digest})
    return records


def export_decision(data, result, mission, output_dir):
    """GeoJSON koordinatları WGS84 boylam/enlem; GeoTIFF kaynak CRS'dedir."""
    output = Path(output_dir)
    features = []

    def geographic(points):
        xs, ys = zip(*points)
        lon, lat = transform(data.crs, 'EPSG:4326', xs, ys)
        return [list(pair) for pair in zip(lon, lat)]

    def feature(kind, geometry, properties=None):
        features.append({'type': 'Feature', 'properties': {'kind': kind, **(properties or {})},
                         'geometry': geometry})

    for kind, point in (('exit', mission.exit_pixel), ('fixed_start', mission.fixed_start_pixel)):
        if point is not None:
            feature(kind, {'type': 'Point', 'coordinates': geographic([data.world(point)])[0]})
    if result.best:
        feature('selected_start', {'type': 'Point',
                'coordinates': geographic([data.world(result.best['center_pixel'])])[0]},
                {'score': result.best['score'], 'status': result.status})
        x, y, width, height = result.best['footprint_with_margin_px']
        # Köşeler hücre sınırlarıdır; merkez noktalarındaki yarım hücre kayması uygulanmaz.
        corners = [(x, y), (x, y+height), (x+width, y+height), (x+width, y), (x, y)]
        projected = [data.transform * point for point in corners]
        feature('footprint_with_margin', {'type': 'Polygon', 'coordinates': [geographic(projected)]})
    if len(result.route_pixels) > 1:
        feature('route', {'type': 'LineString',
                'coordinates': geographic([data.world(p) for p in result.route_pixels])},
                {'length_m': result.best['route_length_m']})
    collection = {'type': 'FeatureCollection', 'features': features,
                  'status': result.status, 'coordinate_order': 'longitude, latitude'}
    (output/'selection.geojson').write_text(json.dumps(collection, indent=2, allow_nan=False), encoding='utf-8')

    classes = np.zeros(result.traversable.shape, dtype='uint8')
    classes[result.traversable] = 1
    classes[result.valid_centers] = 2
    classes[result.eligible_centers if result.eligible_centers is not None else result.reachable_centers] = 3
    with rasterio.open(output/'decision.tif', 'w', driver='GTiff',
                       height=classes.shape[0], width=classes.shape[1], count=1,
                       dtype='uint8', crs=data.crs, transform=data.transform, compress='deflate') as dst:
        dst.write(classes, 1)
        dst.set_band_description(1, 'decision_class')
        dst.update_tags(classes=json.dumps({'0': 'excluded_or_unknown', '1': 'terrain_eligible',
                                           '2': 'valid_start_center', '3': 'eligible_reachable_start_center'}),
                        status=result.status, interpretation='Screening classes, not safety probabilities')
    with rasterio.open(output/'traversability.tif','w',driver='GTiff',
                       height=classes.shape[0],width=classes.shape[1],count=2,dtype='float32',
                       crs=data.crs,transform=data.transform,nodata=-9999,compress='deflate') as dst:
        for band,(name,values) in enumerate((('terrain_severity',result.terrain_risk),
                                             ('cost_per_meter',result.route_cost_map)),start=1):
            dst.write(np.where(np.isfinite(values),values,-9999).astype('float32'),band)
            dst.set_band_description(band,name)
        dst.update_tags(interpretation='Heuristic terrain severity and route cost, not safety probabilities')
