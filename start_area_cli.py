"""Yükseklik tabanlı başlangıç alanı analizi ve izlenebilir çıktı üretimi."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import cv2
import numpy as np
from elevation_analysis import analyze_elevation
from geospatial_io import load_manifest
from mission_constraints import MissionConfig
from rover_profile import RoverProfile
from start_site_selection import select_start_site
from start_area_perception import estimate_surfaces
from obstacle_detection import ObstacleDetectionError


OUTPUT_ARTIFACTS = ('maps.npz','elevation_quality.npz','perception.npz','selection.png')


def input_paths(manifest_path, spec):
    """Geçersiz ayarlarda da okunabilen dosya referanslarını korumak için toplar."""
    manifest = Path(manifest_path).resolve()
    paths = {manifest}
    if not isinstance(spec,dict):
        return paths
    for key in ('rgb','elevation','ground_mask','allowed_mask','start_zone'):
        if isinstance(spec.get(key),str):
            paths.add((manifest.parent/spec[key]).resolve())
    for section,keys in (('surface',('model_path',)),('obstacles',('model_path',)),
                         ('elevation_quality',('validity_mask','error_bound_raster'))):
        if isinstance(spec.get(section),dict):
            for key in keys:
                if isinstance(spec[section].get(key),str):
                    paths.add((manifest.parent/spec[section][key]).resolve())
    return paths


def _run(manifest_path, output_dir):
    data = load_manifest(manifest_path)
    spec = data.manifest
    protected = input_paths(manifest_path,spec)
    if any((Path(output_dir)/name).resolve() in protected for name in ('result.json','result.json.part',*OUTPUT_ARTIFACTS)):
        raise ValueError('Çıktı dosyası bir giriş dosyasıyla çakışıyor; farklı çıktı klasörü seçin.')
    profile = RoverProfile(**spec.get('rover', {}))
    mission = MissionConfig(**spec['mission'])
    terrain = analyze_elevation(data.elevation, data.dx, data.dy, spec.get('window_size', 5),
                                quality_mask=data.elevation_validity,error_bound_m=data.elevation_error_bound_m)
    perception = estimate_surfaces(data, manifest_path)
    result = select_start_site(terrain, profile, mission, data.dx, data.dy,
                              ground_mask=perception.ground, allowed_mask=data.allowed,
                              start_zone=data.start_zone, obstacle_mask=perception.obstacles)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {'report_type':'roversafe_start_area_v1','status': result.status, 'best': result.best, 'artifacts_valid':True, 'rover': asdict(profile),
              'mission': asdict(mission), 'crs': data.crs.to_string(),
              'transform_gdal': list(data.transform.to_gdal()),
              'shape_rows_columns': list(data.elevation.shape),
              'resolution_m': [data.dx, data.dy], 'rejected_pixels': result.rejected_pixels,
              'point_checks':result.point_checks,
              'elevation_quality':{**asdict(data.quality_config),
                  'decision_values':'upper_bounds' if terrain.slope_upper_deg is not None else 'nominal',
                  'interpretation':'User-supplied pointwise absolute error assumption, not a confidence interval.'
                                   if terrain.slope_upper_deg is not None else 'No vertical accuracy bound supplied; nominal screening only.'},
              'counts': {'traversable_pixels':int(result.traversable.sum()),
                         'valid_center_pixels':int(result.valid_centers.sum()),
                         'reachable_center_pixels':int(result.reachable_centers.sum())},
              'dataset_source': spec['dataset_source'], 'ground_mask_source': spec['ground_mask_source'],
              'mission_source': spec['mission_source'],
              'input_manifest': spec,
              'perception': perception.metadata,
              'analysis_settings': {'window_size_px':terrain.window_size,
                                    'score_weights':{'slope':.4,'roughness':.3,'local_step':.1,'clearance':.2},
                                    'score_meaning':'Heuristic ranking, not a safety probability.',
                                    'rejection_counts':'Overlapping pixel counts; do not sum.'},
              'route_pixels': result.route_pixels,
              'route_projected_m': [data.world(p) for p in result.route_pixels],
              'limitations': ['Research screening; not a physical safety certificate.',
                             'Local 3x3 height range is a step indicator, not measured obstacle height.',
                             'Route uses circular footprint clearance and four neighbors; no vehicle dynamics.',
                             'Ground mask may be reference-assisted; see provenance.',
                             'Rover limits require real vehicle verification; verified flag is user supplied.']}
    if result.best:
        report['best']['center_projected_m'] = data.world(result.best['center_pixel'])
    (output_dir / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    np.savez_compressed(output_dir / 'maps.npz', slope_deg=terrain.slope_deg,
                        roughness_m=terrain.roughness_m, step_m=terrain.step_m,
                        traversable=result.traversable, reachable=result.reachable_centers,
                        valid_centers=result.valid_centers, scores=result.score_map)
    quality_maps = {'quality_invalid':terrain.quality_invalid,
                    'decision_slope_deg':terrain.decision_maps()[0],
                    'decision_roughness_m':terrain.decision_maps()[1],
                    'decision_step_m':terrain.decision_maps()[2]}
    if terrain.error_bound_m is not None:
        quality_maps['error_bound_m'] = terrain.error_bound_m
    np.savez_compressed(output_dir / 'elevation_quality.npz',**quality_maps)
    np.savez_compressed(output_dir / 'perception.npz', ground=perception.ground,
                        ground_score=perception.ground_score, obstacles=perception.obstacles)
    overlay = cv2.cvtColor(data.rgb, cv2.COLOR_RGB2BGR)
    tint = overlay.copy()
    tint[~result.traversable] = (30, 30, 220)
    tint[result.reachable_centers] = (40, 200, 40)
    overlay = cv2.addWeighted(overlay, 0.65, tint, 0.35, 0)
    if result.route_pixels:
        cv2.polylines(overlay, [np.array(result.route_pixels, dtype=np.int32)], False, (255, 255, 0), 2)
    if result.best:
        x, y, w, h = result.best['footprint_with_margin_px']
        cv2.rectangle(overlay, (x, y), (x+w-1, y+h-1), (0, 255, 255), 2)
    if mission.fixed_start_pixel:
        cv2.drawMarker(overlay, tuple(mission.fixed_start_pixel), (255, 0, 255), cv2.MARKER_CROSS, 15, 2)
    if mission.exit_pixel:
        cv2.circle(overlay, tuple(mission.exit_pixel), 5, (255, 180, 0), -1)
    cv2.putText(overlay, result.status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
    for index, (label, color) in enumerate([
        ('Excluded', (30,30,220)), ('Reachable centers', (40,200,40)),
        ('Selected footprint', (0,255,255)), ('Route / exit', (255,255,0))]):
        y = 50 + index*22
        cv2.rectangle(overlay,(8,y-12),(24,y),color,-1)
        cv2.putText(overlay,label,(30,y),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    cv2.putText(overlay, 'Research demo - see provenance in result.json', (10,150),
                cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    cv2.putText(overlay, f"Ground: {perception.metadata['ground_mode']} | Objects: "
                f"{'on' if perception.metadata['obstacles']['enabled'] else 'off'}", (10,172),
                cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    cv2.putText(overlay,f"Elevation: {data.quality_config.mode}",(10,194),
                cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    ok, encoded = cv2.imencode('.png', overlay)
    if not ok:
        raise OSError('Görselleştirme kodlanamadı.')
    encoded.tofile(output_dir / 'selection.png')
    return report


def run(manifest_path, output_dir):
    """Başarısız tekrar çalıştırmada önceki önerinin güncel görünmesini engeller."""
    output = Path(output_dir)
    try:
        return _run(manifest_path,output)
    except Exception as error:
        protected = {Path(manifest_path).resolve()}
        try:
            raw = json.loads(Path(manifest_path).read_text(encoding='utf-8-sig'))
            protected = input_paths(manifest_path,raw)
        except (OSError,ValueError,TypeError):
            pass
        try:
            previous = json.loads((output/'result.json').read_text(encoding='utf-8'))
        except (OSError,ValueError):
            previous = {}
        owned = isinstance(previous,dict) and (previous.get('report_type') == 'roversafe_start_area_v1'
            or all(key in previous for key in ('input_manifest','rover','counts','best','status')))
        if owned and all((output/name).resolve() not in protected for name in ('result.json','result.json.part')):
            failure = {'report_type':'roversafe_start_area_v1','status':'analysis_failed','best':None,'route_pixels':[],
                       'artifacts_valid':False,'error':str(error),
                       'source_manifest':str(manifest_path)}
            try:
                partial = output/'result.json.part'
                partial.write_text(json.dumps(failure,indent=2,ensure_ascii=False),encoding='utf-8')
                partial.replace(output/'result.json')
                # Yalnızca bu akışın bilinen çıktı dosyaları; klasör veya kaynak silinmez.
                for name in OUTPUT_ARTIFACTS:
                    if (output/name).resolve() not in protected:
                        (output/name).unlink(missing_ok=True)
            except OSError as cleanup_error:
                raise OSError(f'{error}; önceki çıktı geçersizleştirilemedi: {cleanup_error}') from error
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description='Metre tabanlı rover başlangıç alanı değerlendirmesi')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/start_area'))
    args = parser.parse_args(argv)
    try:
        result = run(args.manifest, args.output_dir)
    except (ValueError, OSError, KeyError, TypeError, ObstacleDetectionError) as exc:
        parser.error(str(exc))
    print(f"{result['status']}: {args.output_dir / 'result.json'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
