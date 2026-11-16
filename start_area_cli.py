"""Yükseklik tabanlı başlangıç alanı analizi ve izlenebilir çıktı üretimi."""

import argparse
from dataclasses import asdict
import json
import platform
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
from start_area_export import EXPORT_ARTIFACTS, export_decision, fingerprint_inputs
from route_cost import RoutingConfig


OUTPUT_ARTIFACTS = ('maps.npz','elevation_quality.npz','perception.npz','selection.png', *EXPORT_ARTIFACTS)


def input_paths(manifest_path, spec):
    """Geçersiz ayarlarda da okunabilen dosya referanslarını korumak için toplar."""
    manifest = Path(manifest_path).resolve()
    paths = {manifest}
    if not isinstance(spec,dict):
        return paths
    for key in ('rgb','elevation','ground_mask','allowed_mask','start_zone'):
        if isinstance(spec.get(key),str):
            paths.add((manifest.parent/spec[key]).resolve())
    for section,keys in (('surface',('model_path','encoder_path')),('obstacles',('model_path',)),
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
    routing = RoutingConfig(**spec.get('routing',{}))
    terrain = analyze_elevation(data.elevation, data.dx, data.dy, spec.get('window_size', 5),
                                quality_mask=data.elevation_validity,error_bound_m=data.elevation_error_bound_m)
    perception = estimate_surfaces(data, manifest_path)
    fingerprints = fingerprint_inputs(protected, manifest_path)
    result = select_start_site(terrain, profile, mission, data.dx, data.dy,
                              ground_mask=perception.ground, allowed_mask=data.allowed,
                              start_zone=data.start_zone, obstacle_mask=perception.obstacles,routing=routing)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {'report_type':'roversafe_start_area_v1','status': result.status, 'best': result.best, 'artifacts_valid':True, 'rover': asdict(profile),
              'mission': asdict(mission), 'crs': data.crs.to_string(),
              'routing':{**asdict(routing),'cost_meaning':'Distance weighted by terrain severity and obstacle proximity; not energy or safety probability.',
                         'terrain_components':{'slope':.5,'roughness':.3,'local_step':.2},
                         'site_selection':'Site ranking is independent of route cost; routing optimizes the route from the selected site.'},
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
                         'reachable_center_pixels':int(result.reachable_centers.sum()),
                         'eligible_start_center_pixels':int(result.eligible_centers.sum())},
              'dataset_source': spec['dataset_source'], 'ground_mask_source': spec['ground_mask_source'],
              'mission_source': spec['mission_source'],
              'input_manifest': spec,
              'rover_profile_provenance':spec.get('rover_profile_provenance'),
              'input_fingerprints': fingerprints,
              'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'opencv': cv2.__version__},
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
                        traversable=result.traversable, reachable=result.reachable_centers,eligible_start=result.eligible_centers,
                        valid_centers=result.valid_centers, scores=result.score_map,
                        terrain_risk=result.terrain_risk,route_cost=result.route_cost_map)
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
    tint[result.eligible_centers] = (40, 200, 40)
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
    display_scale = max(1, min(4, int(np.ceil(360/overlay.shape[0]))))
    if display_scale > 1:
        overlay = cv2.resize(overlay, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_NEAREST)
    # Açıklamalar haritanın üstünü örtmez; küçük girdilerde de okunabilir kalır.
    map_height, map_width = overlay.shape[:2]
    canvas = np.full((max(map_height, 360), map_width+400, 3), 28, dtype='uint8')
    canvas[:map_height, :map_width] = overlay
    panel_x = map_width+16
    cv2.putText(canvas, 'RoverSafe', (panel_x, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
    cv2.putText(canvas, result.status, (panel_x, 54), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1)
    for index, (label, color) in enumerate([
        ('Excluded / unknown', (30,30,220)), ('Eligible start centers', (40,200,40)),
        ('Selected footprint', (0,255,255)), ('Route', (255,255,0)), ('Exit', (255,180,0))]):
        y = 84 + index*23
        cv2.rectangle(canvas,(panel_x,y-12),(panel_x+16,y),color,-1)
        cv2.putText(canvas,label,(panel_x+25,y),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    lines = [f"Ground: {perception.metadata['ground_mode']} | Objects: "
             f"{'on' if perception.metadata['obstacles']['enabled'] else 'off'}",
             f'Elevation: {data.quality_config.mode}',
             f'Rover: {profile.width_m:g} x {profile.length_m:g} m; margin {profile.margin_m:g} m']
    if mission.min_start_exit_distance_m:
        lines += [f'Minimum exit separation: {mission.min_start_exit_distance_m:g} m']
    if result.best:
        lines += [f"Route length: {result.best['route_length_m']:.2f} m",
                  f"Route clearance: {result.best['route_min_clearance_m']:.2f} m"]
    else:
        exit_checks=result.point_checks.get('exit',{})
        failed = exit_checks.get('failed_center_constraints',[])
        envelope=exit_checks.get('failed_motion_envelope_constraints',{})
        if failed:
            lines += ['Exit center: '+', '.join(failed)]
        elif envelope:
            lines += ['Exit envelope: '+', '.join(envelope)]
        elif exit_checks.get('motion_envelope_outside_raster_cells',0):
            lines += ['Exit envelope: outside image']
        else:
            lines += ['See point_checks in result.json']
    lines += ['Screening result; see result.json']
    for index,line in enumerate(lines):
        cv2.putText(canvas,line,(panel_x,218+index*22),cv2.FONT_HERSHEY_SIMPLEX,.43,(225,225,225),1)
    ok, encoded = cv2.imencode('.png', canvas)
    if not ok:
        raise OSError('Görselleştirme kodlanamadı.')
    encoded.tofile(output_dir / 'selection.png')
    export_decision(data, result, mission, output_dir)
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
