"""Ağ veya model gerektirmeyen, metre ölçekli sentetik görev örnekleri hazırlar."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rover_profile import RoverProfile


SCENARIOS = {
    'flat': 'candidate_selected',
    'detour': 'candidate_selected',
    'wall': 'no_reachable_candidate',
    'narrow_passage': 'no_reachable_candidate',
    'small_zone': 'no_reachable_candidate',
    'steep': 'exit_goal_not_traversable',
    'unknown_height': 'exit_goal_not_traversable',
    'uncertain_height': 'exit_goal_not_traversable',
    'forbidden_exit': 'exit_goal_not_traversable',
    'missing_exit': 'exit_goal_required',
    'fixed_valid': 'fixed_start_accepted',
    'fixed_blocked': 'fixed_start_rejected',
}


def prepare(destination, scenario='detour', *, profile=None, resolution_m=.1):
    if type(resolution_m) not in (int,float) or not np.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError('Örnek çözünürlüğü pozitif sonlu metre olmalı.')
    profile = RoverProfile() if profile is None else profile
    if not isinstance(profile,RoverProfile):
        raise ValueError('Örnek için RoverProfile gerekli.')
    if scenario not in SCENARIOS:
        raise ValueError(f'Bilinmeyen senaryo: {scenario}')
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Örnek veri için yeni veya boş bir klasör seçin.')
    destination.mkdir(parents=True, exist_ok=True)
    shape = (120, 160)
    z = np.zeros(shape, dtype='float32')
    ground = np.ones(shape, dtype='uint8')
    allowed = ground.copy()
    zone = np.zeros(shape, dtype='uint8')
    zone[20:100, 15:60] = 1
    mission = {'mode': 'select_within_zone', 'exit_pixel': [130, 60],
               'rule_context': 'Synthetic software verification; no competition rule claim.'}
    if scenario == 'detour':
        ground[35:85, 75:85] = 0
    if scenario in ('wall', 'fixed_blocked'):
        ground[:, 78:83] = 0
    if scenario == 'narrow_passage':
        ground[:, 78:83] = 0
        ground[54:66, 78:83] = 1
    if scenario == 'small_zone':
        zone[:] = 0
        zone[58:63, 33:38] = 1
    if scenario == 'steep':
        z[:] = np.indices(shape)[1] * resolution_m * np.tan(np.deg2rad(25))
    if scenario == 'unknown_height':
        z[60, 130] = np.nan
    if scenario == 'forbidden_exit':
        allowed[:, 120:] = 0
    if scenario == 'missing_exit':
        mission.pop('exit_pixel')
    if scenario in ('fixed_valid', 'fixed_blocked'):
        mission.update(mode='assess_fixed_start', fixed_start_pixel=[35, 60])
    rgb = np.full((3, *shape), 165, dtype='uint8')
    rgb[:, ground == 0] = np.array([65, 80, 100], dtype='uint8')[:, None]
    spec = {'ground_mode': 'reference', 'elevation_unit': 'm', 'elevation_kind': 'DTM',
            'ground_mask_source': 'Generated binary software fixture, not learned perception.',
            'dataset_source': 'RoverSafe deterministic synthetic terrain; not real imagery.',
            'mission_source': 'Hand-defined synthetic zone and exit, independent of planner output.',
            'rover': asdict(profile), 'mission': mission}
    if scenario == 'uncertain_height':
        spec['elevation_quality'] = {'mode': 'bounded_error', 'vertical_error_bound_m': .05,
                                     'source': 'Synthetic pointwise error bound for software test.'}
    for key, array in (('rgb', rgb), ('elevation', z[None]), ('ground_mask', ground[None]),
                       ('allowed_mask', allowed[None]), ('start_zone', zone[None])):
        spec[key] = key+'.tif'
        with rasterio.open(destination/spec[key], 'w', driver='GTiff', height=shape[0], width=shape[1],
                           count=len(array), dtype=array.dtype, crs='EPSG:32633',
                           transform=from_origin(500000, 4500000, resolution_m, resolution_m), compress='deflate') as dst:
            dst.write(array)
    path = destination/'manifest.json'
    path.write_text(json.dumps(spec, indent=2, allow_nan=False), encoding='utf-8')
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/sample_inputs'))
    parser.add_argument('--scenario', choices=SCENARIOS, default='detour')
    args = parser.parse_args()
    print(prepare(args.output_dir, args.scenario))
