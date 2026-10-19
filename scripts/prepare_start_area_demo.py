"""İndirilen Potsdam 2_10 karosundan referans destekli, tekrar üretilebilir demo hazırlar."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt
from elevation_analysis import analyze_elevation
from rover_profile import RoverProfile
from mission_constraints import MissionConfig
from start_site_selection import select_start_site
from potsdam_data import tile_names


def prepare(root, destination, crop, tile='2_10'):
    rgb_name,label_name,dsm_stem = tile_names(tile)
    paths = {'rgb': root/rgb_name,
             'elevation': root/'1_DSM'/(dsm_stem+'.tif'),
             'labels': root/label_name}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f'Önce resmî Potsdam {tile} dosyalarını indirin: {path}')
    destination.mkdir(parents=True, exist_ok=True)
    x, y, width, height = crop
    window = Window(x, y, width, height)
    arrays, grids, hashes = {}, [], {}
    for key, path in paths.items():
        with rasterio.open(path) as src:
            if min(x,y) < 0 or min(width,height) < 1 or x+width > src.width or y+height > src.height:
                raise ValueError('Kırpma kaynak görüntü içinde olmalı.')
            grid = (src.width, src.height, src.crs, src.transform)
            # Resmî etiket TIFF'inde CRS yoktur. Yayıncının aynı ızgara
            # tanımına dayanarak yalnız bu bilinen etiketin referansını RGB'den al.
            if key == 'labels' and src.crs is None:
                if (src.width, src.height) != grids[0][:2]:
                    raise ValueError('Potsdam etiket boyutu RGB ile uyuşmuyor.')
                grid = grids[0]
            grids.append(grid)
            arrays[key] = src.read(window=window)
            if key == 'rgb':
                transform, crs = src.window_transform(window), src.crs
        with path.open('rb') as stream:
            hashes[str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
    if any(grid != grids[0] for grid in grids):
        raise ValueError('Kaynak RGB, DSM ve etiketler aynı ızgarada değil.')
    # Beyaz etiket: impervious surfaces. Bina, ağaç, araç ve diğer sınıflar dışarıda.
    ground = np.all(arrays.pop('labels')[:3] == 255, axis=0)
    allowed = np.ones(ground.shape, dtype=bool)
    profile = RoverProfile()
    terrain = analyze_elevation(arrays['elevation'][0].astype('float64'), transform.a, -transform.e)
    # Çıkış, bu araştırma demosunda en büyük uygun bileşenden deterministik seçilir.
    # Bu, organizatörün gerçek görev hedefi değildir ve başarı değerlendirmesi sayılmaz.
    hard = (terrain.valid & ground & (terrain.slope_deg <= profile.max_slope_deg)
            & (terrain.roughness_m <= profile.max_roughness_m) & (terrain.step_m <= profile.max_step_m))
    clearance = distance_transform_edt(np.pad(hard, 1), sampling=(-transform.e, transform.a))[1:-1,1:-1]
    radius = np.hypot(profile.width_m,profile.length_m)/2 + profile.margin_m + np.hypot(transform.a,transform.e)/2
    motion = (hard & (clearance >= radius)).astype('uint8')
    count, labels, stats, _ = cv2.connectedComponentsWithStats(motion, connectivity=4)
    if count < 2:
        raise ValueError('Bu kırpmada örnek profile uygun bağlantı bulunamadı; eşikler otomatik gevşetilmedi.')
    component = 1 + int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    points = np.argwhere(labels == component)
    row, column = map(int, points[len(points)//5])
    exit_pixel = [column, row]
    for key, array in [('ground_mask',ground), ('allowed_mask',allowed), ('start_zone',allowed)]:
        arrays[key] = array[None].astype('uint8')
    for key, array in arrays.items():
        with rasterio.open(destination/f'{key}.tif', 'w', driver='GTiff', width=width,height=height,
                           count=len(array),dtype=array.dtype,crs=crs,transform=transform,compress='deflate') as dst:
            dst.write(array)
    spec = {key:f'{key}.tif' for key in arrays}
    spec.update({'elevation_unit':'m', 'elevation_kind':'DSM', 'window_size':5,
                 'dataset_source':f'ISPRS Potsdam tile {tile}, BSF Swissphoto; research demo, not ERC/URC terrain.',
                 'ground_mask_source':f'REFERENCE LABELS: white impervious-surface class; NOT an automatic ground detector. Unreferenced official {tile} label TIFF assigned matching RGB grid per ISPRS tile specification.',
                 'mission_source':'Illustrative full-crop permission/start zone; exit chosen from largest feasible component. Not organizer authorization.',
                 'rover':asdict(profile), 'source_sha256':hashes, 'crop_xywh':list(crop),
                 'mission':{'mode':'select_within_zone','exit_pixel':exit_pixel,
                            'rule_context':'synthetic_mission_on_real_potsdam_data_not_competition_approval'}})
    (destination/'manifest.json').write_text(json.dumps(spec, indent=2), encoding='utf-8')
    selected = select_start_site(terrain, profile, MissionConfig(**spec['mission']),transform.a,-transform.e,
                                ground_mask=ground, allowed_mask=allowed, start_zone=allowed)
    if selected.best is None:
        raise ValueError(f'Demo aday üretmedi: {selected.status}')
    for suffix, point in [('fixed_accepted', selected.best['center_pixel']), ('fixed_rejected', [0,0])]:
        variant = dict(spec)
        variant['mission'] = dict(spec['mission'], mode='assess_fixed_start', fixed_start_pixel=point)
        (destination/f'{suffix}.json').write_text(json.dumps(variant, indent=2), encoding='utf-8')
    (destination/'sources.json').write_text(json.dumps({
        'dataset_page':'https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx',
        'download_page':'https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/default.aspx',
        'attribution':'ISPRS benchmark / BSF Swissphoto',
        'sha256':hashes, 'crop_xywh':list(crop), 'reference_assisted':True,
        'note':'5cm grid spacing is not a vertical accuracy guarantee; DSM may include interpolated holes.'}, indent=2), encoding='utf-8')
    return destination/'manifest.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=Path('data/datasets/potsdam'))
    parser.add_argument('--tile',default='2_10')
    parser.add_argument('--output-dir', type=Path, default=Path('data/datasets/potsdam/start_demo'))
    parser.add_argument('--crop', nargs=4, type=int, default=[1200,2100,1200,1200], metavar=('X','Y','WIDTH','HEIGHT'))
    args = parser.parse_args()
    print(prepare(args.source_dir, args.output_dir, args.crop, args.tile))
