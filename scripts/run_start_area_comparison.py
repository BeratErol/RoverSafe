"""Dondurulmuş yüzey modeliyle dört Potsdam kırpmasında yöntem karşılaştırır.

Referans etiketleri tahmin sırasında okunmaz; görev tanımı referans destekli
demo hazırlayıcısından gelir. Sonuçlar aynı karoda aktarım incelemesidir.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import rasterio
from scripts.prepare_start_area_demo import prepare
from start_area_cli import run


def ratio(a, b):
    return float(a / b) if b else None


def semantic_metrics(ground, reference, report):
    tp = int((ground & reference).sum())
    fp = int((ground & ~reference).sum())
    fn = int((~ground & reference).sum())
    metrics = {'tp':tp,'fp':fp,'fn':fn,'precision':ratio(tp,tp+fp),
               'recall':ratio(tp,tp+fn),'iou':ratio(tp,tp+fp+fn),
               'footprint_non_target_fraction':None,'route_swept_non_target_fraction':None}
    if report['best']:
        x,y,w,h = report['best']['footprint_with_margin_px']
        metrics['footprint_non_target_fraction'] = float((~reference[y:y+h,x:x+w]).mean())
        route = np.zeros(reference.shape,dtype='uint8')
        points = np.asarray(report['route_pixels'],dtype='int32')
        route[points[:,1],points[:,0]] = 1
        rover = report['rover']
        dx,dy = report['resolution_m']
        radius = np.hypot(rover['width_m'],rover['length_m'])/2 + rover['margin_m'] + np.hypot(dx,dy)/2
        rx,ry = int(np.ceil(radius/dx)),int(np.ceil(radius/dy))
        yy,xx = np.mgrid[-ry:ry+1,-rx:rx+1]
        kernel = (np.hypot(xx*dx,yy*dy) <= radius).astype('uint8')
        swept = cv2.dilate(route,kernel).astype(bool)
        metrics['route_swept_non_target_fraction'] = float((~reference[swept]).mean())
    return metrics


def compare(source, output, surface_model, yolo_model=None):
    frozen_sha256 = hashlib.sha256(surface_model.read_bytes()).hexdigest()
    crops = [(1200,2100,1200,1200),(3600,1200,1200,1200),
             (0,3600,1200,1200),(3600,3600,1200,1200)]
    rows = []
    output.mkdir(parents=True,exist_ok=True)
    for index,crop in enumerate(crops):
        folder = source/f'comparison/crop_{index}'
        try:
            manifest = prepare(source,folder,crop)
        except ValueError as exc:
            rows.append({'crop':list(crop),'status':'preparation_failed','reason':str(exc)})
            continue
        spec = json.loads(manifest.read_text(encoding='utf-8'))
        methods = ['reference','predicted'] + (['predicted_yolo'] if yolo_model else [])
        for method in methods:
            variant = dict(spec)
            if method != 'reference':
                variant.pop('ground_mask')
                # GT özetinin varlığı dahi algılamada gerekmez; yalnız görüntü/DSM kökeni kalır.
                variant['source_sha256'] = {k:v for k,v in variant['source_sha256'].items() if 'label' not in k}
                variant['ground_mode'] = 'predicted'
                variant['ground_mask_source'] = 'Frozen supplied RGB surface model; no fit or threshold tuning during this comparison.'
                variant['surface'] = {'model_path':os.path.relpath(surface_model.resolve(),folder.resolve()),
                                      'threshold':.6,'grid_step':4}
            if method == 'predicted_yolo':
                variant['obstacles'] = {'model_path':os.path.relpath(yolo_model.resolve(),folder.resolve()),
                                        'margin_m':.2,'image_size':1280,'confidence':.25,'device':'cpu'}
            filename = folder/f'{method}.json'
            filename.write_text(json.dumps(variant,indent=2),encoding='utf-8')
            started = perf_counter()
            result_dir = output/f'crop_{index}'/method
            result = run(filename,result_dir)
            if method != 'reference' and result['perception']['model_sha256'] != frozen_sha256:
                raise ValueError('Yüzey modeli çalışma sırasında değişti; karşılaştırma geçersiz.')
            seconds = perf_counter()-started
            # Referans yalnızca tahmin tamamlandıktan sonra değerlendirme için okunur.
            with rasterio.open(folder/'ground_mask.tif') as src:
                reference = src.read(1) > 0
            with np.load(result_dir/'perception.npz') as maps:
                metrics = semantic_metrics(maps['ground'],reference,result)
                filtered_metrics = semantic_metrics(maps['ground'] & ~maps['obstacles'],reference,result)
            rows.append({'crop':list(crop),'method':method,'status':result['status'],
                         'seconds':seconds,'best':result['best'],'counts':result['counts'],
                         'detections':len(result['perception']['obstacles'].get('detections',[])),
                         'semantic_proxy':metrics,'post_obstacle_semantic_proxy':filtered_metrics})
            print(index,method,result['status'],flush=True)
    summary = {'protocol':{'threshold':.6,'crops_xywh':[list(c) for c in crops],
                'surface_model_sha256':frozen_sha256,
                'note':'Same tile, disjoint crops; mission goals derived from reference preparation. Not independent race or physical safety validation.',
                'training':'Supplied model frozen; no fit or threshold optimization during comparison. Default local model was trained on AeroScapes.',
                'semantic_target':'ISPRS impervious surfaces; all other classes non-target, not physical safety truth.'},'results':rows}
    (output/'comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    write_report(summary,output/'comparison.md')
    return summary


def write_report(summary, path):
    lines = ['# Otomatik zemin ve yükseklik karşılaştırması', '',
             'Dört önceden tanımlanmış, örtüşmeyen Potsdam 2_10 kırpması. '
             'Verilen model ve 0,60 eşiği değiştirilmedi (bu çalışmada AeroScapes modeli). Referans görev '
             'çıkışları hazır etiket destekli hazırlayıcıdan geliyor; otomatik '
             'zemin tahmini etiketi okumuyor. Bu çalışma kör test veya fiziksel '
             'güvenlik değerlendirmesi değildir.', '',
             '| Kırpma x,y | Yöntem | Durum | Zemin IoU | Nesne | Ayak izi hedef dışı oranı |',
             '| --- | --- | --- | --- | --- | --- |']
    for row in summary['results']:
        metrics = row.get('semantic_proxy',{})
        iou = metrics.get('iou')
        fraction = metrics.get('footprint_non_target_fraction')
        crop = row['crop']
        lines.append(f"| {crop[0]},{crop[1]} | {row.get('method','hazırlama')} | {row['status']} | "
                     f"{iou if iou is not None else '—'} | {row.get('detections','—')} | "
                     f"{fraction if fraction is not None else '—'} |")
    lines += ['', 'Referans satırlarının IoU=1 olması etiketin kendisiyle karşılaştırılmasından '
              'kaynaklanır; algılayıcı başarısı değildir. Yüzey IoU değeri nesne filtresinden '
              'önceki maskeye aittir. JSON dosyasında nesne filtresinden sonraki ölçüm de bulunur.', '',
              'Hazırlanamayan kırpmalar saklanır; uygun çıkış üretmek için eşikler gevşetilmez. '
              'Öneri verilmediğinde ayak izi oranı boş kalır; bu sonuç otomatik başarı sayılmaz. '
              'Öneri verilen tek bir kırpmada hedef dışı oranının sıfır olması fiziksel emniyet '
              'kanıtı değildir. Rota boyunca roverın kapladığı dairesel alanın semantik '
              'uyuşmazlığı da JSON içinde değerlendirilir.', '',
              'Kalan temel sorun: AeroScapes renk/doku modelinin Potsdam gibi farklı '
              'görüntülere aktarımı zayıf. Farklı karolara/sahalara ayrılmış hedef veriyle '
              'eğitim, doğrulama ve bağımsız test gerekir. Aynı karonun kırpmaları '
              'bağımsız saha sayılmaz. YOLO genel amaçlıdır; tepeden küçük nesneleri '
              'kaçırabilir. Nesne bulunmaması engel olmadığı anlamına gelmez.', '']
    Path(path).write_text('\n'.join(lines),encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,default=Path('data/datasets/potsdam'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/start_area_comparison'))
    parser.add_argument('--surface-model',type=Path,default=Path('models/surface.joblib'))
    parser.add_argument('--yolo-model',type=Path,help='Verilirse üçüncü yöntem de çalışır.')
    args = parser.parse_args()
    compare(args.source_dir,args.output_dir,args.surface_model,args.yolo_model)
