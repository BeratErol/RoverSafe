"""Sabit görevde varsayılan yükseklik hata sınırının karar etkisini ölçer."""

import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from start_area_cli import run


def absolute_inputs(spec, base):
    result = copy.deepcopy(spec)
    for key in ('rgb','elevation','ground_mask','allowed_mask','start_zone'):
        if key in result:
            result[key] = str((base/result[key]).resolve())
    for section in ('surface','obstacles'):
        if isinstance(result.get(section),dict) and 'model_path' in result[section]:
            result[section]['model_path'] = str((base/result[section]['model_path']).resolve())
    quality = result.get('elevation_quality',{})
    for key in ('validity_mask','error_bound_raster'):
        if quality.get(key) is not None:
            quality[key] = str((base/quality[key]).resolve())
    return result


def study(manifest_path, output):
    original = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    base = absolute_inputs(original,manifest_path.resolve().parent)
    output.mkdir(parents=True,exist_ok=True)
    rows = []
    previous = None
    nominal = None
    fingerprint = None
    cases = [('nominal',None),('bound_0',0),('bound_002',.002),('bound_005',.005),
             ('bound_010',.01),('bound_020',.02),('bound_050',.05)]
    for name,bound in cases:
        spec = copy.deepcopy(base)
        quality = {'mode':'nominal','source':'Sensitivity baseline; no measured accuracy bound supplied.'} if bound is None else {
            'mode':'bounded_error','vertical_error_bound_m':bound,
            'source':'Illustrative sensitivity assumption; NOT measured Potsdam/sensor accuracy.'}
        if base.get('elevation_quality',{}).get('validity_mask'):
            quality['validity_mask'] = base['elevation_quality']['validity_mask']
            quality['source'] += ' Validity mask retained from source manifest.'
        spec['elevation_quality'] = quality
        path = output/f'{name}.json'
        path.write_text(json.dumps(spec,indent=2),encoding='utf-8')
        report = run(path,output/name)
        current_fingerprint = (report['perception'].get('model_sha256'),
                               report['perception']['obstacles'].get('model_sha256'))
        if fingerprint is None:
            fingerprint = current_fingerprint
        elif current_fingerprint != fingerprint:
            raise ValueError('Deney sırasında algılama modeli değişti.')
        with np.load(output/name/'maps.npz') as maps:
            traversable,centers = maps['traversable'],maps['valid_centers']
        if bound is None:
            nominal = (traversable.copy(),centers.copy())
        else:
            if bound == 0 and (not np.array_equal(traversable,nominal[0]) or not np.array_equal(centers,nominal[1])):
                raise ValueError('Sıfır hata sınırı nominal maskeleri değiştirdi.')
            if previous is not None and ((traversable & ~previous[0]).any() or (centers & ~previous[1]).any()):
                raise ValueError('Artan hata sınırı yeni uygun piksel ekledi; deney durduruldu.')
            previous = (traversable.copy(),centers.copy())
        rows.append({'scenario':name,'assumed_error_bound_m':bound,'status':report['status'],
                     'counts':report['counts'],'point_checks':report['point_checks'],'best':report['best']})
        print(name,report['status'],report['counts']['valid_center_pixels'],flush=True)
    record = {'source_manifest':str(manifest_path),'mission':original['mission'],
              'protocol':'Same imagery, masks, rover and task. Assumed pointwise error bounds; no accuracy calibration or confidence claim.',
              'limitations':'Reference-assisted ground/mission may be used; see original manifest. Horizontal error and subpixel hazards are not modeled.',
              'model_sha256':fingerprint,'results':rows}
    (output/'summary.json').write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
    lines = ['# Yükseklik hata sınırı duyarlılık deneyi','',
             'Aynı görüntü, zemin/izin maskeleri, rover profili ve görev noktaları kullanıldı. '
             'Sınırlar varsayımsal mutlak piksel hatalarıdır; veri setinin ölçülmüş düşey doğruluğu değildir. '
             'Model/eşik seçimi veya yeni kör test yapılmadı.','',
             '| Hata sınırı (m) | Durum | Arazi hücresi | Aday merkez | Çıkışa bağlı aday |',
             '| --- | --- | --- | --- | --- |']
    for row in rows:
        label = 'Nominal / bilinmiyor' if row['assumed_error_bound_m'] is None else f"±{row['assumed_error_bound_m']:.3f}"
        count = row['counts']
        lines.append(f"| {label} | {row['status']} | {count['traversable_pixels']} | "
                     f"{count['valid_center_pixels']} | {count['reachable_center_pixels']} |")
    lines += ['', 'Sıfır sınır nominal maskeleri korudu; sınır büyüdükçe arazi ve aday merkez '
              'maskelerine yeni piksel eklenmediği program içinde kontrol edildi. Puanlar ve '
              'seçilen nokta değişebilir; öneri sayısı veya rota uzunluğu tek başına güvenlik ölçüsü değildir.', '',
              'Nominal sonuç doğrulanmış yükseklik doğruluğu içermez. Muhafazakâr sınırların '
              'alanı elemesi gerçek arazinin kesinlikle geçilemez olduğunu göstermez; '
              'verilen hata varsayımı altında uygunluk gösterilemediğini belirtir. '
              'Kalibrasyon veya bağımsız arazi ölçümü yapılmadı.','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bounded = [row for row in rows if row['assumed_error_bound_m'] is not None]
    fig,axes = plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,key,title in zip(axes,('traversable_pixels','valid_center_pixels'),('Terrain cells','Candidate center cells')):
        x = [100*row['assumed_error_bound_m'] for row in bounded]
        ax.plot(x,[row['counts'][key] for row in bounded],'o-',label='Bounded-error assumption')
        ax.axhline(rows[0]['counts'][key],color='gray',linestyle='--',label='Nominal screening')
        ax.set(xlabel='Assumed absolute height error [cm]',ylabel='Pixel count',title=title)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle('Fixed mission sensitivity — assumed bounds, not measured sensor accuracy')
    fig.savefig(output/'sensitivity.png',dpi=160)
    plt.close(fig)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=Path('data/datasets/potsdam/start_demo/manifest.json'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/elevation_uncertainty'))
    args = parser.parse_args()
    study(args.manifest,args.output_dir)
