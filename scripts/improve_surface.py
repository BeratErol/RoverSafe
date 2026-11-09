"""Çok ölçekli modeli yalnız geliştirmede eğitir, model/eşiği yalnız doğrulamada seçer."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from potsdam_data import iter_patches, sha256
from scripts.train_potsdam_surface import train, metric_counts, metrics
from surface_classification import MULTISCALE_VERSION, FEATURE_BORDERS, SurfaceConfig, SurfacePredictor


def choose(rows, precision_floor=.8, recall_floor=.4):
    feasible = [row for row in rows if (row['precision'] or 0) >= precision_floor and (row['recall'] or 0) >= recall_floor]
    pool = feasible or rows
    selected = max(pool,key=lambda row:(row['f1'],row['precision'] or 0,row['threshold'],row['model']))
    return selected,bool(feasible)


def study(protocol_path, baseline, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Yeni veya boş bir geliştirme klasörü seçin.')
    output.mkdir(parents=True,exist_ok=True)
    original = Path(protocol_path).resolve()
    spec = json.loads(original.read_text(encoding='utf-8'))
    for entry in spec['entries']:
        for key in ('rgb','labels'):
            entry[key] = str((original.parent/entry[key]).resolve())
    protocol = output/'protocol.json'
    protocol.write_text(json.dumps(spec,indent=2),encoding='utf-8')
    # Protokol test kayıtlarını taşır; bu komutta test görüntüsü/etiketi okunmaz.
    plan = {'selection_split':'validation','objective':'maximum F1 among candidates with precision>=0.8 and recall>=0.4; if none, maximum F1 with gate failure recorded',
            'thresholds':[.5,.6,.7,.8,.9],'models':['native_v1','multiscale_v2'],
            'feature_border_px':20,'source_protocol_sha256':sha256(original),
            'baseline_model_sha256':sha256(baseline),'test_data_accessed':False}
    (output/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    model = output/'surface_multiscale.joblib'
    train(protocol,model,feature_version=MULTISCALE_VERSION,samples_per_class=512)
    predictors = {'native_v1':SurfacePredictor(SurfaceConfig(model_path=str(baseline),grid_step=4)),
                  'multiscale_v2':SurfacePredictor(SurfaceConfig(model_path=str(model),grid_step=4))}
    counts = {(name,threshold):np.zeros(4,dtype='int64') for name in predictors for threshold in plan['thresholds']}
    border = max(FEATURE_BORDERS[p.feature_version] for p in predictors.values())
    windows = []
    for tile,crop,rgb,binary,_ in iter_patches(protocol,'validation'):
        bgr = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        for name,predictor in predictors.items():
            score = predictor.predict(bgr).probability
            for threshold in plan['thresholds']:
                counts[name,threshold] += metric_counts(score[border:-border,border:-border],binary[border:-border,border:-border],threshold)
        windows.append({'tile':tile,'window':crop})
        if len(windows)%10 == 0:
            print('Validation',len(windows),'windows',flush=True)
    if not windows:
        raise ValueError('Doğrulama penceresi gerekli.')
    rows = []
    for (name,threshold),value in counts.items():
        row = {'model':name,'threshold':threshold,**metrics(value)}
        tp,fp,fn,_ = value
        row['f1'] = float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0
        rows.append(row)
    best,passed = choose(rows)
    selected_path = Path(predictors[best['model']].config.model_path).resolve()
    selection = {'selection_split':'validation','selected_model':best['model'],'selected_model_path':str(selected_path),
                 'threshold':best['threshold'],'feature_grid_step':4,'feature_border_px':border,
                 'model_sha256':sha256(selected_path),'protocol_sha256':sha256(protocol),
                 'selection_rule':plan['objective'],'passes_validation_gate':passed,'candidates':rows,'windows':windows,
                 'test_data_accessed':False,'feature_version':predictors[best['model']].feature_version}
    (output/'selection.json').write_text(json.dumps(selection,indent=2,allow_nan=False),encoding='utf-8')
    return selection


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=Path('outputs/potsdam_native_study/protocol.json'))
    parser.add_argument('--baseline',type=Path,default=Path('models/surface_potsdam.joblib'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/surface_improvement'))
    args = parser.parse_args()
    result = study(args.protocol,args.baseline,args.output_dir)
    print(result['selected_model'],result['threshold'],'validation_gate',result['passes_validation_gate'])
