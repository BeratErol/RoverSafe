"""Ayrı karolarda eğitim, doğrulamayla eşik seçimi ve dondurulmuş test yürütür."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from potsdam_data import PALETTE, build_protocol, iter_patches, sha256
from surface_classification import FEATURE_VERSION, SurfaceConfig, SurfacePredictor, feature_grid


def metric_counts(score, binary, threshold):
    known = binary >= 0
    target = binary == 1
    predicted = score >= threshold
    return np.array([np.sum(predicted & target),np.sum(predicted & ~target & known),
                     np.sum(~predicted & target),np.sum(~predicted & ~target & known)],dtype='int64')


def metrics(counts):
    tp,fp,fn,tn = map(int,counts)
    precision = tp/(tp+fp) if tp+fp else None
    recall = tp/(tp+fn) if tp+fn else None
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':precision,'recall':recall,
            'iou':tp/(tp+fp+fn) if tp+fp+fn else None,
            'f0_5':1.25*tp/(1.25*tp+.25*fn+fp) if 1.25*tp+.25*fn+fp else 0.0}


def train(protocol, output):
    spec = json.loads(Path(protocol).read_text(encoding='utf-8'))
    rng = np.random.default_rng(spec['seed'])
    xs,ys,used = [],[],[]
    start = perf_counter()
    for tile,crop,rgb,binary,_ in iter_patches(protocol,'development'):
        features = feature_grid(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),4)
        labels = binary[::4,::4].copy()
        # Özelliklerin yansıtma sınırı eğitim örneği olmaz (9x9 doku penceresi).
        labels[:2] = labels[-2:] = -1
        labels[:,:2] = labels[:,-2:] = -1
        flat = features.reshape(-1,14)
        for value in (0,1):
            indices = np.flatnonzero(labels.ravel() == value)
            if len(indices):
                chosen = rng.choice(indices,min(spec['max_samples_per_class_per_patch'],len(indices)),replace=False)
                xs.append(flat[chosen]); ys.append(np.full(len(chosen),value,np.uint8))
        used.append({'tile':tile,'window':crop})
        if len(used)%10 == 0:
            print('Eğitim özellikleri:',len(used),'pencere',flush=True)
    if not ys or len(np.unique(np.concatenate(ys))) != 2:
        raise ValueError('Eğitim iki sınıfı da içermeli.')
    x,y = np.concatenate(xs),np.concatenate(ys)
    model = ExtraTreesClassifier(n_estimators=96,max_depth=18,min_samples_leaf=8,
                                class_weight='balanced',random_state=spec['seed'],n_jobs=2)
    model.fit(x,y)
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({'feature_version':FEATURE_VERSION,'model':model},output,compress=3)
    metadata = {'feature_version':FEATURE_VERSION,'dataset':'ISPRS Potsdam / BSF Swissphoto',
                'training_split':'development','used_windows':used,'pixel_count':len(y),
                'class_counts':np.bincount(y).tolist(),'seed':spec['seed'],
                'protocol_sha256':sha256(protocol),'model_sha256':sha256(output),
                'sklearn_version':sklearn.__version__,'parameters':model.get_params(),
                'feature_grid_step':4,'positive_class':'impervious surfaces; not physical safety',
                'seconds':perf_counter()-start,
                'code_sha256':{name:sha256(Path(__file__).resolve().parents[1]/name) for name in
                               ('potsdam_data.py','surface_classification.py','scripts/train_potsdam_surface.py')}}
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return metadata


def select_threshold(protocol, model_path, destination):
    spec = json.loads(Path(protocol).read_text(encoding='utf-8'))
    training = json.loads(Path(model_path).with_suffix('.json').read_text(encoding='utf-8'))
    if (training['training_split'] != 'development' or training['protocol_sha256'] != sha256(protocol)
            or training['model_sha256'] != sha256(model_path)):
        raise ValueError('Modelin eğitim kaydı dondurulmuş protokolle uyuşmuyor.')
    predictor = SurfacePredictor(SurfaceConfig(model_path=str(model_path),grid_step=4))
    totals = {threshold:np.zeros(4,dtype='int64') for threshold in spec['threshold_candidates']}
    windows_used = []
    border = spec['feature_border_px']
    for tile,crop,rgb,binary,_ in iter_patches(protocol,'validation'):
        score = predictor.predict(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)).probability
        for threshold,counts in totals.items():
            counts += metric_counts(score[border:-border,border:-border],binary[border:-border,border:-border],threshold)
        windows_used.append({'tile':tile,'window':crop})
    if not windows_used or any(sum(counts[[0,2]]) == 0 or sum(counts[[1,3]]) == 0 for counts in totals.values()):
        raise ValueError('Doğrulama iki sınıfı da içeren pencereler gerektirir.')
    rows = [{'threshold':threshold,**metrics(counts)} for threshold,counts in totals.items()]
    best = max(rows,key=lambda row:(row['f0_5'],row['precision'] or 0,row['threshold']))
    selection = {'threshold':best['threshold'],'selection_split':'validation','windows':windows_used,
                 'selection_rule':spec['selection'],'candidates':rows,'model_sha256':sha256(model_path),
                 'protocol_sha256':sha256(protocol),'feature_grid_step':4,
                 'note':'Frozen before test. F0.5 favors semantic precision, not physical safety.'}
    Path(destination).write_text(json.dumps(selection,indent=2),encoding='utf-8')
    return selection


def evaluate_test(protocol, model_path, selection_path, baseline_path, output):
    selection = json.loads(Path(selection_path).read_text(encoding='utf-8'))
    if selection['selection_split'] != 'validation' or selection['model_sha256'] != sha256(model_path) or selection['protocol_sha256'] != sha256(protocol):
        raise ValueError('Dondurulmuş model/protokol/eşik kaydı uyuşmuyor.')
    spec = json.loads(Path(protocol).read_text(encoding='utf-8'))
    models = {'aeroscapes_baseline':SurfacePredictor(SurfaceConfig(model_path=str(baseline_path),threshold=.6,grid_step=4)),
              'potsdam_native':SurfacePredictor(SurfaceConfig(model_path=str(model_path),threshold=selection['threshold'],grid_step=4))}
    totals = {name:np.zeros(4,dtype='int64') for name in models}
    class_totals = {name:np.zeros((len(PALETTE),2),dtype='int64') for name in models}
    per_window = []
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    border = spec['feature_border_px']
    for tile,crop,rgb,binary,classes in iter_patches(protocol,'test'):
        bgr = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        target = binary[border:-border,border:-border]
        codes = classes[border:-border,border:-border]
        previews = [bgr]
        for name,predictor in models.items():
            score = predictor.predict(bgr).probability
            prediction = score >= predictor.config.threshold
            counts = metric_counts(score[border:-border,border:-border],target,predictor.config.threshold)
            totals[name] += counts
            for index in range(len(PALETTE)):
                mask = codes == index
                class_totals[name][index] += [int(mask.sum()),int((prediction[border:-border,border:-border] & mask).sum())]
            per_window.append({'tile':tile,'window':crop,'model':name,**metrics(counts)})
            if len(per_window) <= len(models):
                painted = bgr.copy()
                painted[prediction] = (40,200,40)
                previews.append(cv2.addWeighted(bgr,.6,painted,.4,0))
        if len(per_window) == len(models):
            labels = np.zeros_like(bgr)
            labels[binary == 1] = (255,255,255)
            previews.append(labels)
            panel = np.hstack(previews)
            for index,name in enumerate(['RGB','AeroScapes 0.60',f"Potsdam {selection['threshold']:.2f}",'Reference target']):
                cv2.putText(panel,name,(index*rgb.shape[1]+10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
            cv2.imencode('.png',panel)[1].tofile(output/'test_preview.png')
        if len(per_window)%20 == 0:
            print('Test:',len(per_window)//len(models),'pencere',flush=True)
    if not per_window:
        raise ValueError('Test penceresi bulunamadı.')
    summary = {'protocol_sha256':sha256(protocol),'selection_sha256':sha256(selection_path),
               'test_tiles':sorted({row['tile'] for row in per_window}),
               'note':'Guarded tile holdout from one acquisition; semantic target, not physical safety.',
               'models':{},'per_window':per_window}
    for name,predictor in models.items():
        summary['models'][name] = {**metrics(totals[name]),'threshold':predictor.config.threshold,
            'model_sha256':predictor.model_sha256,
            'accepted_fraction_by_class':{key:{'pixels':int(class_totals[name][i,0]),
                'accepted':int(class_totals[name][i,1]),
                'fraction':float(class_totals[name][i,1]/class_totals[name][i,0]) if class_totals[name][i,0] else None}
                for i,key in enumerate(PALETTE)}}
    (output/'test.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    return summary


def study(source, output, model_path, baseline):
    if (output/'test.json').exists():
        raise ValueError('Bu çalışma test sonuçlarını zaten içeriyor; aynı çalışmada yeniden eğitim yapılmadı.')
    output.mkdir(parents=True,exist_ok=True)
    protocol = output/'protocol.json'
    build_protocol(source,protocol)
    training = train(protocol,model_path)
    select_threshold(protocol,model_path,output/'selection.json')
    result = evaluate_test(protocol,model_path,output/'selection.json',baseline,output)
    result['training_pixels'] = training['pixel_count']
    lines = ['# Hedef veriyle yüzey modeli geliştirme','',
             'Eğitim: 2_11 ve 3_11. Doğrulama: 4_11. Dondurulmuş test: 5_11. '
             'Önceden incelenen 2_10 kullanılmadı. Her karo kenarında 128 piksel pay, '
             '512×512 pencereler ve 768 piksel adım kullanıldı. Özellik sınırlarının '
             '8 pikseli ölçüm dışında. Karolar aynı kentsel çekimdendir; bağımsız '
             'rover sahası değildir. Beyaz impervious sınıfı hedef, diğer tanınan '
             'renkler hedef dışıdır; fiziksel güvenlik doğrusu yoktur.','',
             '| Model | Eşik | Kesinlik | Duyarlılık | IoU | F0.5 |',
             '| --- | --- | --- | --- | --- | --- |']
    for name,values in result['models'].items():
        numbers = ['—' if values[key] is None else f"{values[key]:.4f}" for key in ('precision','recall','iou','f0_5')]
        lines.append(f"| {name} | {values['threshold']:.2f} | " + ' | '.join(numbers) + ' |')
    lines += ['',f"Eğitimde {training['pixel_count']:,} dengeli piksel örneği kullanıldı. "
              'Eşik yalnızca doğrulamada F0.5 ile seçilip testten önce kaydedildi. '
              'Test sonrası ayar değiştirilmedi. Model kaynak özetleri ve bölümler '
              'protocol.json, selection.json, test.json içinde kayıtlıdır.','',
              'Bu sonuç sadece yüzey sınıflandırmasıdır. Rover başlangıç/rota '
              'doğruluğunu, taşınabilirliği veya fiziksel emniyeti tek başına doğrulamaz. '
              'Araç, çatı ve diğer sınıflardaki yanlış kabuller test.json içinde '
              'ayrıca raporlanır.','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,default=Path('data/datasets/potsdam'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/potsdam_native_study'))
    parser.add_argument('--model',type=Path,default=Path('models/surface_potsdam.joblib'))
    parser.add_argument('--baseline',type=Path,default=Path('models/surface.joblib'))
    args = parser.parse_args()
    print(json.dumps(study(args.source_dir,args.output_dir,args.model,args.baseline)['models'],indent=2))
