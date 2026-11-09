"""Dondurulmuş görsel kodlayıcıyla bağlam modeli; doğrulamada eşik, sonra tek test."""

import argparse
import json
from pathlib import Path
import sys
from itertools import chain
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from context_features import ContextEncoder,CONTEXT_VERSION
from potsdam_data import iter_patches,sha256
from rugd_data import iter_samples
from scripts.train_potsdam_surface import metric_counts,metrics
from scripts.improve_surface import choose


def additional_samples(path):
    """Eski dış kıyas kümesi açıkça emekliye ayrılmadan eğitime alınmaz."""
    from PIL import Image
    spec = json.loads(Path(path).read_text(encoding='utf-8'))
    if spec.get('retired_external_for_development') is not True or any(s['split'] != 'development' for s in spec['samples']):
        raise ValueError('Ek veri yalnız açıkça development olarak yeniden tanımlanmış eski dış küme olabilir.')
    root = Path(__file__).resolve().parents[1]
    for sample in sorted(spec['samples'],key=lambda s:s['id'])[::4]:
        image_path,label_path = root/sample['image'],root/sample['mask']
        if sha256(image_path) != sample['files']['image']['sha256'] or sha256(label_path) != sample['files']['mask']['sha256']:
            raise ValueError('Ek eğitim görüntüsü/etiketi özeti uyuşmuyor.')
        with Image.open(image_path) as stream: rgb = np.array(stream.convert('RGB'))
        with Image.open(label_path) as stream: label = np.array(stream)
        scale = min(1.,768/max(rgb.shape[:2]))
        if scale < 1:
            width,height = round(rgb.shape[1]*scale),round(rgb.shape[0]*scale)
            rgb = cv2.resize(rgb,(width,height),interpolation=cv2.INTER_AREA)
            label = cv2.resize(label,(width,height),interpolation=cv2.INTER_NEAREST)
        binary = np.full(label.shape,-1,'int8')
        binary[np.isin(label,spec['class_policy']['target'])] = 1
        binary[np.isin(label,spec['class_policy']['risk'])] = 0
        yield sample['id'],rgb,binary


def samples(path,domain,split):
    if domain == 'aerial':
        for tile,crop,rgb,binary,_ in iter_patches(path,split):
            yield f'{tile}_{crop[0]}_{crop[1]}',rgb,binary
    else:
        for record,rgb,binary in iter_samples(path,split):
            yield record['id'],rgb,binary


def scores(classifier,encoder,rgb):
    features = encoder.features(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),4)
    coarse = classifier.predict_proba(features.reshape(-1,419))[:,1].reshape(features.shape[:2])
    return coarse.repeat(4,0).repeat(4,1)[:rgb.shape[0],:rgb.shape[1]]


def study(manifest,weights,domain,output,external_test=None,additional_development=None):
    if external_test and domain != 'ground':
        raise ValueError('RUGD dış test manifesti yalnız ground bağlamında kullanılabilir.')
    if additional_development and domain != 'aerial':
        raise ValueError('Bu ek geliştirme veri seçeneği yalnız aerial içindir.')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Bağlam eğitimi için yeni veya boş klasör seçin.')
    output.mkdir(parents=True,exist_ok=True)
    plan = {'manifest_sha256':sha256(manifest),'domain':domain,'encoder_sha256':sha256(weights),
            'thresholds':[.5,.6,.7,.8,.9],'selection_rule':'maximum validation F1 with precision>=0.8 and recall>=0.4; otherwise maximum F1 and failed gate',
            'encoder_frozen':True,'feature_version':CONTEXT_VERSION,'grid_step':4,'evaluation_border_px':20,
            'external_test_manifest_sha256':sha256(external_test) if external_test else None}
    plan['additional_development_sha256'] = sha256(additional_development) if additional_development else None
    plan['additional_sampling_stride'] = 4 if additional_development else None
    (output/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    encoder = ContextEncoder(weights)
    rng = np.random.default_rng(42)
    xs,ys,used = [],[],[]
    training_iterator = chain(samples(manifest,domain,'development'),
                              additional_samples(additional_development) if additional_development else ())
    for name,rgb,binary in training_iterator:
        features = encoder.features(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),4).reshape(-1,419)
        labels = binary[::4,::4].copy()
        labels[:5] = labels[-5:] = -1; labels[:,:5] = labels[:,-5:] = -1
        for code in (0,1):
            indices = np.flatnonzero(labels.ravel() == code)
            if len(indices):
                chosen = rng.choice(indices,min(512,len(indices)),replace=False)
                xs.append(features[chosen]); ys.append(np.full(len(chosen),code,'uint8'))
        used.append(name)
        if len(used)%10 == 0:
            print('Context training',len(used),'images',flush=True)
    if not ys or len(np.unique(np.concatenate(ys))) != 2:
        raise ValueError('Bağlam eğitimi iki sınıfı da gerektirir.')
    x,y = np.concatenate(xs),np.concatenate(ys)
    classifier = ExtraTreesClassifier(n_estimators=96,max_depth=22,min_samples_leaf=4,class_weight='balanced',random_state=42,n_jobs=2)
    classifier.fit(x,y)
    model = output/'surface_context.joblib'
    joblib.dump({'feature_version':CONTEXT_VERSION,'viewpoint':domain,'encoder_sha256':encoder.sha256,'model':classifier},model,compress=3)
    training = {'training_split':'development','sample_ids':used,'pixel_count':len(y),'feature_version':CONTEXT_VERSION,
                'manifest_sha256':sha256(manifest),'model_sha256':sha256(model),'encoder_sha256':encoder.sha256,
                'parameters':classifier.get_params(),'viewpoint':domain}
    training['additional_development_sha256'] = plan['additional_development_sha256']
    model.with_suffix('.json').write_text(json.dumps(training,indent=2),encoding='utf-8')
    counts = {t:np.zeros(4,'int64') for t in plan['thresholds']}
    validation_ids = []
    for name,rgb,binary in samples(manifest,domain,'validation'):
        score = scores(classifier,encoder,rgb)
        for threshold in counts:
            counts[threshold] += metric_counts(score[20:-20,20:-20],binary[20:-20,20:-20],threshold)
        validation_ids.append(name)
        if len(validation_ids)%10 == 0:
            print('Context validation',len(validation_ids),'images',flush=True)
    candidates = []
    for threshold,value in counts.items():
        tp,fp,fn,_ = value
        candidates.append({'model':CONTEXT_VERSION,'threshold':threshold,**metrics(value),
                           'f1':float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0})
    best,passed = choose(candidates)
    selection = {'selection_split':'validation','threshold':best['threshold'],'feature_grid_step':4,
                 'selected_model_path':str(model.resolve()),'model_sha256':sha256(model),'feature_version':CONTEXT_VERSION,
                 'encoder_path':str(Path(weights).resolve()),'encoder_sha256':encoder.sha256,
                 'manifest_sha256':sha256(manifest),'passes_validation_gate':passed,'candidates':candidates,
                 'protocol_sha256':sha256(manifest),
                 'validation_sample_ids':validation_ids,'test_read_before_selection':False,'viewpoint':domain}
    (output/'selection.json').write_text(json.dumps(selection,indent=2,allow_nan=False),encoding='utf-8')
    total = np.zeros(4,'int64')
    test_ids = []
    if external_test:
        # Dış örnekler eğitim/doğrulama yollarına eklenmez.
        from PIL import Image
        external = json.loads(Path(external_test).read_text(encoding='utf-8'))
        if external.get('split') != 'external_test':
            raise ValueError('Dondurulmuş dış test manifesti gerekli.')
        seen_scenes = {s['scene_id'] for s in json.loads(Path(manifest).read_text(encoding='utf-8'))['samples']}
        if any(s['scene_id'] in seen_scenes for s in external['samples']):
            raise ValueError('Yeni dış test sekansları eğitim protokolünden ayrık olmalı.')
        def external_iterator():
            for sample in external['samples']:
                if sha256(sample['image']) != sample['image_sha256'] or sha256(sample['mask']) != sample['mask_sha256']:
                    raise ValueError('Dış test dosya özeti uyuşmuyor.')
                with Image.open(sample['image']) as stream: rgb = np.array(stream.convert('RGB'))
                with Image.open(sample['mask']) as stream: label = np.array(stream)
                binary = np.full(label.shape,-1,'int8')
                binary[np.isin(label,external['class_policy']['target'])] = 1
                binary[np.isin(label,external['class_policy']['risk'])] = 0
                yield sample['id'],rgb,binary
        test_iterator = external_iterator()
    else:
        test_iterator = samples(manifest,domain,'test')
    for name,rgb,binary in test_iterator:
        score = scores(classifier,encoder,rgb)
        total += metric_counts(score[20:-20,20:-20],binary[20:-20,20:-20],best['threshold'])
        test_ids.append(name)
    if not validation_ids or not test_ids:
        raise ValueError('Doğrulama ve test boş olamaz.')
    tp,fp,fn,_ = total
    test = {**metrics(total),'f1':float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0}
    result = {'selection_sha256':sha256(output/'selection.json'),'selected':selection,'test':test,'test_sample_ids':test_ids,
              'passes_test_gate':(test['precision'] or 0) >= .8 and (test['recall'] or 0) >= .4,
              'interpretation':'Semantic image test, not physical rover safety. Aerial protocol test may have been evaluated in prior studies; external ground sequences are separate.'}
    (output/'test.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--domain',choices=['aerial','ground'],required=True)
    parser.add_argument('--encoder',type=Path,default=Path('models/resnet18-f37072fd.pth'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--external-test-manifest',type=Path)
    parser.add_argument('--additional-development-manifest',type=Path)
    args = parser.parse_args()
    result = study(args.manifest,args.encoder,args.domain,args.output_dir,args.external_test_manifest,args.additional_development_manifest)
    print(json.dumps(result['test'],indent=2))
