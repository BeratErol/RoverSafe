"""Hava görüntüsünde yoğun başlık eğitir; modeli ve eşiği yalnız doğrulamada seçer."""

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from context_features import ContextEncoder
from dense_surface import make_head,DENSE_VERSION
from potsdam_data import iter_patches,sha256
from scripts.train_potsdam_surface import metric_counts,metrics
from scripts.improve_surface import choose


def f1_metrics(counts):
    tp,fp,fn,_ = counts
    return {**metrics(counts),'f1':float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0}


def study(protocol,weights,output,epochs=16,resume_cache=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not resume_cache:
        raise FileExistsError('Eğitim için yeni veya boş bir klasör gerekli.')
    if resume_cache:
        previous=json.loads((output/'plan.json').read_text(encoding='utf-8'))
        if ((output/'selection.json').exists() or (output/'test.json').exists()
                or previous['protocol_sha256'] != sha256(protocol)
                or previous['encoder_sha256'] != sha256(weights)):
            raise ValueError('Yalnız henüz seçim/test yapılmamış aynı protokolün özellik önbelleği kullanılabilir.')
    if type(epochs) is not int or epochs < 1:
        raise ValueError('Epoch sayısı pozitif tam sayı olmalı.')
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    encoder = ContextEncoder(weights)
    plan = {'protocol_sha256':sha256(protocol),'encoder_sha256':encoder.sha256,'epochs':epochs,
            'seed':42,'selection_split':'validation','thresholds':[.5,.6,.7,.8,.9],
            'selection_rule':'maximum validation F1 with precision>=0.8 and recall>=0.4; otherwise failed gate',
            'feature_version':DENSE_VERSION,'loss':'six-class cross entropy, inverse-square-root class weighting',
            'test_role':'previously evaluated Potsdam tile; regression only, not independent natural terrain'}
    (output/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    cache = output/'cache'; cache.mkdir(exist_ok=resume_cache)
    development,validation = [],[]
    moments = []
    class_counts = np.zeros(6,'int64')
    for split,records in (('development',development),('validation',validation)):
        for tile,crop,rgb,_,classes in iter_patches(protocol,split):
            name = f'{tile}_{crop[0]}_{crop[1]}'
            label = classes[::4,::4].astype('int64').copy()
            label[:5] = label[-5:] = -1; label[:,:5] = label[:,-5:] = -1
            path = cache/(name+'.npz')
            if resume_cache and path.exists():
                with np.load(path) as data:
                    features=data['features'].astype('float32')
                    if not np.array_equal(label,data['labels']):
                        raise ValueError('Önbellek etiketi kaynakla eşleşmiyor.')
            else:
                features = encoder.features(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),4)
                np.savez(path,features=features.astype('float16'),labels=label)
            records.append((name,path))
            if split == 'development':
                moments.append(features[5:-5:4,5:-5:4].reshape(-1,419))
                class_counts += np.bincount(label[label >= 0],minlength=6)
            if len(records)%10 == 0: print(split,'cached',len(records),flush=True)
    if not development or not validation or (class_counts == 0).any():
        raise ValueError('Geliştirme ve doğrulama, geliştirmede altı sınıf da gerekli.')
    stats = np.concatenate(moments)
    mean = torch.tensor(stats.mean(0)).view(1,419,1,1)
    std = torch.tensor(np.maximum(stats.std(0),.02)).view(1,419,1,1)
    del moments,stats
    head = make_head()
    optimizer = torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.001)
    class_weights = torch.tensor(1/np.sqrt(class_counts/class_counts.sum()),dtype=torch.float32)
    class_weights /= class_weights.mean()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights,ignore_index=-1)
    history,best = [],None
    def tensors(path):
        with np.load(path) as data:
            x = torch.from_numpy(data['features'].astype('float32').transpose(2,0,1).copy()).unsqueeze(0)
            y = torch.from_numpy(data['labels'].copy()).unsqueeze(0)
        return (x-mean)/std,y
    for epoch in range(1,epochs+1):
        head.train(); losses=[]
        for index in rng.permutation(len(development)):
            x,y = tensors(development[index][1])
            if rng.random() < .5: x,y = x.flip(-1),y.flip(-1)
            if rng.random() < .5: x,y = x.flip(-2),y.flip(-2)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(x),y)
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        head.eval()
        totals = {t:np.zeros(4,'int64') for t in plan['thresholds']}
        with torch.inference_mode():
            for _,path in validation:
                x,y = tensors(path)
                score = head(x).softmax(1)[0,0].numpy()
                labels = y[0].numpy()
                binary = np.where(labels < 0,-1,labels == 0)
                for t in totals: totals[t] += metric_counts(score,binary,t)
        rows = [{'model':DENSE_VERSION,'threshold':t,**f1_metrics(v)} for t,v in totals.items()]
        selected,passed = choose(rows)
        rank = (passed,selected['f1'])
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),'selected':selected,'passes_gate':passed})
        print(json.dumps(history[-1]),flush=True)
        if best is None or rank > best:
            best = rank
            torch.save({'feature_version':DENSE_VERSION,'viewpoint':'aerial','encoder_sha256':encoder.sha256,
                        'head':head.state_dict(),'mean':mean,'std':std},output/'surface_dense.pth')
            selection = {'selection_split':'validation','epoch':epoch,'threshold':selected['threshold'],
                         'feature_grid_step':4,'selected_model_path':str((output/'surface_dense.pth').resolve()),
                         'model_sha256':sha256(output/'surface_dense.pth'),'encoder_path':str(Path(weights).resolve()),
                         'encoder_sha256':encoder.sha256,'protocol_sha256':sha256(protocol),'feature_version':DENSE_VERSION,
                         'passes_validation_gate':passed,'candidates':rows,'viewpoint':'aerial',
                         'validation_sample_ids':[v[0] for v in validation],'test_read_before_selection':False,
                         'validation_grid_step':4,'note':'Validation metrics on 4-pixel sampling; test reports full-resolution metrics.'}
            (output/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
        (output/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    training = {'sample_ids':[v[0] for v in development],'class_counts':class_counts.tolist(),
                'parameters':plan,'torch_version':str(torch.__version__)}
    (output/'training.json').write_text(json.dumps(training,indent=2),encoding='utf-8')
    # Son değerlendirme aynı üretim tahmin yolunu kullanır; seçim artık donmuştur.
    from surface_classification import SurfacePredictor,SurfaceConfig
    selection = json.loads((output/'selection.json').read_text(encoding='utf-8'))
    predictor = SurfacePredictor(SurfaceConfig(model_path=selection['selected_model_path'],threshold=selection['threshold'],
        encoder_path=str(weights),expected_viewpoint='aerial'))
    counts=np.zeros(4,'int64'); ids=[]
    for tile,crop,rgb,binary,_ in iter_patches(protocol,'test'):
        score=predictor.predict(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)).probability
        counts += metric_counts(score[20:-20,20:-20],binary[20:-20,20:-20],selection['threshold'])
        ids.append(f'{tile}_{crop[0]}_{crop[1]}')
    result={'selection_sha256':sha256(output/'selection.json'),'test':f1_metrics(counts),'test_sample_ids':ids,
            'interpretation':plan['test_role']}
    result['passes_test_gate']=(result['test']['precision'] or 0)>=.8 and (result['test']['recall'] or 0)>=.4
    (output/'test.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Regression',json.dumps(result['test']),flush=True)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=Path('outputs/potsdam_native_study/protocol.json'))
    parser.add_argument('--encoder',type=Path,default=Path('models/resnet18-f37072fd.pth'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=16)
    parser.add_argument('--resume-cache',action='store_true')
    args=parser.parse_args()
    study(args.protocol,args.encoder,args.output_dir,args.epochs,args.resume_cache)
