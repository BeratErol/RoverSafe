"""İki dondurulmuş hava modelini aynı test karosu ve piksel sınırında karşılaştırır."""

import argparse
from html import escape
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from potsdam_data import iter_patches,sha256,PALETTE
from surface_classification import SurfaceConfig,SurfacePredictor
from scripts.train_dense_aerial import f1_metrics
from scripts.train_potsdam_surface import metric_counts


def compare(protocol,baseline_selection,improved_selection,output):
    output=Path(output)
    if output.exists() and any(output.iterdir()): raise FileExistsError('Yeni çıktı klasörü gerekli.')
    predictors={}; selections={}
    for name,path in (('baseline',baseline_selection),('improved',improved_selection)):
        selected=json.loads(Path(path).read_text(encoding='utf-8'))
        if selected['model_sha256']!=sha256(selected['selected_model_path']): raise ValueError('Model kaydı değişmiş.')
        predictors[name]=SurfacePredictor(SurfaceConfig(model_path=selected['selected_model_path'],encoder_path=selected['encoder_path'],
            threshold=selected['threshold'],expected_viewpoint='aerial'))
        selections[name]={'selection_sha256':sha256(path),'model_sha256':selected['model_sha256'],'threshold':selected['threshold']}
    output.mkdir(parents=True,exist_ok=True)
    totals={name:np.zeros(4,'int64') for name in predictors}
    class_counts={name:np.zeros((6,2),'int64') for name in predictors}
    rows=[]; cards=[]
    for tile,crop,rgb,binary,classes in iter_patches(protocol,'test'):
        image=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR); panels=[image]
        for name,predictor in predictors.items():
            score=predictor.predict(image).probability
            counts=metric_counts(score[20:-20,20:-20],binary[20:-20,20:-20],predictor.config.threshold)
            totals[name]+=counts; prediction=score>=predictor.config.threshold
            codes=classes[20:-20,20:-20]
            for index in range(6):
                mask=codes==index
                class_counts[name][index]+=[mask.sum(),(prediction[20:-20,20:-20]&mask).sum()]
            rows.append({'tile':tile,'window':crop,'model':name,**f1_metrics(counts)})
            painted=image.copy(); painted[prediction & (binary==1)]=(40,200,40)
            painted[prediction & (binary==0)]=(40,40,220); painted[~prediction & (binary==1)]=(230,100,20)
            painted[binary<0]=(120,120,120)
            panels.append(cv2.addWeighted(image,.4,painted,.6,0))
        if len(cards)<8:
            filename=f'{tile}_{crop[0]}_{crop[1]}.png'
            cv2.imencode('.png',np.hstack(panels))[1].tofile(output/filename)
            cards.append(f'<article><h2>{escape(filename)}</h2><img src="{escape(filename)}"></article>')
        if len(rows)%20==0: print('Urban comparison',len(rows)//2,'patches',flush=True)
    summary={'protocol_sha256':sha256(protocol),'models':{name:{**f1_metrics(totals[name]),**selections[name],
        'accepted_fraction_by_class':{key:float(v[1]/v[0]) if v[0] else None for key,v in zip(PALETTE,class_counts[name])}}
        for name in predictors},'per_window':rows,
        'interpretation':'Same held-out tile, border and resolution. Improvement combines architecture and additional development data; not a controlled architecture-only ablation or physical safety test.'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    page='<!doctype html><html lang="tr"><meta charset="utf-8"><title>Hava modeli karşılaştırması</title><style>body{font:17px system-ui;background:#101820;color:#eef3f5;max-width:1500px;margin:24px auto;padding:16px}img{width:100%}pre{white-space:pre-wrap}</style>'
    page+='<h1>Aynı hava verisinde model karşılaştırması</h1><p>Sol: RGB. Orta: önceki model. Sağ: yeni model. Yeşil doğru kabul, kırmızı yanlış kabul, mavi kaçırılan hedef, gri bilinmeyen.</p>'
    page+='<pre>'+escape(json.dumps(summary['models'],indent=2))+'</pre>'+''.join(cards)+'</html>'
    (output/'report.html').write_text(page,encoding='utf-8')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=Path('outputs/dense_expanded_protocol.json'))
    parser.add_argument('--baseline',type=Path,default=Path('outputs/context_aerial/selection.json'))
    parser.add_argument('--improved',type=Path,default=Path('outputs/dense_aerial_full/selection.json'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/aerial_urban_comparison'))
    args=parser.parse_args()
    compare(args.protocol,args.baseline,args.improved,args.output_dir)
