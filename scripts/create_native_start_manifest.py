"""Doğrulamada seçilmiş model/eşiği mevcut metre tabanlı göreve bağlar."""

import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from potsdam_data import sha256


def create(base_manifest, destination, model, selection_path, protocol, yolo_model=None):
    spec = json.loads(base_manifest.read_text(encoding='utf-8'))
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    if (selection['selection_split'] != 'validation' or selection['model_sha256'] != sha256(model)
            or selection['protocol_sha256'] != sha256(protocol)):
        raise ValueError('Model/eşik/protokol doğrulama kaydıyla eşleşmiyor.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    for key in ('rgb','elevation','allowed_mask','start_zone'):
        absolute = (base_manifest.parent/spec[key]).resolve()
        spec[key] = os.path.relpath(absolute,destination.parent.resolve())
    if isinstance(spec.get('elevation_quality'),dict):
        for key in ('validity_mask','error_bound_raster'):
            if spec['elevation_quality'].get(key) is not None:
                absolute = (base_manifest.parent/spec['elevation_quality'][key]).resolve()
                spec['elevation_quality'][key] = os.path.relpath(absolute,destination.parent.resolve())
    spec.pop('ground_mask',None)
    spec['source_sha256'] = {k:v for k,v in spec.get('source_sha256',{}).items() if 'label' not in k.lower()}
    spec['ground_mode'] = 'predicted'
    spec['ground_mask_source'] = 'Locally trained Potsdam impervious-surface classifier; fixed validation threshold; no reference ground mask at inference.'
    spec['surface'] = {'model_path':os.path.relpath(model.resolve(),destination.parent.resolve()),
                       'threshold':selection['threshold'],'grid_step':selection['feature_grid_step']}
    spec['surface_selection'] = {'split':'validation','threshold':selection['threshold'],
                                 'model_sha256':selection['model_sha256'],
                                 'selection_sha256':sha256(selection_path),'protocol_sha256':sha256(protocol)}
    if yolo_model:
        spec['obstacles'] = {'model_path':os.path.relpath(yolo_model.resolve(),destination.parent.resolve()),
                             'margin_m':.2,'confidence':.25,'image_size':1280,'device':'cpu'}
    elif 'obstacles' in spec:
        # Taşınan eski nesne ayarı açıkça yeniden tanımlanmalıdır.
        raise ValueError('Kaynak manifestte nesne ayarı var; --yolo-model açıkça verilmeli.')
    destination.write_text(json.dumps(spec,indent=2),encoding='utf-8')
    return spec


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=Path('models/surface_potsdam.joblib'))
    parser.add_argument('--selection',type=Path,default=Path('outputs/potsdam_native_study/selection.json'))
    parser.add_argument('--protocol',type=Path,default=Path('outputs/potsdam_native_study/protocol.json'))
    parser.add_argument('--yolo-model',type=Path)
    args = parser.parse_args()
    create(args.base_manifest,args.output,args.model,args.selection,args.protocol,args.yolo_model)
    print(args.output)
