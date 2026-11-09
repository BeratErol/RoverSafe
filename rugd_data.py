"""Doğal arazi sekanslarında bölüm ayrımı ve dosya bütünlüğünü korur."""

import json
from pathlib import Path
import numpy as np
from PIL import Image
from potsdam_data import sha256


def load_manifest(path):
    spec = json.loads(Path(path).read_text(encoding='utf-8'))
    if spec.get('split') != 'development_validation_test' or not spec.get('samples'):
        raise ValueError('RUGD eğitim/doğrulama/test manifesti gerekli.')
    groups,hashes,ids = {},{},set()
    for sample in spec['samples']:
        split = sample['split']
        if split not in ('development','validation','test') or sample['id'] in ids:
            raise ValueError('Bilinmeyen bölüm veya tekrar eden örnek kimliği.')
        ids.add(sample['id'])
        for registry,key in ((groups,sample['scene_id']),(hashes,sample['image_sha256'])):
            if key in registry and registry[key] != split:
                raise ValueError('Aynı sekans veya aynı görüntü bölümler arasında paylaşılamaz.')
            registry[key] = split
    if set(groups.values()) != {'development','validation','test'}:
        raise ValueError('Üç veri bölümü de dolu olmalı.')
    policy = spec['class_policy']
    for a,b in (('target','risk'),('target','unknown'),('risk','unknown')):
        if set(policy[a]) & set(policy[b]):
            raise ValueError('Etiket politikaları çakışıyor.')
    return spec


def iter_samples(path, split):
    spec = load_manifest(path)
    for sample in spec['samples']:
        if sample['split'] != split:
            continue
        if sha256(sample['image']) != sample['image_sha256'] or sha256(sample['mask']) != sample['mask_sha256']:
            raise ValueError('RUGD görüntü veya etiket dosyası değişmiş.')
        with Image.open(sample['image']) as source:
            image = np.array(source.convert('RGB'))
        with Image.open(sample['mask']) as source:
            labels = np.array(source)
        policy = spec['class_policy']
        if image.shape[:2] != labels.shape or not np.isin(labels,sum(policy.values(),[])).all():
            raise ValueError('RUGD etiket boyutu veya sınıfları geçersiz.')
        binary = np.full(labels.shape,-1,'int8')
        binary[np.isin(labels,policy['target'])] = 1
        binary[np.isin(labels,policy['risk'])] = 0
        yield sample,image,binary
