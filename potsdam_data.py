"""Potsdam karolarını mekânsal olarak ayırır ve sabit renk etiketlerini okur."""

import hashlib
import json
import os
from pathlib import Path
import re
import numpy as np
import rasterio
from rasterio.windows import Window

PALETTE = {'impervious':(255,255,255),'building':(0,0,255),
           'low_vegetation':(0,255,255),'tree':(0,255,0),'car':(255,255,0),'clutter':(255,0,0)}
SPLITS = {'development':['2_11','3_11'],'validation':['4_11'],'test':['5_11']}


def tile_names(tile):
    if not isinstance(tile,str) or not re.fullmatch(r'[1-9][0-9]*_[1-9][0-9]*',tile):
        raise ValueError('Karo adı satır_sütun biçiminde olmalı: 2_11.')
    row,column = map(int,tile.split('_'))
    return f'top_potsdam_{tile}_RGB.tif',f'top_potsdam_{tile}_label.tif',f'dsm_potsdam_{row:02d}_{column:02d}'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def windows(width, height, patch_size, stride, guard):
    if any(type(value) is not int for value in (width,height,patch_size,stride,guard)):
        raise ValueError('Pencere ayarları tam sayı olmalı.')
    if patch_size < 32 or patch_size % 4 or stride < patch_size or guard < 8:
        raise ValueError('Pencere en az 32 ve 4 katı; adım pencere kadar; kenar payı en az 8 olmalı.')
    return [(x,y,patch_size,patch_size) for y in range(guard,height-guard-patch_size+1,stride)
            for x in range(guard,width-guard-patch_size+1,stride)]


def build_protocol(source, destination, splits=None, patch_size=512, stride=768, guard=128):
    splits = SPLITS if splits is None else splits
    if set(splits) != {'development','validation','test'} or any(not v for v in splits.values()):
        raise ValueError('Eğitim, doğrulama ve test için ayrı, boş olmayan karo listeleri gerekli.')
    seen,images,labels_seen = set(),{},{}
    entries = []
    destination = Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    for split,tiles in splits.items():
        for tile in tiles:
            if tile in seen or tile == '2_10':
                raise ValueError('Karo bölümler arasında paylaşılmaz; incelenmiş 2_10 yeni teste alınmaz.')
            seen.add(tile)
            rgb,label,_ = tile_names(tile)
            rgb_path,label_path = Path(source)/rgb,Path(source)/label
            image_hash,label_hash = sha256(rgb_path),sha256(label_path)
            for registry,digest in ((images,image_hash),(labels_seen,label_hash)):
                if digest in registry and registry[digest] != split:
                    raise ValueError('Bölümler arasında birebir aynı görüntü veya etiket var.')
                registry[digest] = split
            with rasterio.open(rgb_path) as src,rasterio.open(label_path) as target:
                if src.count != 3 or src.dtypes != ('uint8',)*3 or target.count != 3 or target.dtypes != ('uint8',)*3:
                    raise ValueError('RGB ve renk etiketi üç uint8 bantlı olmalı.')
                if src.shape != target.shape or not src.crs or not src.crs.is_projected or src.crs.linear_units != 'metre':
                    raise ValueError('Boyutlar eşleşmeli ve RGB metre tabanlı projeksiyonda olmalı.')
                if src.transform.b or src.transform.d or src.transform.a <= 0 or src.transform.e >= 0:
                    raise ValueError('Döndürülmemiş, kuzey yönlü RGB gerekli.')
                if target.crs is not None and (target.crs != src.crs or target.transform != src.transform):
                    raise ValueError('Etiketin mevcut coğrafi referansı RGB ile uyuşmuyor.')
                patches = windows(src.width,src.height,patch_size,stride,guard)
                if not patches:
                    raise ValueError('Karo pencere ve kenar payı için fazla küçük.')
                bounds = list(src.bounds)
                crs = src.crs.to_string()
                for previous in entries:
                    if previous['crs'] != crs:
                        raise ValueError('Bu protokol aynı CRS içinde karolar gerektirir.')
                    left,bottom,right,top = previous['bounds']
                    if (previous['split'] != split and min(right,bounds[2])-max(left,bounds[0]) > 1e-6
                            and min(top,bounds[3])-max(bottom,bounds[1]) > 1e-6):
                        raise ValueError('Farklı bölümlerin karo coğrafi alanları örtüşüyor.')
            entries.append({'tile':tile,'split':split,'rgb':os.path.relpath(rgb_path.resolve(),destination.parent.resolve()),
                            'labels':os.path.relpath(label_path.resolve(),destination.parent.resolve()),
                            'image_sha256':image_hash,'label_sha256':label_hash,'bounds':bounds,'crs':crs,'windows':patches})
    spec = {'dataset':'ISPRS Potsdam / BSF Swissphoto','entries':entries,'feature_grid_step':4,
            'patch_size':patch_size,'stride':stride,'guard_px':guard,'feature_border_px':8,
            'excluded_inspected_tile':'2_10','palette':PALETTE,'positive_class':'impervious',
            'threshold_candidates':[.5,.6,.7,.8,.9],'selection':'maximum validation F0.5, then precision, then higher threshold',
            'seed':42,'max_samples_per_class_per_patch':256,
            'source':'https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx',
            'limitations':'Different guarded tiles from one urban acquisition; not independent rover terrains. Semantic target is not physical safety.'}
    destination.write_text(json.dumps(spec,indent=2),encoding='utf-8')
    return spec


def decode_labels(rgb):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Etiket uint8 RGB olmalı.')
    classes = np.full(rgb.shape[:2],-1,dtype='int8')
    for index,color in enumerate(PALETTE.values()):
        classes[np.all(rgb == color,axis=2)] = index
    # Tanınmayan renkler (ör. siyah sınırlar) eğitim/ölçüm dışında kalır.
    binary = np.where(classes < 0,-1,classes == 0).astype('int8')
    return binary,classes


def iter_patches(protocol_path, split):
    path = Path(protocol_path)
    spec = json.loads(path.read_text(encoding='utf-8'))
    if split not in ('development','validation','test'):
        raise ValueError('Bilinmeyen veri bölümü.')
    for entry in spec['entries']:
        if entry['split'] != split:
            continue
        rgb_path,label_path = path.parent/entry['rgb'],path.parent/entry['labels']
        if sha256(rgb_path) != entry['image_sha256'] or sha256(label_path) != entry['label_sha256']:
            raise ValueError('Kaynak dosya dondurulmuş protokolle uyuşmuyor.')
        with rasterio.open(rgb_path) as src,rasterio.open(label_path) as target:
            for crop in entry['windows']:
                window = Window(*crop)
                rgb = src.read([1,2,3],window=window).transpose(1,2,0)
                label = target.read([1,2,3],window=window).transpose(1,2,0)
                binary,classes = decode_labels(label)
                valid = np.all(src.read_masks([1,2,3],window=window) > 0,axis=0)
                binary[~valid] = -1
                classes[~valid] = -1
                yield entry['tile'],crop,rgb,binary,classes
