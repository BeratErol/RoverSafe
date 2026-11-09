"""Dondurulmuş ResNet özellikleri üzerinde öğrenilmiş yoğun yüzey bölümleme."""

from pathlib import Path
import hashlib
import numpy as np

DENSE_VERSION = 'resnet18_dense_v4'


def make_head(num_classes=6):
    import torch.nn as nn
    # Altı arazi sınıfını ayrı öğrenmek farklı engelleri tek sınıfa sıkıştırmaz.
    return nn.Sequential(nn.Conv2d(419,64,1),nn.ReLU(),
                         nn.Conv2d(64,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,num_classes,1))


class DenseSurfacePredictor:
    def __init__(self, config):
        import torch
        from context_features import ContextEncoder
        if config.grid_step != 4:
            raise ValueError('Yoğun modelin dondurulmuş örnekleme adımı 4 olmalı.')
        path = Path(config.model_path)
        bundle = torch.load(path,map_location='cpu',weights_only=True)
        if bundle.get('feature_version') != DENSE_VERSION or bundle.get('viewpoint') != 'aerial':
            raise ValueError('Yoğun model sürümü veya bakış açısı uyuşmuyor.')
        if config.expected_viewpoint not in ('any','aerial'):
            raise ValueError('Yoğun model bakış açısı bu iş akışıyla uyuşmuyor.')
        if not config.encoder_path:
            raise ValueError('Yoğun model için encoder_path gerekli.')
        self.encoder = ContextEncoder(config.encoder_path,bundle['encoder_sha256'])
        if bundle.get('num_classes',6) not in (2,6):
            raise ValueError('Yoğun model sınıf sayısı desteklenmiyor.')
        self.head = make_head(bundle.get('num_classes',6))
        self.head.load_state_dict(bundle['head'])
        self.head.eval()
        self.mean,self.std = bundle['mean'],bundle['std']
        if (self.mean.shape != (1,419,1,1) or self.std.shape != self.mean.shape
                or not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all()
                or (self.std <= 0).any()):
            raise ValueError('Yoğun model normalizasyonu geçersiz.')
        self.feature_version,self.viewpoint = DENSE_VERSION,'aerial'
        self.model_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.config,self.torch = config,torch

    def score_patch(self, image):
        torch = self.torch
        features = self.encoder.features(image,4)
        value = torch.from_numpy(features.transpose(2,0,1).copy()).unsqueeze(0)
        with torch.inference_mode():
            score = self.head((value-self.mean)/self.std).softmax(1)[0,0].numpy()
        return score.repeat(4,0).repeat(4,1)[:image.shape[0],:image.shape[1]]

    def predict(self, image):
        from surface_classification import SurfaceResult
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Yoğun model girdisi uint8 BGR olmalı.')
        height,width = image.shape[:2]
        if min(height,width) < 32:
            raise ValueError('Yoğun model görüntüsü en az 32×32 olmalı.')
        total,weight = np.zeros((height,width),'float32'),np.zeros((height,width),'float32')
        # Eğitim ölçeğini koruyan, global 4 piksel ızgarasına hizalı örtüşen pencereler.
        def starts(length):
            if length <= 512:
                return [0]
            last = ((length-512+3)//4)*4
            return sorted(set([*range(0,last,448),last]))
        for y in starts(height):
            for x in starts(width):
                patch = image[y:min(y+512,height),x:min(x+512,width)]
                h,w = patch.shape[:2]
                if min(h,w) < 32:
                    import cv2
                    patch = cv2.copyMakeBorder(patch,0,max(0,32-h),0,max(0,32-w),cv2.BORDER_REFLECT_101)
                score = self.score_patch(patch)[:h,:w]
                window = np.maximum(np.outer(np.hanning(h),np.hanning(w)),.01).astype('float32')
                total[y:y+h,x:x+w] += score*window
                weight[y:y+h,x:x+w] += window
        probability = total/weight
        if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
            raise ValueError('Yoğun model geçersiz yüzey puanı üretti.')
        return SurfaceResult(probability,probability < self.config.threshold,self.model_sha256,
                             self.encoder.sha256,self.feature_version,self.viewpoint)
