"""Dondurulmuş ResNet-18 bağlamını yerel renk/doku özellikleriyle birleştirir."""

from pathlib import Path
import numpy as np
from potsdam_data import sha256

CONTEXT_VERSION = 'resnet18_context_v3'


class ContextEncoder:
    def __init__(self, weights_path, expected_sha256=None):
        import torch
        from torchvision.models import resnet18
        path = Path(weights_path)
        if not path.is_file():
            raise FileNotFoundError('Yerel ResNet-18 ağırlığı gerekli; scripts/download_context_encoder.py çalıştırın.')
        self.sha256 = sha256(path)
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ValueError('Bağlam kodlayıcı dosyasının özeti model kaydıyla uyuşmuyor.')
        torch.set_num_threads(2)
        self.torch = torch
        self.network = resnet18(weights=None)
        self.network.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        self.network.eval()
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)

    def features(self,image,step=4):
        from surface_classification import feature_grid,MULTISCALE_VERSION
        import cv2
        torch = self.torch
        local = feature_grid(image,step,MULTISCALE_VERSION)
        scale = min(1.,512/max(image.shape[:2]))
        resized = cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA) if scale < 1 else image
        rgb = resized[:,:,::-1].copy().astype('float32')/255
        tensor = torch.from_numpy(rgb.transpose(2,0,1)).unsqueeze(0)
        mean = torch.tensor([.485,.456,.406]).view(1,3,1,1)
        std = torch.tensor([.229,.224,.225]).view(1,3,1,1)
        net = self.network
        with torch.inference_mode():
            value = net.maxpool(net.relu(net.bn1(net.conv1((tensor-mean)/std))))
            value = net.layer1(value)
            middle = net.layer2(value)
            deep = net.layer3(middle)
            combined = [local]
            for feature in (middle,deep):
                expanded = torch.nn.functional.interpolate(feature,size=local.shape[:2],mode='bilinear',align_corners=False)
                combined.append(expanded[0].permute(1,2,0).numpy())
        return np.ascontiguousarray(np.concatenate(combined,axis=2),dtype='float32')
