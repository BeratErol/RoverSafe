"""PyTorch'un resmî ResNet-18 ImageNet ağırlığını indirip dosya özetini doğrular."""

import argparse
import hashlib
from pathlib import Path
import shutil
import ssl
import urllib.request
import truststore

URL = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'


def download(path):
    path = Path(path)
    if path.exists():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if not digest.startswith('f37072fd'):
            raise ValueError('Mevcut kodlayıcı ağırlığı resmî özetle uyuşmuyor.')
        return digest
    path.parent.mkdir(parents=True,exist_ok=True)
    partial = path.with_suffix('.pth.part')
    with urllib.request.urlopen(URL,context=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),timeout=60) as response,partial.open('wb') as stream:
        shutil.copyfileobj(response,stream)
    digest = hashlib.sha256(partial.read_bytes()).hexdigest()
    if not digest.startswith('f37072fd'):
        raise ValueError('İndirilen kodlayıcı ağırlığının özeti uyuşmuyor.')
    partial.replace(path)
    return digest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('models/resnet18-f37072fd.pth'))
    args = parser.parse_args()
    print(download(args.output))
