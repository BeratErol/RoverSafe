"""Ek geliştirme karolarını ve ayrı hava test karosunu değişmez protokole kaydeder."""

import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from potsdam_data import build_protocol


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,default=Path('data/datasets/potsdam'))
    parser.add_argument('--output',type=Path,default=Path('outputs/dense_expanded_protocol.json'))
    args=parser.parse_args()
    if args.output.exists(): parser.error('Yeni bir protokol dosyası seçin; mevcut kayıt değiştirilmez.')
    build_protocol(args.source_dir,args.output,splits={'development':['2_11','3_11','4_10','5_10'],
        'validation':['4_11'],'test':['6_11']})
    print(args.output)
