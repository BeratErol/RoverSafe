"""Potsdam arşivinden belirli karoların RGB/etiket ve isteğe bağlı DSM verisini indirir."""

import argparse
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rarfile
from scripts.download_potsdam_sample import RemoteArchive, ArchiveSlice, write_stream
from potsdam_data import tile_names


def stored_member(remote, outer, name):
    entry = outer.getinfo(name)
    if entry.compress_type != zipfile.ZIP_STORED:
        raise ValueError('İç arşiv ZIP_STORED değil; kaynak yapısı değişmiş.')
    remote.seek(entry.header_offset)
    header = struct.unpack('<4s5H3L2H',remote.read(30))
    return ArchiveSlice(remote,entry.header_offset+30+header[-2]+header[-1],entry.file_size)


def download_tiles(url, root, tiles, dsm_tiles=()):
    tiles = list(dict.fromkeys(tiles))
    for tile in tiles:
        tile_names(tile)
    if not set(dsm_tiles) <= set(tiles):
        raise ValueError('DSM karoları RGB listesinde de bulunmalı.')
    root.mkdir(parents=True,exist_ok=True)
    remote = RemoteArchive(url)
    with zipfile.ZipFile(remote) as outer:
        with zipfile.ZipFile(stored_member(remote,outer,'Potsdam/2_Ortho_RGB.zip')) as archive:
            for tile in tiles:
                rgb,_,_ = tile_names(tile)
                target = root/rgb
                if not target.exists():
                    with archive.open('2_Ortho_RGB/'+rgb) as source:
                        write_stream(source,target)
                print(tile,'RGB hazır',flush=True)
        labels = root/'labels.zip'
        if not labels.exists():
            with outer.open('Potsdam/5_Labels_all.zip') as source:
                write_stream(source,labels)
        with zipfile.ZipFile(labels) as archive:
            for tile in tiles:
                _,label,_ = tile_names(tile)
                if not (root/label).exists():
                    with archive.open(label) as source:
                        write_stream(source,root/label)
        missing_dsm = [tile for tile in dsm_tiles if not (root/'1_DSM'/(tile_names(tile)[2]+'.tif')).exists()]
        if missing_dsm:
            tar = shutil.which('tar')
            if not tar:
                raise OSError('RAR destekli tar/libarchive gerekli.')
            part = stored_member(remote,outer,'Potsdam/1_DSM.rar')
            with rarfile.RarFile(part) as archive:
                if archive.is_solid():
                    raise ValueError('Solid RAR bağımsız DSM çıkarmaya uygun değil.')
                main_size = min(item.header_offset for item in archive.infolist())
                part.seek(0)
                main_header = part.read(main_size)
                if not main_header.startswith(b'Rar!\x1a\x07\x00'):
                    raise ValueError('Beklenen RAR4 biçimi değil.')
                for tile in missing_dsm:
                    stem = tile_names(tile)[2]
                    expected = [f'1_DSM/{stem}.tfw',f'1_DSM/{stem}.tif']
                    entries = [archive.getinfo(name) for name in expected]
                    subset = root/f'sample_dsm_{tile}.rar'
                    partial = subset.with_suffix('.rar.part')
                    with partial.open('wb') as output:
                        output.write(main_header)
                        for entry in entries:
                            start = part.offset+entry.header_offset
                            end = part.offset+entry.data_offset+entry.compress_size-1
                            with remote.fetch(start,end) as response:
                                shutil.copyfileobj(response,output)
                    partial.replace(subset)
                    listing = subprocess.run([tar,'-tf',str(subset)],check=True,capture_output=True,text=True)
                    if listing.stdout.splitlines() != expected:
                        raise ValueError('Beklenmeyen DSM arşiv içeriği.')
                    subprocess.run([tar,'-xf',str(subset),'-C',str(root)],check=True)
                    print(tile,'DSM hazır',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-url',required=True)
    parser.add_argument('--output-dir',type=Path,default=Path('data/datasets/potsdam'))
    parser.add_argument('--tiles',nargs='+',required=True)
    parser.add_argument('--dsm-tiles',nargs='*',default=[])
    args = parser.parse_args()
    download_tiles(args.archive_url,args.output_dir,args.tiles,args.dsm_tiles)
