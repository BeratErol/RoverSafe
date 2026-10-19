"""Resmî Potsdam.zip içinden yalnız 2_10 RGB/DSM/etiket örneğini indirir.

--archive-url: ISPRS indirme sayfasındaki Potsdam bağlantısının geçici gerçek
indirme URL'si. Parola veya oturum URL'si kaynak manifestine yazılmaz.
Windows tar (libarchive) RAR açmak için gereklidir; tüm 13 GB indirilmez.
"""

import argparse
from collections import OrderedDict
import io
from pathlib import Path
import shutil
import ssl
import struct
import subprocess
import urllib.request
import zipfile
import rarfile
import truststore


class RemoteArchive(io.RawIOBase):
    """Sınırlı önbellek ve HTTP Range ile seek edilebilir arşiv erişimi."""
    def __init__(self, url):
        self.url, self.position, self.cache = url, 0, OrderedDict()
        self.context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        with self.fetch(0, 0) as response:
            self.length = int(response.headers['Content-Range'].split('/')[-1])

    def fetch(self, start, end):
        request = urllib.request.Request(self.url, headers={'Range':f'bytes={start}-{end}'})
        response = urllib.request.urlopen(request, context=self.context, timeout=90)
        expected = f'bytes {start}-{end}/'
        if response.status != 206 or not response.headers.get('Content-Range','').startswith(expected):
            response.close()
            raise ValueError('Sunucu istenen HTTP Range aralığını sağlamadı; tam arşiv indirilmedi.')
        return response

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position+offset if whence == 1 else self.length+offset
        if self.position < 0:
            raise ValueError('Negatif arşiv konumu.')
        return self.position

    def read(self, size=-1):
        size = min(size if size >= 0 else self.length-self.position, self.length-self.position)
        chunks = []
        while size > 0:
            block, offset = divmod(self.position, 2**20)
            if block not in self.cache:
                start = block*2**20
                end = min(start+2**20-1, self.length-1)
                with self.fetch(start, end) as response:
                    value = response.read()
                if len(value) != end-start+1:
                    raise OSError('Eksik arşiv bloğu; yeniden çalıştırın.')
                self.cache[block] = value
                if len(self.cache) > 8:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(block)
            value = self.cache[block][offset:offset+size]
            chunks.append(value)
            self.position += len(value)
            size -= len(value)
        return b''.join(chunks)


class ArchiveSlice(io.RawIOBase):
    def __init__(self, remote, offset, length):
        self.remote, self.offset, self.length, self.position = remote, offset, length, 0

    tell = RemoteArchive.tell
    seek = RemoteArchive.seek
    readable = RemoteArchive.readable
    seekable = RemoteArchive.seekable

    def read(self, size=-1):
        size = min(size if size >= 0 else self.length-self.position, self.length-self.position)
        self.remote.seek(self.offset+self.position)
        value = self.remote.read(max(0, size))
        self.position += len(value)
        return value


def write_stream(source, target):
    partial = target.with_suffix(target.suffix+'.part')
    with partial.open('wb') as output:
        shutil.copyfileobj(source, output)
    partial.replace(target)


def download(url, root):
    root.mkdir(parents=True, exist_ok=True)
    outputs = [root/'top_potsdam_2_10_RGB.tif', root/'1_DSM/dsm_potsdam_02_10.tif',
               root/'top_potsdam_2_10_label.tif']
    if all(path.exists() for path in outputs):
        print('Örnek zaten mevcut; hazırlayıcı kaynak özetlerini hesaplayabilir.')
        return
    tar = shutil.which('tar')
    if tar is None:
        raise OSError('RAR desteği olan tar/libarchive gerekli (Windows tar.exe desteklenir).')
    remote = RemoteArchive(url)
    with zipfile.ZipFile(remote) as outer:
        def stored_slice(name):
            entry = outer.getinfo(name)
            if entry.compress_type != zipfile.ZIP_STORED:
                raise ValueError('Kaynak arşiv yapısı değişmiş; alt arşiv seek edilemiyor.')
            remote.seek(entry.header_offset)
            header = struct.unpack('<4s5H3L2H', remote.read(30))
            return ArchiveSlice(remote, entry.header_offset+30+header[-2]+header[-1],entry.file_size)

        if not outputs[1].exists():
            part = stored_slice('Potsdam/1_DSM.rar')
            with rarfile.RarFile(part) as archive:
                if archive.is_solid():
                    raise ValueError('Solid RAR tek karo indirmeye uygun değil.')
                first = archive.infolist()[:2]
                expected = ['1_DSM/dsm_potsdam_02_10.tfw','1_DSM/dsm_potsdam_02_10.tif']
                if [entry.filename for entry in first] != expected:
                    raise ValueError('DSM arşiv sırası değişmiş; güvenli biçimde duruldu.')
                end = first[-1].data_offset + first[-1].compress_size
            subset = root/'sample_dsm.rar'
            with remote.fetch(part.offset,part.offset+end-1) as response:
                write_stream(response,subset)
            listing = subprocess.run([tar,'-tf',str(subset)],check=True,capture_output=True,text=True)
            if listing.stdout.splitlines() != expected:
                raise ValueError('Beklenmeyen DSM arşiv içeriği.')
            subprocess.run([tar,'-xf',str(subset),'-C',str(root)],check=True)
            print('DSM hazır.',flush=True)
        if not outputs[0].exists():
            with zipfile.ZipFile(stored_slice('Potsdam/2_Ortho_RGB.zip')) as archive:
                with archive.open('2_Ortho_RGB/top_potsdam_2_10_RGB.tif') as source:
                    write_stream(source,outputs[0])
            print('RGB hazır.',flush=True)
        if not outputs[2].exists():
            labels = root/'labels.zip'
            if not labels.exists():
                with outer.open('Potsdam/5_Labels_all.zip') as source:
                    write_stream(source,labels)
            with zipfile.ZipFile(labels) as archive:
                with archive.open('top_potsdam_2_10_label.tif') as source:
                    write_stream(source,outputs[2])
            print('Etiketler hazır.',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-url', required=True, help='Resmî paylaşımın güncel doğrudan Potsdam.zip URL’si')
    parser.add_argument('--output-dir',type=Path,default=Path('data/datasets/potsdam'))
    args = parser.parse_args()
    download(args.archive_url,args.output_dir)
