"""Aynı kayıtlı karelerde video hızını ölçer; doğruluk veya saha emniyeti testi değildir."""

import argparse
from dataclasses import asdict
from html import escape
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from configuration import load_config
from potsdam_data import sha256
from video import process_video


def prepare_clip(source, destination, width, frames):
    """Ölçüm girdisini bir kez üretir; kaynak zamanlaması korunur, ölçek kaydedilir."""
    capture = cv2.VideoCapture(str(source.resolve()))
    writer = None
    count = 0
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError('Kaynak video veya kare hızı okunamadı.')
        while count < frames:
            ok, image = capture.read()
            if not ok:
                break
            if writer is None:
                original = [image.shape[1], image.shape[0]]
                size = (min(width, original[0]), max(1, round(original[1] * min(width, original[0]) / original[0])))
                writer = cv2.VideoWriter(str(destination.resolve()), cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
                if not writer.isOpened():
                    raise OSError('Ölçüm videosu yazılamadı.')
            writer.write(cv2.resize(image, size, interpolation=cv2.INTER_AREA))
            count += 1
        if count < 2:
            raise ValueError('Ölçüm için en az iki kare gerekir.')
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return {'original_size': original, 'benchmark_size': list(size), 'frames': count, 'source_fps': fps,
            'transform': 'first frames; aspect-preserving resize; MJPG re-encode; no accuracy evaluation'}


def benchmark(source, output, *, surface_config=None, width=640, frames=24, repeats=3, strides=(1, 3)):
    if any(type(value) is not int or value < 1 for value in (width, frames, repeats, *strides)) or not strides:
        raise ValueError('Boyut, kare, tekrar ve adımlar pozitif tam sayı olmalı.')
    output = Path(output)
    source = Path(source)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Yeni veya boş çıktı klasörü gerekli.')
    if not source.is_file():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)
    clip = output / 'input.avi'
    inputs = {'source': str(source.resolve()), 'source_sha256': sha256(source),
              **prepare_clip(source, clip, width, frames), 'clip_sha256': sha256(clip)}
    profiles = {'classic': {}}
    if surface_config:
        profiles['aerial_model'] = load_config(surface_config)
        if profiles['aerial_model']['surface_config'] is None:
            raise ValueError('Yüzey modeli içeren bir yapılandırma gerekli.')
    rows = []
    # Her tekrar modelini yeniden yükler; ilk kare ile sonraki kareler ayrı raporlanır.
    for name, options in profiles.items():
        for stride in strides:
            for repeat in range(repeats):
                case = output / f'{name}_stride{stride}_run{repeat + 1}'
                tick = perf_counter()
                summary = process_video(clip, case, options=options, stride=stride)
                total = perf_counter() - tick
                logs = [json.loads(line) for line in (case / 'frames.jsonl').read_text().splitlines()]
                indices = list(range(0, inputs['frames'], stride))
                if [row['frame_index'] for row in logs] != indices:
                    raise RuntimeError('Seçilen kareler beklenen adımla eşleşmiyor.')
                capture = cv2.VideoCapture(str((case / 'annotated.avi').resolve()))
                try:
                    decoded = 0
                    while capture.read()[0]:
                        decoded += 1
                finally:
                    capture.release()
                if decoded != len(indices):
                    raise RuntimeError('Çıktı videosu tam olarak okunamadı.')
                settings = json.loads((case / 'settings.json').read_text())
                times = [row['processing_seconds'] for row in logs]
                rows.append({'profile': name, 'stride': stride, 'repeat': repeat + 1,
                             'processed_frames': len(logs), 'selected_frames': summary['selected_frames'],
                             'first_frame_seconds': times[0], 'median_frame_seconds': float(np.median(times)),
                             'p95_frame_seconds': float(np.percentile(times, 95)),
                             'processing_fps': summary['processing_fps'], 'total_seconds': total,
                             'end_to_end_fps': len(logs) / total,
                             'model_setup_seconds': max(0., total - summary['wall_seconds']),
                             'surface_model_sha256': settings.get('surface_model_sha256'),
                             'surface_encoder_sha256': settings.get('surface_encoder_sha256'),
                             'surface_threshold': (settings.get('surface') or {}).get('threshold'),
                             'settings_sha256': sha256(case / 'settings.json'),
                             'artifacts': case.name, 'output_decode_passed': True})
                print(f'{name} stride={stride} run={repeat + 1}: {summary["processing_fps"]:.2f} processing FPS', flush=True)
    packages = {}
    for package in ('numpy', 'opencv-python', 'torch', 'torchvision'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    result = {'completed': True, 'input': inputs, 'repeats': repeats,
              'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                              'processor': platform.processor(), 'opencv_threads': cv2.getNumThreads(), 'packages': packages},
              'profiles': {name: {key: asdict(value) if hasattr(value, '__dataclass_fields__') else value
                                  for key, value in options.items()} for name, options in profiles.items()},
              'results': rows,
              'interpretation': 'Recorded RGB screening only; no tracking, metric motion inference or accuracy evaluation. '
                                'Stride lowers sampling frequency, not inference cost per frame. '
                                'Total time includes model setup, decode, render and encode; setup estimate includes pre-loop initialization.'}
    (output / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    table = ''.join('<tr>' + ''.join(f'<td>{escape(str(value))}</td>' for value in
                    (row['profile'], row['stride'], row['repeat'], row['processed_frames'],
                     f"{row['median_frame_seconds'] * 1000:.1f}", f"{row['p95_frame_seconds'] * 1000:.1f}",
                     f"{row['processing_fps']:.2f}", f"{row['end_to_end_fps']:.2f}")) +
                    f'<td><a href="{row["artifacts"]}/first_frame.png">Görsel</a> · '
                    f'<a href="{row["artifacts"]}/annotated.avi">Video</a></td></tr>' for row in rows)
    page = '<!doctype html><html lang="tr"><meta charset="utf-8"><title>RoverSafe video ölçümü</title>'
    page += '<style>body{font:16px system-ui;max-width:1200px;margin:35px auto;padding:15px}td,th{padding:9px;border-bottom:1px solid #ccc;text-align:left}table{border-collapse:collapse}pre{white-space:pre-wrap}</style>'
    page += '<h1>Video ve performans ölçümü</h1><p>Aynı karelerde RGB ön eleme. Algılama doğruluğu veya gerçek zaman garantisi ölçülmez.</p>'
    page += f'<p>Girdi: {inputs["benchmark_size"]}, {inputs["frames"]} kare, {inputs["source_fps"]:g} FPS. Her profil/adım için {repeats} tekrar.</p>'
    page += '<table><tr><th>Profil</th><th>Adım</th><th>Tekrar</th><th>Kare</th><th>Medyan ms</th><th>P95 ms</th><th>İşleme FPS</th><th>Uçtan uca FPS</th><th>Çıktı</th></tr>' + table + '</table>'
    page += '<p>İşleme FPS model yüklemesini ve video I/O işlemlerini içermez. Uçtan uca FPS tümünü içerir. Kare atlamak tek kare çıkarımını hızlandırmaz. Kaynak klibin zamanlaması yapay olabilir; kaynak manifesti ayrıca incelenmelidir.</p><p><a href="summary.json">Tüm ölçümler ve ortam kaydı</a></p></html>'
    (output / 'report.html').write_text(page, encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/video_benchmark'))
    parser.add_argument('--surface-config', type=Path)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--frames', type=int, default=24)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--strides', type=int, nargs='+', default=[1, 3])
    args = parser.parse_args()
    benchmark(args.input, args.output_dir, surface_config=args.surface_config, width=args.width,
              frames=args.frames, repeats=args.repeats, strides=tuple(args.strides))
    print(args.output_dir / 'report.html')
