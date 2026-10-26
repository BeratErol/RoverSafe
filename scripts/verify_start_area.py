"""Sentetik görevlerde tam dosya akışını ve rota koşullarını kontrol eder."""

import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.create_start_area_sample import SCENARIOS, prepare
from start_area_cli import run


def verify(output_dir, *, profile=None, resolution_m=.1):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Doğrulama için yeni veya boş bir çıktı klasörü seçin.')
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario, expected in SCENARIOS.items():
        tick = perf_counter()
        row = {'scenario': scenario, 'expected_status': expected, 'status': 'analysis_failed',
               'passed': False, 'route_length_m': None, 'error': None}
        try:
            manifest = prepare(output/scenario/'inputs', scenario,profile=profile,resolution_m=resolution_m)
            folder = output/scenario/'result'
            report = run(manifest, folder)
            row['status'] = report['status']
            row['passed'] = report['status'] == expected
            if report['best']:
                route = report['route_pixels']
                best = report['best']
                row['route_length_m'] = best['route_length_m']
                steps_valid = all(abs(a[0]-b[0])+abs(a[1]-b[1]) == 1 for a,b in zip(route,route[1:]))
                endpoints_valid = (list(route[0]) == best['center_pixel']
                                   and list(route[-1]) == list(report['mission']['exit_pixel']))
                clearance_valid = best['route_min_clearance_m'] >= best['required_motion_radius_m']
                with np.load(folder/'maps.npz') as maps:
                    x,y,w,h = best['footprint_with_margin_px']
                    footprint_valid = bool(maps['traversable'][y:y+h,x:x+w].all())
                row['passed'] &= steps_valid and endpoints_valid and clearance_valid and footprint_valid
            else:
                row['passed'] &= not report['route_pixels']
        except Exception as error:
            row['passed'] = False
            row['error'] = str(error)
        row['seconds'] = perf_counter()-tick
        rows.append(row)
    summary = {'kind': 'synthetic_software_acceptance', 'passed': all(row['passed'] for row in rows),
               'case_count': len(rows), 'passed_count': sum(row['passed'] for row in rows),
               'interpretation': 'Known synthetic scenarios verify software behavior, not real-world perception or rover safety.',
               'cases': rows}
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    with (output/'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Başlangıç alanı yazılım doğrulaması', '', summary['interpretation'], '',
             '| Senaryo | Beklenen | Gerçekleşen | Kontrol |', '| --- | --- | --- | --- |']
    for row in rows:
        lines.append(f"| {row['scenario']} | {row['expected_status']} | {row['status']} | {'PASS' if row['passed'] else 'FAIL'} |")
    (output/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/acceptance'))
    args = parser.parse_args()
    result = verify(args.output_dir)
    print(f"{result['passed_count']}/{result['case_count']} passed: {args.output_dir/'summary.json'}")
    raise SystemExit(0 if result['passed'] else 1)
