"""Dağıtılan örneği, coğrafi çıktıları ve geçersiz rota uçlarını kontrol eder."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.warp import transform
from scripts.create_start_area_sample import prepare
from scripts.verify_start_area import verify
from start_area_cli import run
from start_site_selection import find_route


class StartDeliveryTests(unittest.TestCase):
    def test_synthetic_acceptance_suite(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = verify(Path(directory)/'suite')
            self.assertTrue(summary['passed'], summary['cases'])
            self.assertEqual(summary['case_count'], 12)

    def test_geographic_export_alignment_and_input_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = prepare(root/'inputs', 'fixed_valid')
            report = run(manifest, root/'out')
            geo = json.loads((root/'out/selection.geojson').read_text(encoding='utf-8'))
            features = {item['properties']['kind']: item for item in geo['features']}
            center = features['selected_start']['geometry']['coordinates']
            expected = transform('EPSG:32633', 'EPSG:4326', [500003.55], [4499993.95])
            np.testing.assert_allclose(center, [expected[0][0], expected[1][0]])
            self.assertEqual(features['route']['geometry']['coordinates'][0], center)
            polygon = features['footprint_with_margin']['geometry']['coordinates'][0]
            self.assertEqual(polygon[0], polygon[-1])
            signed_area = sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(polygon,polygon[1:]))
            self.assertGreater(signed_area, 0)  # GeoJSON dış halkası saat yönünün tersidir.
            with rasterio.open(root/'out/decision.tif') as dst:
                self.assertEqual(dst.crs.to_epsg(), 32633)
                classes = dst.read(1)
                self.assertEqual(classes[60,35], 3)
                self.assertEqual(classes[0,0], 0)
                self.assertIn('reachable_start_center', dst.tags()['classes'])
            with rasterio.open(root/'out/traversability.tif') as dst:
                self.assertEqual(dst.count,2)
                self.assertEqual(dst.descriptions,('terrain_severity','cost_per_meter'))
                self.assertEqual(dst.nodata,-9999)
                self.assertTrue(np.isfinite(dst.read(2,masked=True).compressed()).all())
            records = {item['path']: item for item in report['input_fingerprints']}
            self.assertEqual(len(records), 6)
            self.assertEqual(records['manifest.json']['sha256'], hashlib.sha256(manifest.read_bytes()).hexdigest())

    def test_rejection_exports_no_selected_start_or_route(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = prepare(root/'inputs', 'wall')
            run(manifest, root/'out')
            geo = json.loads((root/'out/selection.geojson').read_text(encoding='utf-8'))
            self.assertEqual([f['properties']['kind'] for f in geo['features']], ['exit'])

    def test_failed_rerun_removes_geographic_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = prepare(root/'inputs')
            run(manifest, root/'out')
            spec = json.loads(manifest.read_text(encoding='utf-8'))
            spec['mission']['exit_pixel'] = [9999, 1]
            manifest.write_text(json.dumps(spec), encoding='utf-8')
            with self.assertRaises(ValueError):
                run(manifest, root/'out')
            self.assertFalse((root/'out/selection.geojson').exists())
            self.assertFalse((root/'out/decision.tif').exists())

    def test_exit_diagnostics_show_nearby_obstacle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = prepare(root/'inputs', 'flat')
            with rasterio.open(root/'inputs/ground_mask.tif', 'r+') as dst:
                ground = dst.read(1)
                ground[60, 131] = 0
                dst.write(ground, 1)
            report = run(manifest, root/'out')
            self.assertEqual(report['status'], 'exit_goal_not_traversable')
            checks = report['point_checks']['exit']
            self.assertEqual(checks['failed_center_constraints'], [])
            self.assertEqual(checks['failed_footprint_constraints']['not_ground'], 1)
            self.assertEqual(checks['failed_motion_envelope_constraints']['not_ground'], 1)

    def test_sample_never_overwrites_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root/'keep.txt'
            marker.write_text('keep', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                prepare(root)
            self.assertEqual(marker.read_text(encoding='utf-8'), 'keep')

    def test_route_rejects_blocked_endpoints_and_invalid_coordinates(self):
        mask = np.ones((15,20), bool)
        mask[3,3] = False
        self.assertEqual(find_route(mask, (3,3), (3,3), .1, .2), [])
        self.assertEqual(find_route(mask, (3,3), (10,10), .1, .2), [])
        self.assertEqual(find_route(mask, (10,10), (3,3), .1, .2), [])
        for start in ((-1,0), (20,0), (1.5,2)):
            with self.assertRaises(ValueError):
                find_route(mask, start, (10,10), .1, .2)

    def test_anisotropic_route_uses_shorter_metric_path(self):
        mask = np.ones((8,9), bool)
        mask[1:7,4] = False
        route = find_route(mask, (1,4), (7,4), .2, .1)
        length = sum(abs(b[0]-a[0])*.2+abs(b[1]-a[1])*.1 for a,b in zip(route,route[1:]))
        self.assertAlmostEqual(length, 1.8)


if __name__ == '__main__':
    unittest.main()
