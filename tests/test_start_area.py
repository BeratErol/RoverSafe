"""Metre ölçeği, engel açıklığı ve görev kısıtlarının davranış testleri."""

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from elevation_analysis import analyze_elevation
from geospatial_io import load_manifest
from mission_constraints import MissionConfig
from rover_profile import RoverProfile
from start_site_selection import select_start_site
from start_area_cli import run


class StartAreaTests(unittest.TestCase):
    def setUp(self):
        self.z = np.zeros((120, 160), dtype=np.float64)
        self.ground = np.ones(self.z.shape, bool)
        self.zone = self.ground.copy()
        self.profile = RoverProfile()

    def select(self, mission=None, **kwargs):
        return select_start_site(analyze_elevation(self.z, .1, .1), self.profile,
                                 mission or MissionConfig(exit_pixel=(130, 60)), .1, .1,
                                 ground_mask=kwargs.get('ground', self.ground),
                                 allowed_mask=kwargs.get('allowed', self.ground), start_zone=self.zone)

    def test_plane_slope_and_residual(self):
        y, x = np.indices((50, 60))
        z = 10000 + x * .1 * np.tan(np.deg2rad(12)) + y * .2 * .1
        result = analyze_elevation(z, .1, .2)
        expected = np.rad2deg(np.arctan(np.hypot(np.tan(np.deg2rad(12)), .1)))
        np.testing.assert_allclose(result.slope_deg[result.valid], expected, atol=1e-5)
        self.assertLess(np.nanmax(result.roughness_m), 1e-5)

    def test_no_data_and_borders_blocked(self):
        self.z[60, 80] = np.nan
        terrain = analyze_elevation(self.z, .1, .1)
        self.assertFalse(terrain.valid[58:63, 78:83].any())
        self.assertFalse(terrain.valid[0].any())
        self.assertFalse(self.select().traversable[60, 80])

    def test_flat_roof_excluded_by_ground_mask(self):
        self.ground[:, 80:] = False
        result = self.select(MissionConfig(exit_pixel=(40, 60)))
        self.assertEqual(result.status, 'candidate_selected')
        self.assertLess(result.best['center_pixel'][0], 80)
        self.assertTrue(all(x < 80 for x, y in result.route_pixels))

    def test_fixed_start_never_relocated_and_wall_blocks_route(self):
        self.ground[:, 79:82] = False
        mission = MissionConfig(mode='assess_fixed_start', fixed_start_pixel=(35, 60), exit_pixel=(130, 60))
        self.assertEqual(self.select(mission).status, 'fixed_start_rejected')
        self.assertIsNone(self.select(mission).best)
        diagnostics = self.select(mission).point_checks['fixed_start']
        self.assertTrue(diagnostics['motion_clearance_ok'])
        self.assertFalse(diagnostics['reachable_start_candidate'])

    def test_start_zone_requires_entire_footprint(self):
        self.zone[:] = False
        self.zone[50:55, 50:55] = True
        self.assertEqual(self.select().status, 'no_reachable_candidate')

    def test_allowed_area_hard_constraint_and_missing_exit(self):
        allowed = self.ground.copy()
        allowed[:, 100:] = False
        self.assertEqual(self.select(allowed=allowed).status, 'exit_goal_not_traversable')
        self.assertIn('outside_allowed_area',self.select(allowed=allowed).point_checks['exit']['failed_center_constraints'])
        self.assertEqual(self.select(MissionConfig()).status, 'exit_goal_required')

    def test_steep_plane_rejected(self):
        self.z[:] = np.indices(self.z.shape)[1] * .1 * np.tan(np.deg2rad(30))
        self.assertEqual(self.select().status, 'exit_goal_not_traversable')

    def test_resolution_and_unknown_height_rejected(self):
        with self.assertRaises(ValueError):
            self.profile.footprint_pixels(.2, .2)
        self.z[:] = np.nan
        self.assertEqual(self.select().status, 'exit_goal_not_traversable')

    def test_route_and_metric_length(self):
        result = self.select(MissionConfig(mode='assess_fixed_start', fixed_start_pixel=(35, 60), exit_pixel=(130, 60)))
        self.assertEqual(result.status, 'fixed_start_accepted')
        self.assertEqual(result.best['center_pixel'], [35, 60])
        self.assertAlmostEqual(result.best['route_length_m'], 9.5)
        self.assertTrue(all(abs(a[0]-b[0])+abs(a[1]-b[1]) == 1 for a,b in zip(result.route_pixels,result.route_pixels[1:])))

    def test_geotiff_roundtrip_and_grid_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {'elevation_unit':'m', 'elevation_kind':'DSM', 'ground_mask_source':'synthetic',
                    'mission_source':'test fixture', 'dataset_source':'synthetic',
                    'mission': {'exit_pixel':[130,60]}}
            transform = from_origin(500000, 4500000, .1, .1)
            for key in ('rgb','elevation','ground_mask','allowed_mask','start_zone'):
                spec[key] = key + '.tif'
                array = np.full((3,120,160), 150, np.uint8) if key == 'rgb' else (
                    self.z[None].astype('float32') if key == 'elevation' else self.ground[None].astype('uint8'))
                with rasterio.open(root/spec[key], 'w', driver='GTiff', width=160,height=120,
                                   count=len(array),dtype=array.dtype,crs='EPSG:32636',transform=transform) as dst:
                    dst.write(array)
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps(spec), encoding='utf-8')
            result = run(manifest, root/'out')
            self.assertEqual(result['status'], 'candidate_selected')
            self.assertTrue((root/'out/selection.png').exists())
            inputs = load_manifest(manifest)
            self.assertEqual(inputs.world((0,0)), [500000.05,4499999.95])
            with rasterio.open(root/'allowed_mask.tif','r+') as dst:
                dst.transform = from_origin(500001,4500000,.1,.1)
            with self.assertRaises(ValueError):
                load_manifest(manifest)


if __name__ == '__main__':
    unittest.main()
