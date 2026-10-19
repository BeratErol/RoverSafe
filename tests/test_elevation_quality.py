"""Sınırlı yükseklik hatasının kararlara ve veri sözleşmesine etkisini sınar."""

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from elevation_analysis import analyze_elevation
from elevation_quality import ElevationQualityConfig
from geospatial_io import load_manifest
from mission_constraints import MissionConfig
from rover_profile import RoverProfile
from start_site_selection import select_start_site
from start_area_cli import run


class ElevationQualityTests(unittest.TestCase):
    def select(self, terrain):
        mask = np.ones(terrain.valid.shape,bool)
        return select_start_site(terrain,RoverProfile(),MissionConfig(exit_pixel=(75,45)),.1,.1,
                                 ground_mask=mask,allowed_mask=mask,start_zone=mask)

    def test_invalid_quality_configuration(self):
        for kwargs in ({'mode':'confidence'},{'vertical_error_bound_m':.01},
                       {'validity_mask':'mask.tif'},
                       {'mode':'bounded_error'},
                       {'mode':'bounded_error','vertical_error_bound_m':True,'source':'fixture'},
                       {'mode':'bounded_error','vertical_error_bound_m':float('nan'),'source':'fixture'},
                       {'mode':'bounded_error','vertical_error_bound_m':-.1,'source':'fixture'},
                       {'mode':'bounded_error','vertical_error_bound_m':.1,'error_bound_raster':'b.tif','source':'fixture'}):
            with self.assertRaises(ValueError):
                ElevationQualityConfig(**kwargs)
        self.assertEqual(ElevationQualityConfig(mode='bounded_error',vertical_error_bound_m=0,source='exact synthetic').mode,'bounded_error')

    def test_zero_bound_preserves_nominal_maps_and_selection(self):
        y,x = np.indices((90,110))
        z = .002*x + .001*y
        nominal = analyze_elevation(z,.1,.1)
        bounded = analyze_elevation(z,.1,.1,error_bound_m=0)
        for observed,upper in zip(nominal.decision_maps(),bounded.decision_maps()):
            np.testing.assert_allclose(observed,upper,atol=1e-6,equal_nan=True)
        self.assertEqual(self.select(nominal).best['center_pixel'],self.select(bounded).best['center_pixel'])

    def test_upper_bounds_cover_bounded_perturbations(self):
        rng = np.random.default_rng(42)
        y,x = np.indices((25,31))
        z = 500 + .003*x + .002*np.sin(y)
        error = rng.uniform(.001,.004,z.shape)
        bounded = analyze_elevation(z,.1,.2,error_bound_m=error)
        for _ in range(24):
            realized = analyze_elevation(z+rng.uniform(-1,1,z.shape)*error,.1,.2)
            for actual,upper in zip(realized.decision_maps(),bounded.decision_maps()):
                self.assertTrue(np.all(actual[bounded.valid] <= upper[bounded.valid]+1e-5))

    def test_increasing_bound_never_adds_traversable_pixels(self):
        z = np.zeros((90,110))
        previous = None
        for error in (0,.005,.01,.02,.03):
            result = self.select(analyze_elevation(z,.1,.1,error_bound_m=error))
            if previous is not None:
                self.assertFalse((result.traversable & ~previous).any())
            previous = result.traversable
        self.assertEqual(result.status,'exit_goal_not_traversable')
        self.assertIn('slope_limit',result.point_checks['exit']['failed_center_constraints'])

    def test_unknown_error_and_quality_mask_exclude_neighborhoods(self):
        z = np.zeros((90,110))
        error = np.full(z.shape,.001)
        error[45,75] = np.nan
        terrain = analyze_elevation(z,.1,.1,error_bound_m=error)
        self.assertFalse(terrain.valid[43:48,73:78].any())
        result = self.select(terrain)
        self.assertIn('unknown_elevation_quality',result.point_checks['exit']['failed_center_constraints'])
        quality = np.ones(z.shape,bool); quality[45,75] = False
        masked = analyze_elevation(z,.1,.1,quality_mask=quality)
        self.assertFalse(masked.valid[43:48,73:78].any())
        self.assertFalse(masked.quality_invalid[0,0])

    def test_invalid_uncertainty_inputs(self):
        z = np.zeros((20,30))
        for value in (True,-.1,float('inf'),np.zeros((10,10)),np.full(z.shape,-.1)):
            with self.assertRaises(ValueError):
                analyze_elevation(z,.1,.1,error_bound_m=value)
        with self.assertRaises(ValueError):
            analyze_elevation(z,.1,.1,quality_mask=np.ones(z.shape,np.uint8))

    def fixture(self, root):
        spec = {'elevation_unit':'m','elevation_kind':'DSM','ground_mask_source':'synthetic',
                'mission_source':'fixture','dataset_source':'synthetic','mission':{'exit_pixel':[75,45]}}
        for key in ('rgb','elevation','ground_mask','allowed_mask','start_zone','quality_mask','error_bound'):
            name = key+'.tif'
            if key not in ('quality_mask','error_bound'):
                spec[key] = name
            array = np.full((3,90,110),150,np.uint8) if key == 'rgb' else (
                np.zeros((1,90,110),np.float32) if key in ('elevation','error_bound') else np.ones((1,90,110),np.uint8))
            with rasterio.open(root/name,'w',driver='GTiff',width=110,height=90,count=len(array),
                               dtype=array.dtype,crs='EPSG:32633',transform=from_origin(500000,4500000,.1,.1)) as dst:
                dst.write(array)
        path = root/'manifest.json'
        path.write_text(json.dumps(spec),encoding='utf-8')
        return spec,path

    def test_geospatial_quality_rasters_and_output_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec,path = self.fixture(root)
            spec['elevation_quality'] = {'mode':'bounded_error','error_bound_raster':'error_bound.tif',
                                         'validity_mask':'quality_mask.tif','source':'synthetic exact bounds'}
            path.write_text(json.dumps(spec),encoding='utf-8')
            result = run(path,root/'out')
            self.assertEqual(result['status'],'candidate_selected')
            self.assertEqual(result['elevation_quality']['decision_values'],'upper_bounds')
            self.assertEqual(result['best']['terrain_value_kind'],'upper_bounds')
            self.assertEqual(result['point_checks']['exit']['error_bound_m'],0.0)
            self.assertEqual(result['point_checks']['exit']['center_observed'],result['point_checks']['exit']['center_decision'])
            with np.load(root/'out/elevation_quality.npz') as maps:
                self.assertIn('error_bound_m',maps.files)
                self.assertEqual(maps['quality_invalid'].dtype,bool)
            with rasterio.open(root/'error_bound.tif','r+') as dst:
                dst.transform = from_origin(500001,4500000,.1,.1)
            with self.assertRaises(ValueError):
                load_manifest(path)

    def test_error_raster_nodata_and_negative_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec,path = self.fixture(root)
            spec['elevation_quality'] = {'mode':'bounded_error','error_bound_raster':'error_bound.tif','source':'fixture'}
            path.write_text(json.dumps(spec),encoding='utf-8')
            with rasterio.open(root/'error_bound.tif','r+') as dst:
                array = dst.read(); array[0,45,75] = -9999; dst.write(array); dst.nodata = -9999
            self.assertEqual(run(path,root/'nodata')['status'],'exit_goal_not_traversable')
            with rasterio.open(root/'error_bound.tif','r+') as dst:
                array = dst.read(); array[0,20,20] = -.01; dst.write(array)
            with self.assertRaisesRegex(ValueError,'negatif'):
                load_manifest(path)

    def test_nominal_run_does_not_claim_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _,path = self.fixture(root)
            result = run(path,root/'out')
            self.assertEqual(result['elevation_quality']['mode'],'nominal')
            self.assertIsNone(result['elevation_quality']['vertical_error_bound_m'])
            self.assertEqual(result['best']['terrain_value_kind'],'nominal')

    def test_quality_mask_nodata_blocks_finite_height_in_nominal_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec,path = self.fixture(root)
            spec['elevation_quality'] = {'mode':'nominal','validity_mask':'quality_mask.tif','source':'known interpolated patch'}
            path.write_text(json.dumps(spec),encoding='utf-8')
            with rasterio.open(root/'quality_mask.tif','r+') as dst:
                array = dst.read(); array[0,45,75] = 255; dst.write(array); dst.nodata = 255
            data = load_manifest(path)
            self.assertTrue(np.isfinite(data.elevation[45,75]))
            self.assertFalse(data.elevation_validity[45,75])
            result = run(path,root/'out')
            self.assertEqual(result['status'],'exit_goal_not_traversable')
            self.assertIn('unknown_elevation_quality',result['point_checks']['exit']['failed_center_constraints'])

    def test_failed_rerun_does_not_leave_previous_recommendation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec,path = self.fixture(root)
            output = root/'out'
            self.assertIsNotNone(run(path,output)['best'])
            spec['elevation_quality'] = {'mode':'bounded_error','vertical_error_bound_m':-.01,'source':'invalid fixture'}
            path.write_text(json.dumps(spec),encoding='utf-8')
            with self.assertRaises(ValueError):
                run(path,output)
            result = json.loads((output/'result.json').read_text(encoding='utf-8'))
            self.assertEqual(result['status'],'analysis_failed')
            self.assertIsNone(result['best'])
            self.assertFalse(result['artifacts_valid'])
            self.assertFalse((output/'selection.png').exists())
            self.assertFalse((output/'maps.npz').exists())
            self.assertTrue((root/'elevation.tif').exists())

    def test_output_name_collision_never_overwrites_source_or_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec,path = self.fixture(root)
            (root/'rgb.tif').rename(root/'selection.png')
            spec['rgb'] = 'selection.png'
            path.write_text(json.dumps(spec),encoding='utf-8')
            before = (root/'selection.png').read_bytes()
            with self.assertRaisesRegex(ValueError,'çakışıyor'):
                run(path,root)
            self.assertEqual((root/'selection.png').read_bytes(),before)
            spec['elevation_quality'] = {'mode':'bounded_error'}
            manifest = root/'result.json'
            manifest.write_text(json.dumps(spec),encoding='utf-8')
            before = manifest.read_bytes()
            with self.assertRaises(ValueError):
                run(manifest,root)
            self.assertEqual(manifest.read_bytes(),before)
            self.assertTrue((root/'selection.png').exists())
            # Hata kaydının geçici adı bile giriş olarak kullanılıyorsa korunmalı.
            (root/'selection.png').rename(root/'result.json.part')
            spec.pop('elevation_quality')
            spec['rgb'] = 'result.json.part'
            path.write_text(json.dumps(spec),encoding='utf-8')
            manifest.write_text(json.dumps({'report_type':'roversafe_start_area_v1','status':'candidate_selected','best':{}}),encoding='utf-8')
            before = (root/'result.json.part').read_bytes()
            with self.assertRaisesRegex(ValueError,'çakışıyor'):
                run(path,root)
            self.assertEqual((root/'result.json.part').read_bytes(),before)


if __name__ == '__main__':
    unittest.main()
