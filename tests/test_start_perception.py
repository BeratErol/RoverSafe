"""Otomatik modda etiket sızıntısını, nesne elemesini ve hata davranışını doğrular."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import rasterio
from rasterio.transform import from_origin
from geospatial_io import load_manifest
from obstacle_detection import ObstacleDetectionError
from start_area_cli import run
from start_area_perception import estimate_surfaces
from scripts.run_start_area_comparison import semantic_metrics


class StartPerceptionTests(unittest.TestCase):
    def inputs(self):
        rgb = np.zeros((120,160,3),np.uint8)
        rgb[:,:,0] = 200
        return SimpleNamespace(rgb=rgb,rgb_valid=np.ones(rgb.shape[:2],bool), ground=None,
            dx=.1,dy=.05,manifest={'ground_mode':'predicted','ground_mask_source':'test model',
                                  'surface':{'model_path':'model.joblib','threshold':.6}})

    def predictor(self, score):
        return SimpleNamespace(model_sha256='fixture_hash',
                               predict=lambda image: SimpleNamespace(probability=score))

    def test_bgr_input_threshold_and_rgb_nodata(self):
        data = self.inputs()
        data.rgb_valid[60,80] = False
        score = np.full(data.rgb.shape[:2],.7,np.float32)
        score[:,:40] = .59
        predictor = self.predictor(score)
        captured = []
        predictor.predict = lambda image: (captured.append(image.copy()) or SimpleNamespace(probability=score))
        with patch('start_area_perception.SurfacePredictor',return_value=predictor):
            result = estimate_surfaces(data,Path('example/manifest.json'))
        np.testing.assert_array_equal(captured[0][0,0], [0,0,200])
        self.assertFalse(result.ground[:,:40].any())
        self.assertFalse(result.ground[60,80])
        self.assertTrue(result.ground[60,81])
        self.assertEqual(result.metadata['ground_mode'],'predicted')
        self.assertEqual(result.metadata['model_sha256'],'fixture_hash')

    def test_invalid_scores_fail_without_fallback(self):
        data = self.inputs()
        for score in (np.full((120,160),np.nan), np.full((120,160),1.1),np.ones((10,10))):
            with patch('start_area_perception.SurfacePredictor',return_value=self.predictor(score)):
                with self.assertRaises(ValueError):
                    estimate_surfaces(data,'manifest.json')

    def test_frozen_selection_rejects_model_or_threshold_changes(self):
        data = self.inputs()
        data.manifest['surface_selection'] = {'split':'validation','model_sha256':'fixture_hash','threshold':.6}
        with patch('start_area_perception.SurfacePredictor',return_value=self.predictor(np.ones((120,160)))):
            self.assertEqual(estimate_surfaces(data,'manifest.json').metadata['surface_selection']['threshold'],.6)
            for key,value in (('threshold',.5),('model_sha256','other_model'),('split','test')):
                old = data.manifest['surface_selection'][key]
                data.manifest['surface_selection'][key] = value
                with self.assertRaisesRegex(ValueError,'dondurulmuş'):
                    estimate_surfaces(data,'manifest.json')
                data.manifest['surface_selection'][key] = old

    def test_metric_object_margin_and_failure(self):
        data = self.inputs()
        data.manifest['obstacles'] = {'model_path':'yolo.pt','margin_m':.21,'classes':['person']}
        blocked = np.zeros((120,160),bool)
        blocked[:,79:82] = True
        detection = SimpleNamespace(exclusion_mask=blocked,model_sha256='yolo_hash',
                                   ultralytics_version='fixture',inference_seconds=.01,detections=[])
        with patch('start_area_perception.SurfacePredictor',return_value=self.predictor(np.ones((120,160)))), \
             patch('start_area_perception.YOLOObstacleDetector') as detector:
            detector.return_value.detect.return_value = detection
            result = estimate_surfaces(data,'manifest.json')
            self.assertEqual(detector.call_args.args[0].margin,5)
            self.assertEqual(detector.call_args.args[0].classes,('person',))
            np.testing.assert_array_equal(result.obstacles,blocked)
            detector.return_value.detect.side_effect = ObstacleDetectionError('fixture failure')
            with self.assertRaises(ObstacleDetectionError):
                estimate_surfaces(data,'manifest.json')

    def make_fixture(self, root):
        spec = {'ground_mode':'predicted','elevation_unit':'m','elevation_kind':'DSM',
                'ground_mask_source':'fixture model','dataset_source':'synthetic','mission_source':'fixture',
                'surface':{'model_path':'model.joblib','threshold':.6},
                'mission':{'mode':'assess_fixed_start','fixed_start_pixel':[35,60],'exit_pixel':[130,60]}}
        for key in ('rgb','elevation','allowed_mask','start_zone'):
            spec[key] = key+'.tif'
            array = np.full((3,120,160),150,np.uint8) if key == 'rgb' else (
                np.zeros((1,120,160),np.float32) if key == 'elevation' else np.ones((1,120,160),np.uint8))
            with rasterio.open(root/spec[key],'w',driver='GTiff',height=120,width=160,count=len(array),
                               dtype=array.dtype,crs='EPSG:32636',transform=from_origin(500000,4500000,.1,.1)) as dst:
                dst.write(array)
        path = root/'manifest.json'
        path.write_text(json.dumps(spec),encoding='utf-8')
        return spec,path

    def test_predicted_run_needs_no_reference_and_object_wall_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec,path = self.make_fixture(root)
            self.assertIsNone(load_manifest(path).ground)
            with patch('start_area_perception.SurfacePredictor',return_value=self.predictor(np.ones((120,160)))):
                accepted = run(path,root/'accepted')
                self.assertEqual(accepted['status'],'fixed_start_accepted')
                self.assertEqual(accepted['perception']['ground_mode'],'predicted')
                spec['obstacles'] = {'model_path':'yolo.pt','margin_m':0}
                path.write_text(json.dumps(spec),encoding='utf-8')
                blocked = np.zeros((120,160),bool)
                blocked[:,79:82] = True
                detection = SimpleNamespace(exclusion_mask=blocked,model_sha256='fixture',
                              ultralytics_version='fixture',inference_seconds=.01,detections=[])
                with patch('start_area_perception.YOLOObstacleDetector') as detector:
                    detector.return_value.detect.return_value = detection
                    rejected = run(path,root/'rejected')
                    self.assertEqual(rejected['status'],'fixed_start_rejected')
                    self.assertGreater(rejected['rejected_pixels']['detected_obstacle'],0)
                    detector.return_value.detect.side_effect = ObstacleDetectionError('fixture error')
                    with self.assertRaises(ObstacleDetectionError):
                        run(path,root/'failed')
                    self.assertFalse((root/'failed/result.json').exists())

    def test_ambiguous_reference_and_predicted_modes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec,path = self.make_fixture(root)
            spec['ground_mask'] = 'must_not_be_opened.tif'
            path.write_text(json.dumps(spec),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'ground_mask'):
                load_manifest(path)
            spec.pop('ground_mask')
            spec['ground_mode'] = 'reference'
            path.write_text(json.dumps(spec),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'surface'):
                load_manifest(path)

    def test_semantic_metrics_do_not_turn_abstention_into_success(self):
        reference = np.zeros((12,12),bool)
        reference[:6] = True
        report = {'best':None}
        metrics = semantic_metrics(np.ones_like(reference),reference,report)
        self.assertEqual(metrics['precision'],.5)
        self.assertIsNone(metrics['footprint_non_target_fraction'])
        report.update(best={'footprint_with_margin_px':[4,4,4,4]},route_pixels=[[5,5]],
                      rover={'width_m':.2,'length_m':.2,'margin_m':0},resolution_m=[.1,.1])
        metrics = semantic_metrics(np.ones_like(reference),reference,report)
        self.assertEqual(metrics['footprint_non_target_fraction'],.5)
        self.assertGreater(metrics['route_swept_non_target_fraction'],0)


if __name__ == '__main__':
    unittest.main()
