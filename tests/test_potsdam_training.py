"""Karo ayrımı, etiket politikası ve doğrulama/test sınırlarını sınar."""

import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import rasterio
from rasterio.transform import from_origin
from potsdam_data import build_protocol,decode_labels,iter_patches,tile_names,windows
if importlib.util.find_spec('sklearn') is None or importlib.util.find_spec('joblib') is None:
    raise unittest.SkipTest('Hedef model eğitimi için requirements-surface.txt gerekli.')
from scripts.train_potsdam_surface import train,select_threshold,evaluate_test,metric_counts,metrics
from scripts.create_native_start_manifest import create
from potsdam_data import sha256


class PotsdamTrainingTests(unittest.TestCase):
    def fixture(self, root):
        splits = {'development':['2_11'],'validation':['4_11'],'test':['5_11']}
        for index,tile in enumerate(('2_11','4_11','5_11')):
            rgb = np.full((3,96,96),150,np.uint8)
            rgb[:,:,48:] = np.array([20,100,20])[:,None,None]
            rgb[:,0,0] = index
            labels = np.full((3,96,96),255,np.uint8)
            labels[:,:,48:] = np.array([0,255,0])[:,None,None]
            labels[:,0,:index+1] = 0
            for name,array in zip(tile_names(tile)[:2],(rgb,labels)):
                with rasterio.open(root/name,'w',driver='GTiff',width=96,height=96,count=3,
                                   dtype='uint8',crs='EPSG:32633',transform=from_origin(500000+index*20,4500000,.1,.1)) as dst:
                    dst.write(array)
        protocol = root/'protocol.json'
        build_protocol(root,protocol,splits,patch_size=32,stride=32,guard=8)
        return splits,protocol

    def test_known_colors_and_unknown_are_distinct(self):
        rgb = np.array([[[255,255,255],[0,0,255],[255,255,0],[0,0,0]]],np.uint8)
        binary,classes = decode_labels(rgb)
        np.testing.assert_array_equal(binary,[[1,0,0,-1]])
        self.assertEqual(classes[0,2],4)
        np.testing.assert_array_equal(metric_counts(np.ones((1,4)),binary,.6),[1,2,0,0])
        self.assertAlmostEqual(metrics([1,2,0,0])['iou'],1/3)

    def test_window_guards_and_nonoverlap(self):
        crops = windows(96,96,32,32,8)
        self.assertEqual(len(crops),4)
        self.assertTrue(all(x >= 8 and y >= 8 and x+w <= 88 and y+h <= 88 for x,y,w,h in crops))
        for args in ((96,96,32,16,8),(96,96,32,32,0),(96,96,33,40,8)):
            with self.assertRaises(ValueError):
                windows(*args)
        with self.assertRaises(ValueError):
            tile_names('../../file')

    def test_duplicate_tiles_inspected_tile_and_geographic_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits,protocol = self.fixture(root)
            for bad in (dict(splits,test=['2_11']),dict(splits,test=['2_10'])):
                with self.assertRaises(ValueError):
                    build_protocol(root,root/'bad.json',bad,32,32,8)
            for name in tile_names('4_11')[:2]:
                with rasterio.open(root/name,'r+') as dst:
                    dst.transform = from_origin(500000,4500000,.1,.1)
            with self.assertRaisesRegex(ValueError,'örtüşüyor'):
                build_protocol(root,root/'overlap.json',splits,32,32,8)

    def test_source_changes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _,protocol = self.fixture(root)
            with rasterio.open(root/tile_names('2_11')[0],'r+') as dst:
                array = dst.read(); array[:,20,20] = 10; dst.write(array)
            with self.assertRaisesRegex(ValueError,'uyuşmuyor'):
                list(iter_patches(protocol,'development'))

    def test_train_validation_and_frozen_test_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _,protocol = self.fixture(root)
            model = root/'model.joblib'
            splits_read = []
            def before_test(path, split):
                self.assertNotEqual(split,'test')
                splits_read.append(split)
                return iter_patches(path,split)
            with patch('scripts.train_potsdam_surface.iter_patches',side_effect=before_test):
                training = train(protocol,model)
                selection = select_threshold(protocol,model,root/'selection.json')
            self.assertEqual(splits_read,['development','validation'])
            self.assertEqual({row['tile'] for row in training['used_windows']},{'2_11'})
            self.assertEqual({row['tile'] for row in selection['windows']},{'4_11'})
            result = evaluate_test(protocol,model,root/'selection.json',model,root/'results')
            self.assertEqual(result['test_tiles'],['5_11'])
            self.assertEqual(result['models']['potsdam_native']['iou'],1.0)
            selection['model_sha256'] = 'changed'
            (root/'selection.json').write_text(json.dumps(selection),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Dondurulmuş'):
                evaluate_test(protocol,model,root/'selection.json',model,root/'bad_test')

    def test_native_manifest_preserves_mission_and_rebases_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root/'model.joblib'
            model.write_bytes(b'fixture only; not loaded')
            protocol = root/'protocol.json'
            protocol.write_text('{}',encoding='utf-8')
            selection = root/'selection.json'
            selection.write_text(json.dumps({'selection_split':'validation','threshold':.9,
                'feature_grid_step':4,'model_sha256':sha256(model),'protocol_sha256':sha256(protocol)}),encoding='utf-8')
            base = root/'base.json'
            mission = {'mode':'assess_fixed_start','fixed_start_pixel':[10,10],'exit_pixel':[20,20]}
            base.write_text(json.dumps({'rgb':'rgb.tif','elevation':'z.tif','allowed_mask':'allowed.tif',
                'start_zone':'zone.tif','ground_mask':'reference.tif','mission':mission,
                'elevation_quality':{'mode':'bounded_error','error_bound_raster':'bound.tif','validity_mask':'quality.tif','source':'fixture'},
                'source_sha256':{'label.tif':'label_hash','rgb.tif':'rgb_hash'}}),encoding='utf-8')
            output = root/'nested/new.json'
            spec = create(base,output,model,selection,protocol)
            self.assertEqual(spec['mission'],mission)
            self.assertNotIn('ground_mask',spec)
            self.assertNotIn('label.tif',spec['source_sha256'])
            self.assertEqual((output.parent/spec['rgb']).resolve(),root/'rgb.tif')
            self.assertEqual(spec['surface']['threshold'],.9)
            self.assertEqual((output.parent/spec['elevation_quality']['error_bound_raster']).resolve(),root/'bound.tif')
            self.assertEqual((output.parent/spec['elevation_quality']['validity_mask']).resolve(),root/'quality.tif')
            self.assertEqual(spec['surface_selection']['model_sha256'],sha256(model))


if __name__ == '__main__':
    unittest.main()
