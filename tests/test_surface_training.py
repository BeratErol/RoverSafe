"""Yüzey özellikleri, model bakış açısı ve dondurulmuş eğitim seçimini doğrular."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from surface_classification import MULTISCALE_VERSION, feature_grid, SurfaceConfig, SurfacePredictor
from scripts.improve_surface import choose
from rugd_data import load_manifest
from potsdam_data import sha256
from configuration import cli_defaults

class SurfaceTrainingTests(unittest.TestCase):
    def test_context_training_freezes_selection_before_test(self):
        from types import SimpleNamespace
        from scripts.train_context_surface import study
        from rugd_data import iter_samples
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for i,split in enumerate(('development','validation','test')):
                rgb = np.zeros((96,96,3),'uint8'); rgb[:,:48,0] = 200; rgb[:,48:,1] = 200; rgb[0,0,2] = i
                labels = np.zeros((96,96),'uint8'); labels[:,:48] = 1
                image,mask = root/(split+'.png'),root/(split+'_mask.png')
                Image.fromarray(rgb).save(image); Image.fromarray(labels).save(mask)
                records.append({'id':split,'scene_id':split,'split':split,'image':str(image),'mask':str(mask),
                                'image_sha256':sha256(image),'mask_sha256':sha256(mask)})
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps({'split':'development_validation_test','samples':records,
                'class_policy':{'target':[1],'risk':[0],'unknown':[255]}}),encoding='utf-8')
            weights = root/'weights.pth'; weights.write_bytes(b'fixture')
            output = root/'study'
            seen = []
            def read(path,split):
                if split == 'test':
                    self.assertTrue((output/'selection.json').exists())
                seen.append(split)
                return iter_samples(path,split)
            fake = SimpleNamespace(sha256=sha256(weights),features=lambda image,step:
                np.tile(image[::step,::step].astype('float32')/255,(1,1,140))[:,:,:419])
            with patch('scripts.train_context_surface.ContextEncoder',return_value=fake),patch('scripts.train_context_surface.iter_samples',side_effect=read):
                result = study(manifest,weights,'ground',output)
            self.assertEqual(seen,['development','validation','test'])
            self.assertTrue(result['passes_test_gate'])
            self.assertEqual(result['test_sample_ids'],['test'])

    def test_context_cli_configuration_keeps_encoder_and_viewpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config.json'
            path.write_text(json.dumps({'surface':{'model_path':'model.joblib','encoder_path':'encoder.pth',
                                                  'expected_viewpoint':'ground'}}),encoding='utf-8')
            defaults = cli_defaults(path)
            self.assertEqual(defaults['surface_encoder'],'encoder.pth')
            self.assertEqual(defaults['surface_viewpoint'],'ground')

    @unittest.skipUnless(importlib.util.find_spec('torchvision'),'İsteğe bağlı bağlam bağımlılıkları yok')
    def test_context_encoder_checks_hash_and_feature_shape(self):
        import torch
        from torchvision.models import resnet18
        from context_features import ContextEncoder
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'encoder.pth'
            torch.manual_seed(42)
            torch.save(resnet18(weights=None).state_dict(),path)
            encoder = ContextEncoder(path,sha256(path))
            features = encoder.features(np.zeros((48,64,3),'uint8'))
            self.assertEqual(features.shape,(12,16,419))
            self.assertTrue(np.isfinite(features).all())
            with self.assertRaisesRegex(ValueError,'özeti'):
                ContextEncoder(path,'changed')

    def test_multiscale_features_are_finite_and_keep_source_color(self):
        image = np.random.default_rng(42).integers(0,256,(43,55,3),dtype='uint8')
        result = feature_grid(image,4,MULTISCALE_VERSION)
        self.assertEqual(result.shape,(11,14,35))
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result[:,:,:3],image[::4,::4]/255,atol=1e-7)

    def test_ground_model_cannot_silently_become_aerial_model(self):
        import joblib
        from sklearn.ensemble import ExtraTreesClassifier
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.joblib'
            classifier = ExtraTreesClassifier(n_estimators=2,random_state=42).fit(np.array([[0]*35,[1]*35]),[0,1])
            joblib.dump({'feature_version':MULTISCALE_VERSION,'viewpoint':'ground','model':classifier},path)
            with self.assertRaisesRegex(ValueError,'bakış açısı'):
                SurfacePredictor(SurfaceConfig(model_path=str(path),expected_viewpoint='aerial'))
            model = SurfacePredictor(SurfaceConfig(model_path=str(path),expected_viewpoint='ground'))
            self.assertEqual(model.predict(np.zeros((24,28,3),'uint8')).probability.shape,(24,28))

    def test_failed_validation_gate_is_not_reported_as_pass(self):
        rows = [{'model':'one','precision':.7,'recall':.6,'f1':.65,'threshold':.6},
                {'model':'two','precision':.9,'recall':.1,'f1':.18,'threshold':.9}]
        row,passed = choose(rows)
        self.assertFalse(passed)
        self.assertEqual(row['model'],'one')

    def test_rugd_sequence_and_duplicate_hash_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = [{'id':str(i),'scene_id':str(i),'split':split,'image_sha256':str(i)}
                       for i,split in enumerate(('development','validation','test'))]
            spec = {'split':'development_validation_test','samples':samples,
                    'class_policy':{'target':[1],'risk':[0],'unknown':[255]}}
            path = root/'manifest.json'
            path.write_text(json.dumps(spec),encoding='utf-8')
            self.assertEqual(len(load_manifest(path)['samples']),3)
            samples[2]['image_sha256'] = samples[0]['image_sha256']
            path.write_text(json.dumps(spec),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'bölümler'):
                load_manifest(path)
