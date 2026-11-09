"""Yoğun tahminde pencere hizalamasını ve testten önce seçim dondurmayı kontrol eder."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from dense_surface import DenseSurfacePredictor,DENSE_VERSION
from potsdam_data import sha256


class DenseAerialTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torchvision'),'İsteğe bağlı bağlam bağımlılıkları yok')
    def test_binary_dense_bundle_loads_through_shared_surface_api(self):
        import torch
        from dense_surface import make_head
        from surface_classification import SurfaceConfig,SurfacePredictor
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.pth'
            torch.save({'feature_version':DENSE_VERSION,'viewpoint':'aerial','num_classes':2,
                        'encoder_sha256':'fixed','head':make_head(2).state_dict(),
                        'mean':torch.zeros(1,419,1,1),'std':torch.ones(1,419,1,1)},path)
            fake=SimpleNamespace(sha256='fixed',features=lambda image,step:np.zeros((16,16,419),'float32'))
            with patch('context_features.ContextEncoder',return_value=fake) as encoder:
                predictor=SurfacePredictor(SurfaceConfig(model_path=str(path),encoder_path='encoder.pth',expected_viewpoint='aerial'))
                result=predictor.predict(np.zeros((64,64,3),'uint8'))
                self.assertEqual(result.probability.shape,(64,64))
                self.assertTrue(np.isfinite(result.probability).all())
                encoder.assert_called_once_with('encoder.pth','fixed')
            with self.assertRaisesRegex(ValueError,'bakış açısı'):
                SurfacePredictor(SurfaceConfig(model_path=str(path),encoder_path='encoder.pth',expected_viewpoint='ground'))

    def predictor(self):
        value = DenseSurfacePredictor.__new__(DenseSurfacePredictor)
        value.config = SimpleNamespace(threshold=.5)
        value.model_sha256='fixture'; value.encoder=SimpleNamespace(sha256='encoder')
        value.feature_version=DENSE_VERSION; value.viewpoint='aerial'
        value.score_patch=lambda image:image[:,:,0].astype('float32')/255
        return value

    def test_tiling_covers_non_multiple_shapes_without_color_or_position_shift(self):
        for height,width in ((32,33),(512,512),(513,961),(1025,549)):
            image=np.random.default_rng(42).integers(0,256,(height,width,3),dtype='uint8')
            result=self.predictor().predict(image)
            np.testing.assert_allclose(result.probability,image[:,:,0]/255,atol=2e-7)
            np.testing.assert_array_equal(result.exclusion_mask,image[:,:,0]<128)

    def test_training_size_does_not_create_redundant_tail_tiles(self):
        predictor=self.predictor()
        with patch.object(predictor,'score_patch',wraps=predictor.score_patch) as score:
            predictor.predict(np.zeros((512,512,3),'uint8'))
            self.assertEqual(score.call_count,1)

    @unittest.skipUnless(importlib.util.find_spec('torchvision'),'İsteğe bağlı bağlam bağımlılıkları yok')
    def test_dense_training_freezes_selection_before_regression_test(self):
        from scripts.train_dense_aerial import study
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            protocol=root/'protocol.json'; protocol.write_text('{}')
            weights=root/'encoder.pth'; weights.write_bytes(b'fixture')
            output=root/'study'; seen=[]
            rgb=np.random.default_rng(42).integers(0,256,(96,96,3),dtype='uint8')
            classes=np.tile((np.arange(96)//8)%6,(96,1)).astype('int8')
            def iterator(path,split):
                seen.append(split)
                if split == 'test': self.assertTrue((output/'selection.json').is_file())
                yield split,(0,0,96,96),rgb,(classes==0).astype('int8'),classes
            fake=SimpleNamespace(sha256=sha256(weights),features=lambda image,step:
                np.tile(image[::step,::step].astype('float32')/255,(1,1,140))[:,:,:419])
            prediction=SimpleNamespace(probability=np.full((96,96),.1,'float32'))
            with patch('scripts.train_dense_aerial.ContextEncoder',return_value=fake),\
                    patch('scripts.train_dense_aerial.iter_patches',side_effect=iterator),\
                    patch('surface_classification.SurfacePredictor',return_value=SimpleNamespace(predict=lambda image:prediction)):
                result=study(protocol,weights,output,epochs=1)
            self.assertEqual(seen,['development','validation','test'])
            self.assertEqual(result['selection_sha256'],sha256(output/'selection.json'))
            with self.assertRaises(ValueError):
                study(protocol,weights,output,epochs=1,resume_cache=True)


if __name__ == '__main__': unittest.main()
