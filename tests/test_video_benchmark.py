"""Video ölçümünde aynı kareleri, çıktı okunabilirliğini ve girdi kayıtlarını doğrular."""
import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from potsdam_data import sha256
from scripts.benchmark_video import benchmark


class VideoBenchmarkTests(unittest.TestCase):
    def test_sampling_and_output_decode_preserve_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'source.avi'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),10,(80,60))
            self.assertTrue(writer.isOpened())
            try:
                for value in range(6):
                    writer.write(np.full((60,80,3),100+value,'uint8'))
            finally:
                writer.release()
            before=sha256(source)
            result=benchmark(source,root/'result',width=64,frames=6,repeats=1,strides=(1,2))
            self.assertTrue(result['completed'])
            self.assertEqual(result['input']['benchmark_size'],[64,48])
            self.assertEqual(result['input']['frames'],6)
            self.assertEqual(result['input']['source_sha256'],before)
            self.assertEqual(sha256(source),before)
            self.assertEqual([row['processed_frames'] for row in result['results']],[6,3])
            for row in result['results']:
                self.assertTrue(row['output_decode_passed'])
                case=root/'result'/row['artifacts']
                logs=[json.loads(line) for line in (case/'frames.jsonl').read_text().splitlines()]
                self.assertEqual([record['frame_index'] for record in logs],list(range(0,6,row['stride'])))
                self.assertEqual(row['settings_sha256'],sha256(case/'settings.json'))
            self.assertTrue((root/'result/report.html').is_file())
            with self.assertRaises(FileExistsError):
                benchmark(source,root/'result')
            self.assertEqual(sha256(source),before)

    def test_invalid_settings_fail_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'out'
            for args in ({'width':0},{'frames':True},{'repeats':-1},{'strides':()},{'strides':(0,)}):
                with self.subTest(args=args),self.assertRaises(ValueError):
                    benchmark(Path(directory)/'missing.avi',output,**args)
            self.assertFalse(output.exists())
