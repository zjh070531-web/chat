import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch
import unittest
import cv2
import numpy as np
from src.pipeline import ROOT, cli, run


class FakeCapture:
    """13.5 second source with deliberately underestimated total-frame metadata."""
    def __init__(self):
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, key):
        return 2. if key == cv2.CAP_PROP_FPS else 20.

    def read(self):
        if self.index == 27:
            return False, None
        self.index += 1
        return True, np.full((32,32,3), self.index, dtype=np.uint8)

    def release(self):
        self.released = True


class VideoDurationTests(unittest.TestCase):
    def test_cli_default_is_full_video(self):
        config = {'video':Path('videos/test.mp4'), 'output_dir':Path('outputs')}
        with patch('sys.argv',['main.py']), patch('src.pipeline.read_config',return_value=config), patch('src.pipeline.run') as runner:
            cli()
        self.assertIsNone(runner.call_args.args[-1])

    def test_cli_explicit_seconds(self):
        config = {'video':Path('videos/test.mp4'), 'output_dir':Path('outputs')}
        with patch('sys.argv',['main.py','--seconds','10']), patch('src.pipeline.read_config',return_value=config), patch('src.pipeline.run') as runner:
            cli()
        self.assertEqual(runner.call_args.args[-1],10.)

    def exercise(self, seconds):
        cap = FakeCapture()
        pipeline = Mock()
        pipeline.process.return_value = ([],[])
        pipeline.pose.person_calls = 0
        config = dict(standing_filter={'enabled':False}, device='cpu', reference={'axis_degrees':30}, display={})
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as directory:
            root = Path(directory)
            video = root/'source.mp4'
            video.touch()
            output = root/'result'
            with (patch('src.pipeline.cv2.VideoCapture',return_value=cap),
                  patch('src.pipeline.Pipeline',return_value=pipeline),
                  patch('src.pipeline.config_snapshot',return_value=config),
                  patch('src.pipeline.draw_frame',side_effect=lambda frame,*args:frame),
                  contextlib.redirect_stdout(io.StringIO())):
                run(config,video,output,seconds)
            summary = json.loads((output/'summary.json').read_text(encoding='utf-8'))
            self.assertTrue((output/f"preview_{summary['frames']-1:04d}.jpg").is_file())
            self.assertTrue(cap.released)
            return summary,pipeline.process.call_count

    def test_full_video_continues_beyond_ten_seconds_and_metadata(self):
        summary,calls = self.exercise(None)
        self.assertEqual(calls,27)
        self.assertEqual(summary['frames'],27)
        self.assertEqual(summary['processed_seconds'],13.5)
        self.assertIsNone(summary['requested_seconds'])
        self.assertEqual(summary['processing_scope'],'full_video')

    def test_explicit_duration_stops_at_limit(self):
        summary,calls = self.exercise(3.)
        self.assertEqual(calls,6)
        self.assertEqual(summary['processed_seconds'],3.)
        self.assertEqual(summary['processing_scope'],'duration_limit')


if __name__ == '__main__':
    unittest.main()
