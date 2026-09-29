import base64
import contextlib
import io
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from src.event_report import EventReport, video_time
from src.pipeline import ROOT, run
from test_video_duration import FakeCapture


class EventReportTests(unittest.TestCase):
    def test_full_frame_embedded_and_two_people_on_same_frame(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as directory:
            path = Path(directory)/'中文报告.html'
            frame = np.full((600, 1000, 3), 100, dtype=np.uint8)
            original = frame.copy()
            report = EventReport(path, '<教室&>.mp4')
            for ident, x in [(16, 50), (17, 650)]:
                event = dict(event_id=f'P{ident}-E1', person_id=ident, start_time=64.9, confirmed_time=65.05)
                person = dict(body_box=[x, 100, x+200, 550], head_box=[x+30, 110, x+120, 200])
                report.add(event, frame, person, 1301)
                report.add(event, frame, person, 1301)
            # Each entry is on disk already, before finishing the video.
            html = path.read_text(encoding='utf-8')
            self.assertEqual(html.count('<article '), 2)
            self.assertIn('00:01:05.050', html)
            self.assertIn('&lt;教室&amp;&gt;.mp4', html)
            images = re.findall(r'src="data:image/jpeg;base64,([^"]+)"', html)
            decoded = [cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR) for data in images]
            self.assertEqual(len(decoded), 2)
            self.assertTrue(all(image.shape == original.shape for image in decoded))
            self.assertFalse(np.array_equal(decoded[0], decoded[1]))
            np.testing.assert_array_equal(frame, original)
            report.finish(True)
            self.assertIn('共保存 2 次', path.read_text(encoding='utf-8'))
            with self.assertRaises(FileExistsError):
                EventReport(path, 'second.mp4')

    def test_empty_and_interrupted_reports(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as directory:
            for complete in (True, False):
                path = Path(directory)/f'{complete}.html'
                report = EventReport(path, 'video.mp4')
                report.finish(complete)
                html = path.read_text(encoding='utf-8')
                self.assertIn('尚无已确认的转头事件', html)
                self.assertIn('处理完成' if complete else '处理已中断', html)
                self.assertTrue(html.endswith('</main></body></html>\n'))
        self.assertEqual(video_time(3599.9996), '01:00:00.000')

    def test_pipeline_keeps_event_report_on_keyboard_interrupt(self):
        cap = FakeCapture()
        pipeline = Mock()
        person = dict(person_id=16, pose=None, is_standing=False, transition_status='VALID',
                      state='TURNED', smoothed_axis_diff_deg=60,
                      body_box=[2, 2, 25, 30], head_box=[4, 4, 16, 16])
        event = dict(event_id='P16-E1', person_id=16, start_time=0, confirmed_time=0)
        pipeline.process.side_effect = [([person], [event]), KeyboardInterrupt()]
        config = dict(standing_filter={'enabled': False}, device='cpu', reference={'axis_degrees': 30}, display={})
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as directory:
            root = Path(directory)
            video = root/'source.mp4'
            video.touch()
            with (patch('src.pipeline.cv2.VideoCapture', return_value=cap),
                  patch('src.pipeline.Pipeline', return_value=pipeline),
                  patch('src.pipeline.config_snapshot', return_value=config),
                  patch('src.pipeline.draw_frame', side_effect=lambda frame, *args: frame),
                  contextlib.redirect_stdout(io.StringIO())):
                with self.assertRaises(KeyboardInterrupt):
                    run(config, video, root/'result')
            html = (root/'result/turn_report.html').read_text(encoding='utf-8')
            self.assertEqual(html.count('<article '), 1)
            self.assertIn('处理已中断', html)
            self.assertIn('data:image/jpeg;base64,', html)
            self.assertTrue(cap.released)


if __name__ == '__main__':
    unittest.main()
