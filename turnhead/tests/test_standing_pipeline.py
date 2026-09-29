import unittest
from unittest.mock import Mock
import numpy as np
from src.pipeline import Pipeline, read_config, ROOT
from src.tracker import PersonTracker


class StandingPipelineTests(unittest.TestCase):
    def setup_pipeline(self):
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.c = read_config(ROOT/'config.yaml')
        # This fixture uses 0.05s steps and a 0.4s normal-acquisition interval.
        # Keep its timing independent of the user's current tuning parameters.
        pipeline.c['temporal']['min_normal_stable_seconds'] = .4
        pipeline.tracker = PersonTracker()
        pipeline.detector = Mock()
        pipeline.detector.predict.side_effect = lambda frame: [
            dict(body_box=[0,0,100,300], head_box=[20,10,60,50], pose=None),
            dict(body_box=[150,0,250,300], head_box=[170,10,210,50], pose=None)]
        pipeline.pose = Mock()
        pipeline.batch_size = 8
        # Valid normal 30 degree projected direction, preserving a proper rotation.
        angle = np.radians(30)
        matrix = [[0,-np.sin(angle),np.cos(angle)], [0,np.cos(angle),np.sin(angle)],[-1,0,0]]
        pipeline.pose.predict_batch.side_effect = lambda items,batch: [dict(rotation_matrix=matrix, yaw=0.,pitch=0.,roll=0.) for _ in items]
        pipeline.turns = {}
        pipeline.standing = Mock()
        standing = [True]
        def mark(people, frame, t, active):
            for i,p in enumerate(people):
                p.update(is_standing=bool(i==0 and standing[0]), posture_label='STANDING' if i==0 and standing[0] else 'SITTING',
                         turn_excluded_reason='STANDING' if i==0 and standing[0] else None)
        pipeline.standing.update.side_effect = mark
        return pipeline, standing

    def test_standing_skips_head_pose_and_turn_evidence(self):
        pipeline, standing = self.setup_pipeline()
        frame = np.zeros((400,300,3), dtype=np.uint8)
        for i in range(12):
            people, events = pipeline.process(frame, i*.05)
            self.assertEqual(events, [])
            self.assertIsNone(people[0]['pose'])
            self.assertIsNone(people[0]['smoothed_head_angle'])
            self.assertEqual(people[0]['transition_status'], 'EXCLUDED_STANDING')
            self.assertEqual(people[0]['state'], 'UNKNOWN')
            self.assertEqual(len(pipeline.pose.predict_batch.call_args.args[0]), 1)
        self.assertEqual(people[1]['state'], 'NORMAL')
        standing[0] = False
        people,_ = pipeline.process(frame, .6)
        self.assertEqual(len(pipeline.pose.predict_batch.call_args.args[0]), 2)
        self.assertEqual(people[0]['state'], 'UNKNOWN')
        for i in range(13,22):
            people,_ = pipeline.process(frame, i*.05)
        self.assertEqual(people[0]['state'], 'NORMAL')

    def test_disabled_filter_runs_every_head(self):
        pipeline,_ = self.setup_pipeline()
        pipeline.standing = None
        people,events = pipeline.process(np.zeros((400,300,3),dtype=np.uint8),0)
        self.assertEqual(len(pipeline.pose.predict_batch.call_args.args[0]),2)
        self.assertTrue(all(not p['is_standing'] for p in people))
        self.assertEqual(events,[])


if __name__ == '__main__':
    unittest.main()
