import math
import unittest
import numpy as np
from src.head_direction import axis_angle_diff_deg, smooth_direction, direction_from_pose


class DirectionTests(unittest.TestCase):
    def test_axis_examples_and_range(self):
        for head, expected in [(30, 0), (210, 0), (40, 10), (200, 10), (120, 90), (90, 60)]:
            self.assertEqual(axis_angle_diff_deg(head, 30), expected)
        for head in range(-720, 721):
            self.assertTrue(0 <= axis_angle_diff_deg(head, 30) <= 90)

    def test_wrap_smoothing(self):
        vectors = [(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in (359, 1)]
        angle = smooth_direction(vectors)[2]
        self.assertLess(min(angle, 360-angle), 1e-6)

    def test_opposite_cancellation(self):
        self.assertIsNone(smooth_direction([(1, 0), (-1, 0)]))

    def test_rotation_column_and_quality(self):
        # z-forward rotates to +x: third column, not third row.
        pose = dict(rotation_matrix=[[0, 0, 1], [0, 1, 0], [-1, 0, 0]], yaw=0, pitch=0, roll=0)
        config = dict(min_projection_length=.12, max_abs_pitch_deg=90)
        result = direction_from_pose(pose, config)
        self.assertTrue(result['direction_valid'])
        self.assertEqual(result['head_angle'], 0)
        pose['pitch'] = 90
        self.assertEqual(direction_from_pose(pose, config)['pose_status'], 'OUT_OF_SCOPE')
        pose.update(pitch=0, rotation_matrix=np.eye(3).tolist())
        self.assertEqual(direction_from_pose(pose, config)['pose_status'], 'LOW_PROJECTION')


if __name__ == '__main__':
    unittest.main()
