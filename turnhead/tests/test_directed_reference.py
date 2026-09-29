import unittest

from src.turn_detector import TurnDetector
from test_turn_detector import C, direction


class DirectedReferenceTests(unittest.TestCase):
    def replay(self, turn, angles, start=0):
        rows, events = [], []
        for i, angle in enumerate(angles):
            row, event = turn.update((start+i)*.1, direction(angle))
            rows.append(row)
            if event:
                events.append(event)
        return rows, events

    def test_both_rotation_directions_stay_turned_at_opposite_heading(self):
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                turn = TurnDetector(1, 30, C)
                angles = [35]*6 + [35+sign*a for a in range(15, 181, 15)] + [35+sign*180]*8
                rows, events = self.replay(turn, angles)
                self.assertEqual(len(events), 1)
                self.assertEqual(rows[-1]['state'], 'TURNED')
                self.assertEqual(rows[-1]['normal_reference_angle'], 35)
                self.assertAlmostEqual(rows[-1]['smoothed_reference_diff_deg'], 180)
                self.assertAlmostEqual(rows[-1]['smoothed_axis_diff_deg'], 5)
                self.assertGreater(events[0]['max_reference_diff'], C['turn_threshold_deg'])

    def test_return_requires_original_heading_then_allows_next_event(self):
        turn = TurnDetector(1, 30, C)
        outward = [30]*6 + list(range(45, 211, 15)) + [210]*6
        rows, events = self.replay(turn, outward)
        self.assertEqual(rows[-1]['state'], 'TURNED')
        back = list(range(195, 29, -15)) + [30]*6
        rows, extra = self.replay(turn, back, len(outward))
        self.assertIn('RETURNING', [r['state'] for r in rows])
        self.assertEqual(rows[-1]['state'], 'NORMAL')
        self.assertIsNone(rows[-1]['event_id'])
        self.assertEqual(extra, [])
        _, extra = self.replay(turn, list(range(45, 121, 15))+[120]*6, len(outward)+len(back))
        self.assertEqual([e['event_id'] for e in events+extra], ['P1-E1', 'P1-E2'])

    def test_reference_does_not_drift_with_gradual_motion(self):
        turn = TurnDetector(1, 30, C)
        rows, events = self.replay(turn, [30]*6 + list(range(31, 91)) + [90]*6)
        for row in rows[5:]:
            self.assertAlmostEqual(row['normal_reference_angle'], 30)
        self.assertEqual(len(events), 1)

    def test_wraparound_uses_small_angular_distance(self):
        turn = TurnDetector(1, 0, C)
        rows, _ = self.replay(turn, [355]*6 + [0, 5, 10])
        self.assertEqual(rows[-1]['normal_reference_angle'], 355)
        self.assertAlmostEqual(rows[-1]['smoothed_reference_diff_deg'], 15)

    def test_invalid_does_not_make_opposite_direction_a_new_normal(self):
        turn = TurnDetector(1, 30, C)
        self.replay(turn, [30]*6 + [45, 60])
        turn.update(.8, dict(direction_valid=False, pose_status='LOW_PROJECTION'))
        rows, events = self.replay(turn, [210]*15, 9)
        self.assertAlmostEqual(rows[-1]['normal_reference_angle'], 30)
        self.assertEqual(rows[-1]['state'], 'UNKNOWN')
        self.assertEqual(events, [])

    def test_people_facing_opposite_directions_have_independent_references(self):
        left, right = TurnDetector(1, 30, C), TurnDetector(2, 30, C)
        self.replay(left, [30]*6)
        self.replay(right, [210]*6)
        self.assertAlmostEqual(left.normal_reference_angle, 30)
        self.assertAlmostEqual(right.normal_reference_angle, 210)


if __name__ == '__main__':
    unittest.main()
