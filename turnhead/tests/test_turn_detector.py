import math
import unittest
from src.turn_detector import TurnDetector

C = dict(normal_threshold_deg=10., turn_threshold_deg=35.,
         min_normal_stable_seconds=.4, min_turned_stable_seconds=.4,
         max_frame_direction_jump_deg=30., smooth_window_frames=1,
         min_smooth_resultant=.2, max_valid_gap_seconds=.3,
         min_transition_samples=3, trend_window_frames=3, trend_min_change_deg=2.)


def direction(angle):
    r = math.radians(angle)
    return dict(head_angle=angle % 360, head_direction_x=math.cos(r),
                head_direction_y=math.sin(r), direction_valid=True, pose_status='VALID')


class StateTests(unittest.TestCase):
    def setUp(self):
        self.turn = TurnDetector(1, 30., C)
        self.t = -.2

    def feed(self, differences, ray=30.):
        states, events = [], []
        for diff in differences:
            self.t += .2
            row, event = self.turn.update(self.t, direction(ray+diff))
            states.append(row['state'])
            if event:
                events.append(event)
        return states, events

    def test_normal(self):
        states, events = self.feed([3, 5, 7])
        self.assertEqual(states[-1], 'NORMAL')
        self.assertEqual(events, [])

    def test_continuous_turn_return_rearm(self):
        self.feed([3, 5, 7])
        states, events = self.feed([5, 9, 14, 20, 28, 36, 45, 45, 45])
        self.assertIn('TURNING', states)
        self.assertEqual(states[-1], 'TURNED')
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['state_sequence'], 'NORMAL -> TURNING -> TURNED')
        self.assertEqual(self.feed([42]*8)[1], [])
        states, _ = self.feed([42, 35, 25, 15, 8, 5, 5])
        self.assertIn('RETURNING', states)
        self.assertEqual(states[-1], 'NORMAL')
        self.assertIsNone(self.turn.event_id)
        self.assertEqual(len(self.feed([14, 20, 28, 36, 45, 45, 45])[1]), 1)

    def test_jump_no_event(self):
        self.feed([3, 5, 7])
        _, events = self.feed([5, 60, 6, 6, 6])
        self.assertEqual(events, [])
        self.assertEqual(self.turn.events, [])

    def test_raw_jump_not_hidden_by_smoothing(self):
        self.turn = TurnDetector(1, 30., {**C, 'smooth_window_frames': 5})
        self.feed([5]*5)
        self.t += .2
        row, event = self.turn.update(self.t, direction(90))
        self.assertEqual(row['transition_status'], 'POSE_JUMP')
        self.assertLess(row['direction_jump_deg'], 30)
        self.assertIsNone(event)

    def test_initial_turned_cannot_create_event(self):
        self.feed([40]*10)
        self.assertEqual(self.turn.state, 'UNKNOWN')
        self.assertEqual(self.turn.events, [])

    def test_invalid_interrupts_duration_and_path(self):
        self.feed([5]*3)
        self.feed([14, 20, 28, 36])
        self.t += .1
        old_state = self.turn.state
        row, _ = self.turn.update(self.t, dict(direction_valid=False, pose_status='OUT_OF_SCOPE'))
        self.assertEqual(row['state'], old_state)
        self.feed([40]*5)
        self.assertEqual(self.turn.events, [])

    def test_missing_gap_cannot_complete_event(self):
        self.feed([5]*3)
        self.feed([14, 20, 28, 36])
        self.t += 1
        self.feed([40]*5)
        self.assertEqual(self.turn.events, [])

    def test_opposite_row_and_independent_people(self):
        self.feed([3, 5, 7], ray=210)
        self.assertEqual(self.turn.active_reference_ray, 210)
        _, events = self.feed([14, 20, 28, 36, 45, 45, 45], ray=210)
        self.assertEqual(len(events), 1)
        other = TurnDetector(2, 30, C)
        self.assertEqual(other.state, 'UNKNOWN')
        self.assertEqual(other.events, [])

    def test_default_five_frame_smoothing(self):
        self.turn = TurnDetector(1, 30, {**C, 'smooth_window_frames': 5})
        self.feed([5]*8)
        _, events = self.feed([5, 9, 14, 20, 28, 36, 40]+[40]*8)
        self.assertEqual(len(events), 1)

    def test_standing_cancels_candidate_and_requires_fresh_normal(self):
        self.feed([5]*3)
        self.feed([14,20,28,36])
        row,event = self.turn.suspend(self.t+.05)
        self.assertIsNone(event)
        self.assertEqual(row['state'], 'UNKNOWN')
        self.assertEqual(row['transition_status'], 'EXCLUDED_STANDING')
        self.assertEqual(len(self.turn.history), 0)
        self.assertIsNone(self.turn.path)
        self.assertEqual(self.feed([40]*8)[1], [])
        self.feed([5]*4)
        self.assertEqual(len(self.feed([14,20,28,36,45,45,45])[1]), 1)

    def test_standing_clears_latch_but_preserves_unique_event_numbers(self):
        self.feed([5]*3)
        self.feed([14,20,28,36,45,45,45])
        self.assertEqual(self.turn.event_id, 'P1-E1')
        self.turn.suspend(self.t+.05)
        self.assertIsNone(self.turn.event_id)
        self.feed([5]*4)
        self.feed([14,20,28,36,45,45,45])
        self.assertEqual(self.turn.event_id, 'P1-E2')


if __name__ == '__main__':
    unittest.main()
