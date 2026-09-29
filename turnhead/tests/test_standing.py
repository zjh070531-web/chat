import unittest
import numpy as np
from src.standing import DEFAULTS, classify_posture, StandingEvidence, match_pose_boxes


def skeleton(sitting=False, legs=True):
    points = np.zeros((17, 3))
    coords = {0:(100,10), 5:(80,50), 6:(120,50), 11:(85,150), 12:(115,150)}
    if legs:
        coords.update({13:(85,225),14:(115,225),15:(85,310),16:(115,310)} if not sitting else
                      {13:(160,150),14:(190,150),15:(160,230),16:(190,230)})
    for i,(x,y) in coords.items():
        points[i] = (x,y,.95)
    return points


def posture(score=.95, label='STANDING'):
    return dict(standing_score=score, posture_label=label)


class StandingTests(unittest.TestCase):
    def test_geometry_standing_and_sitting(self):
        standing = classify_posture(skeleton(), [60,0,140,320], DEFAULTS)
        self.assertEqual(standing['posture_label'], 'STANDING')
        self.assertGreater(standing['standing_score'], .9)
        self.assertEqual(standing['standing_coverage'], 1.)
        sitting = classify_posture(skeleton(sitting=True), [60,0,200,240], DEFAULTS)
        self.assertEqual(sitting['posture_label'], 'SITTING')
        self.assertLess(sitting['standing_score'], .6)

    def test_upper_body_is_not_standing(self):
        result = classify_posture(skeleton(legs=False), [60,0,140,190], DEFAULTS)
        self.assertEqual(result['posture_label'], 'UNKNOWN')
        self.assertFalse(result['standing_leg_evidence'])
        self.assertFalse(StandingEvidence(DEFAULTS).update(result, 0)['is_standing'])

    def test_bad_keypoints(self):
        points = skeleton()
        points[0,0] = np.nan
        self.assertEqual(classify_posture(points, [0,0,100,100], DEFAULTS)['posture_label'], 'UNKNOWN')

    def test_single_spike_rejected(self):
        state = StandingEvidence(DEFAULTS)
        self.assertFalse(state.update(posture(), 0)['is_standing'])
        for i in range(1,30):
            self.assertFalse(state.update(posture(.2, 'SITTING'), i*.05)['is_standing'])

    def test_sustained_standing_then_sitting(self):
        state = StandingEvidence(DEFAULTS)
        for i in range(10):
            result = state.update(posture(), i*.05)
        self.assertTrue(result['is_standing'])
        for i in range(10,60):
            result = state.update(posture(.1, 'SITTING'), i*.05)
        self.assertFalse(result['is_standing'])
        self.assertEqual(result['standing_state'], 'S0')

    def test_short_missing_hold_then_expire(self):
        state = StandingEvidence(DEFAULTS)
        for i in range(5):
            state.update(posture(), i*.05)
        missing = posture(None, 'UNKNOWN')
        self.assertTrue(state.update(missing, .25)['is_standing'])
        for i in range(6,20):
            result = state.update(missing, i*.05)
        self.assertFalse(result['is_standing'])

    def test_gap_resets_standing_evidence(self):
        state = StandingEvidence(DEFAULTS)
        for i in range(5):
            state.update(posture(), i*.05)
        result = state.update(posture(), 1.)
        self.assertFalse(result['is_standing'])

    def test_precision_gate(self):
        state = StandingEvidence(DEFAULTS)
        for i in range(20):
            result = state.update(posture(.7), i*.05)
        self.assertEqual(result['standing_state'], 'S2')
        self.assertFalse(result['is_standing'])

    def test_matching_does_not_exclude_neighbor(self):
        people = [dict(body_box=[0,0,100,300]), dict(body_box=[110,0,210,200])]
        poses = [dict(box=[0,0,100,300]),dict(box=[110,0,210,200])]
        matches = match_pose_boxes(people, poses, .3)
        self.assertIs(matches[0][0], poses[0])
        self.assertIs(matches[1][0], poses[1])
        self.assertEqual(len(match_pose_boxes(people, poses[:1], .3)), 1)
        self.assertEqual(match_pose_boxes(people, [dict(box=[500,0,600,200])], .3), {})


if __name__ == '__main__':
    unittest.main()
