import unittest
from src.standing_perspective import fit_curve, curve_factors
from src.standing import classify_posture, DEFAULTS
from test_standing import skeleton


class PerspectiveTests(unittest.TestCase):
    def curve(self):
        samples = [dict(y=300.,leg_drop_ratio=1.5,body_extension=3.,bbox_aspect=3.) for _ in range(120)]
        samples += [dict(y=700.,leg_drop_ratio=1.,body_extension=2.,bbox_aspect=2.) for _ in range(120)]
        return fit_curve(samples,[720,1280],'same.mp4')

    def test_near_compensation_and_upper_unchanged(self):
        curve = self.curve()
        self.assertEqual(curve['status'],'ok')
        self.assertEqual(curve_factors(curve,300,[720,1280],'same.mp4'),(1.,1.,1.))
        self.assertTrue(all(k>=1.4 for k in curve_factors(curve,700,[720,1280],'same.mp4')))

    def test_insufficient_samples(self):
        curve = fit_curve([], [720,1280], 'same.mp4')
        self.assertEqual(curve['status'],'insufficient_data')
        self.assertEqual(curve_factors(curve,700,[720,1280]),(1.,1.,1.))

    def test_different_video_or_resolution_is_not_compensated(self):
        curve = self.curve()
        self.assertEqual(curve_factors(curve,700,[1080,1920],'same.mp4'),(1.,1.,1.))
        self.assertEqual(curve_factors(curve,700,[720,1280],'other.mp4'),(1.,1.,1.))

    def test_compensation_never_decreases_score(self):
        k = skeleton()
        raw = classify_posture(k,[0,0,200,320],DEFAULTS)
        corrected = classify_posture(k,[0,0,200,320],DEFAULTS,(1.5,1.5,1.5))
        self.assertGreaterEqual(corrected['standing_score'],raw['standing_score'])
        self.assertEqual(raw['standing_features'],corrected['standing_raw_features'])
