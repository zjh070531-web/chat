"""Independent same-camera standing perspective curve, fitted from visible legs."""
import numpy as np

KEYS = ('leg_drop_ratio', 'body_extension', 'bbox_aspect')


def fit_curve(samples, shape, source_video, min_samples=200, windows=20, max_k=2., y0_fraction=.625):
    curve = dict(status='insufficient_data', samples=len(samples), shape=list(shape),
                 source_video=source_video, y0=shape[0]*y0_fraction, points=[])
    if len(samples) < min_samples:
        return curve
    ordered = sorted(samples, key=lambda s: s['y'])
    width = max(40, len(samples)//windows)
    step = max(1, width//2)
    points = []
    for start in range(0,len(samples),step):
        chunk = ordered[start:start+width]
        points.append(dict(y=float(np.median([s['y'] for s in chunk])),
                           **{key:float(np.median([s[key] for s in chunk])) for key in KEYS}))
        if start+width >= len(samples):
            break
    reference = points[0]
    factors = []
    previous = {key:1. for key in KEYS}
    for p in points:
        values = {key:max(previous[key], float(np.clip(reference[key]/max(p[key],1e-6),1,max_k))) for key in KEYS}
        previous = values
        factors.append(dict(y=p['y'], **values))
    # Merge duplicate positions; numpy interpolation requires increasing knots.
    unique = {}
    for p in factors:
        unique[p['y']] = p
    curve.update(status='ok', points=list(unique.values()))
    return curve


def curve_factors(curve, y, shape, source_video=None):
    if (not curve or curve['status'] != 'ok' or list(shape) != curve['shape']
            or (source_video is not None and source_video != curve['source_video'])
            or y < curve['y0']):
        return (1.,1.,1.)
    points = curve['points']
    return tuple(float(np.interp(y,[p['y'] for p in points],[p[key] for p in points])) for key in KEYS)
