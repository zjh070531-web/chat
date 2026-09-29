"""Local standing geometry and temporal evidence. No imports from the reference app.

Adapted principles: six visible geometric features, mean+EMA, three-state
Viterbi and a standing precision gate. Uses the existing BPJDet person IDs.
"""
from collections import deque
import math
import os
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from .tracker import iou
from .standing_perspective import curve_factors, KEYS

FEATURES = {
    'torso_verticality': (3., .55, .93),
    'leg_drop_ratio': (3., .95, 1.35),
    'knee_angle': (2., 120., 160.),
    'body_extension': (2., 2., 2.70),
    'bbox_aspect': (1.5, 1.30, 2.10),
    'hip_height_ratio': (1., .30, .50),
}
LEG_FEATURES = ('leg_drop_ratio', 'knee_angle', 'body_extension')
DEFAULTS = dict(enabled=True, weights='weights/yolo26s-pose.pt', image_size=640,
    confidence=.25, nms_iou=.7, keypoint_confidence=.35, match_iou=.3,
    min_coverage=.3, require_leg_evidence=True, enter_score=.6, release_score=.45,
    min_standing_score=.8, smooth_window_frames=15, ema_alpha=.35,
    viterbi_window_frames=38, viterbi_stay=.92, viterbi_sigma=.15,
    min_valid_observations=2, evidence_grace_seconds=.6, max_sample_gap_seconds=.2,
    perspective_calibration=None)


def validate_standing_config(config):
    if not isinstance(config['enabled'], bool) or not isinstance(config['require_leg_evidence'], bool):
        raise ValueError('standing_filter switches must be boolean')
    for key in ('confidence', 'nms_iou', 'keypoint_confidence', 'match_iou',
                'min_coverage', 'ema_alpha', 'min_standing_score'):
        if not math.isfinite(config[key]) or not 0 < config[key] <= 1:
            raise ValueError(f'standing_filter.{key} must be in (0,1]')
    if not 0 < config['release_score'] < config['enter_score'] <= config['min_standing_score']:
        raise ValueError('standing scores must satisfy 0 < release < enter <= min_standing_score')
    if not .5 <= config['viterbi_stay'] < 1:
        raise ValueError('viterbi_stay must be in [0.5,1)')
    for key in ('viterbi_sigma', 'max_sample_gap_seconds'):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'standing_filter.{key} must be positive')
    if not math.isfinite(config['evidence_grace_seconds']) or config['evidence_grace_seconds'] < 0:
        raise ValueError('evidence grace must be nonnegative')
    for key in ('image_size', 'smooth_window_frames', 'viterbi_window_frames', 'min_valid_observations'):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f'standing_filter.{key} must be a positive integer')
    if config['viterbi_window_frames'] < 3 or config['min_valid_observations'] < 2:
        raise ValueError('standing Viterbi window >=3 and valid observations >=2')


def classify_posture(keypoints, box, config, factors=(1.,1.,1.)):
    k = np.asarray(keypoints, dtype=float)
    empty = dict(posture_label='UNKNOWN', standing_score=None, standing_coverage=0.,
                 standing_leg_evidence=False, standing_features={})
    if k.shape != (17, 3) or not np.isfinite(k).all():
        return empty
    visible = k[:, 2] >= config['keypoint_confidence']

    def middle(a, b):
        points = [k[i, :2] for i in (a, b) if visible[i]]
        return np.mean(points, axis=0) if points else None

    shoulder, hip, ankle = middle(5, 6), middle(11, 12), middle(15, 16)
    face = [k[i, :2] for i in range(5) if visible[i]]
    head = min(face, key=lambda p: p[1]) if face else shoulder
    width, height = max(1., box[2]-box[0]), max(1., box[3]-box[1])
    feats = dict(bbox_aspect=height/width)
    torso_length = None
    if shoulder is not None and hip is not None:
        torso = shoulder-hip
        torso_length = float(np.linalg.norm(torso))
        if torso_length > 1:
            feats['torso_verticality'] = float(-torso[1]/torso_length)
        else:
            torso_length = None
    if torso_length is not None and hip is not None and ankle is not None:
        feats['leg_drop_ratio'] = float((ankle[1]-hip[1])/torso_length)
    angles = []
    for a,b,c in ((11,13,15), (12,14,16)):
        if all(visible[i] for i in (a,b,c)):
            u,v = k[a,:2]-k[b,:2], k[c,:2]-k[b,:2]
            lengths = np.linalg.norm(u)*np.linalg.norm(v)
            if lengths > 1e-6:
                angles.append(math.degrees(math.acos(float(np.clip(np.dot(u,v)/lengths, -1, 1)))))
    if angles:
        feats['knee_angle'] = float(np.mean(angles))
    if torso_length is not None and ankle is not None and head is not None:
        feats['body_extension'] = float((ankle[1]-head[1])/torso_length)
    if hip is not None and ankle is not None:
        feats['hip_height_ratio'] = float((ankle[1]-hip[1])/height)
    raw_features = dict(feats)
    for key,factor in zip(KEYS,factors):
        if key in feats:
            feats[key] *= factor
    total = sum(weight for weight, _, _ in FEATURES.values())
    available = sum(FEATURES[key][0] for key in feats)
    coverage = available/total
    score = sum(FEATURES[key][0]*float(np.clip((value-FEATURES[key][1])/(FEATURES[key][2]-FEATURES[key][1]), 0, 1))
                for key,value in feats.items())/available
    legs = any(key in feats for key in LEG_FEATURES)
    torso_cos = feats.get('torso_verticality')
    if coverage < config['min_coverage'] or (config['require_leg_evidence'] and not legs):
        label = 'UNKNOWN'
    elif ((torso_cos is not None and torso_cos < .3 and feats['bbox_aspect'] < 1.6)
          or (torso_cos is None and feats['bbox_aspect'] < .95)):
        label = 'LYING'
    else:
        label = 'STANDING' if score >= config['enter_score'] else 'SITTING'
    return dict(posture_label=label, standing_score=score, standing_coverage=coverage,
                standing_leg_evidence=legs, standing_features=feats,
                standing_raw_features=raw_features, standing_perspective_factors=factors)


class StandingEvidence:
    def __init__(self, config):
        self.c = config
        self.scores = deque(maxlen=config['smooth_window_frames'])
        self.observations = deque(maxlen=config['viterbi_window_frames'])
        self.ema = self.smooth = None
        self.state = self.valid_count = 0
        self.last_time = self.last_evidence_time = None
        stay = config['viterbi_stay']
        self.transitions = np.log([[stay, 1-stay, 1.], [(1-stay)/2, stay, (1-stay)/2], [1., 1-stay, stay]])
        self.transitions[0,2] = self.transitions[2,0] = -np.inf
        mid = (config['enter_score']+config['release_score'])/2
        self.means = np.array([2*config['release_score']-mid, mid, 2*config['enter_score']-mid])

    def update(self, posture, t):
        if self.last_time is not None and t-self.last_time > self.c['max_sample_gap_seconds']+1e-8:
            self.__init__(self.c)
        self.last_time = t
        evidence = (posture['standing_score'] is not None and posture['posture_label'] != 'UNKNOWN')
        if evidence:
            score = posture['standing_score']
            self.scores.append(score)
            self.ema = score if self.ema is None else self.c['ema_alpha']*score+(1-self.c['ema_alpha'])*self.ema
            self.smooth = (float(np.mean(self.scores))+self.ema)/2
            observation = self.smooth
            self.valid_count += 1
            self.last_evidence_time = t
        elif (self.last_evidence_time is not None
              and t-self.last_evidence_time > self.c['evidence_grace_seconds']+1e-8):
            observation = self.c['release_score']-.05
        else:
            observation = None
        self.observations.append(observation)
        prior = np.full(3, math.log(1-self.c['viterbi_stay']))
        prior[self.state] = 0.
        for value in self.observations:
            emission = 0. if value is None else -.5*((value-self.means)/self.c['viterbi_sigma'])**2
            prior = np.max(prior[:,None]+self.transitions, axis=0)+emission
        self.state = int(np.argmax(prior))
        if self.state == 2 and self.valid_count < self.c['min_valid_observations']:
            self.state = 1
        in_grace = self.last_evidence_time is not None and t-self.last_evidence_time <= self.c['evidence_grace_seconds']+1e-8
        standing = self.state == 2 and self.smooth is not None and self.smooth >= self.c['min_standing_score'] and in_grace
        return dict(is_standing=bool(standing), standing_smooth_score=self.smooth,
                    standing_state=f'S{self.state}', standing_evidence_valid=evidence)


def match_pose_boxes(people, poses, threshold):
    """One-to-one maximum-IoU matching; invalid pairs cannot steal a valid match."""
    if not people or not poses:
        return {}
    overlaps = np.array([[iou(p['body_box'], q['box']) for q in poses] for p in people])
    scores = np.where(overlaps >= threshold, overlaps, 0.)
    indices, matches = linear_sum_assignment(-scores)
    return {int(i): (poses[j], float(overlaps[i,j])) for i,j in zip(indices,matches) if overlaps[i,j] >= threshold}


class StandingFilter:
    def __init__(self, config, device, source_video=None):
        self.c, self.device, self.tracks = config, device, {}
        self.source_video = Path(source_video).name if source_video is not None else None
        self.curve = None
        if config['perspective_calibration']:
            self.curve = json.loads(Path(config['perspective_calibration']).read_text(encoding='utf-8'))
        weights = Path(config['weights'])
        if not weights.is_file():
            raise FileNotFoundError(f'Missing local standing weights: {weights.name}')
        root = Path(__file__).resolve().parents[1]
        cache = root/'.cache/ultralytics'
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault('YOLO_CONFIG_DIR', str(cache))
        os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
        from ultralytics import YOLO, settings
        settings.update({'sync': False, 'datasets_dir': 'datasets', 'weights_dir': 'weights', 'runs_dir': 'runs'})
        self.model = YOLO(str(weights), task='pose')

    def predict(self, frame):
        result = self.model.predict(frame, imgsz=self.c['image_size'], conf=self.c['confidence'],
                    iou=self.c['nms_iou'], device=self.device, verbose=False)[0]
        if result.keypoints is None:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        points = result.keypoints.data.cpu().numpy()
        return [dict(box=box.tolist(), keypoints=k) for box,k in zip(boxes,points)]

    def update(self, people, frame, t, active_ids):
        matches = match_pose_boxes(people, self.predict(frame), self.c['match_iou']) if people else {}
        self.tracks = {ident: state for ident,state in self.tracks.items() if ident in active_ids}
        for i,person in enumerate(people):
            matched = matches.get(i)
            if matched:
                pose, overlap = matched
                factors = curve_factors(self.curve,pose['box'][3],frame.shape[:2],self.source_video)
                posture = classify_posture(pose['keypoints'], pose['box'], self.c, factors)
                person['standing_box'] = pose['box']
                person['standing_match_iou'] = overlap
            else:
                posture = dict(posture_label='UNKNOWN', standing_score=None, standing_coverage=0.,
                               standing_leg_evidence=False, standing_features={})
                person['standing_box'] = None
                person['standing_match_iou'] = None
            person.update(posture)
            ident = person['person_id']
            if ident not in self.tracks:
                self.tracks[ident] = StandingEvidence(self.c)
            person.update(self.tracks[ident].update(posture, t))
            person['turn_excluded_reason'] = 'STANDING' if person['is_standing'] else None
