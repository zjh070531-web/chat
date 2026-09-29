import argparse
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import time
import cv2
import torch
import yaml
from .detector import BodyHeadDetector
from .head_pose import HeadPoseDetector
from .tracker import PersonTracker
from .head_direction import direction_from_pose, axis_angle_diff_deg
from .turn_detector import TurnDetector
from .visualizer import draw_frame
from .event_report import EventReport
from .standing import DEFAULTS, StandingFilter, validate_standing_config

ROOT = Path(__file__).resolve().parents[1]
FRAME_FIELDS = ['frame_index', 'time_seconds', 'person_id', 'body_box', 'head_box',
    'body_confidence', 'head_confidence', 'yaw', 'pitch', 'roll', 'rotation_matrix',
    'head_direction_x', 'head_direction_y', 'head_angle', 'projection_length', 'direction_valid',
    'pose_status', 'reference_axis_deg', 'axis_diff_deg', 'smoothed_head_angle',
    'smoothed_axis_diff_deg', 'normal_reference_angle', 'smoothed_reference_diff_deg',
    'direction_jump_deg', 'raw_direction_jump_deg',
    'transition_status', 'state', 'state_start_time', 'active_reference_ray', 'event_id',
    'posture_label', 'is_standing', 'standing_score', 'standing_smooth_score',
    'standing_state', 'standing_coverage', 'standing_leg_evidence', 'standing_evidence_valid',
    'standing_match_iou', 'standing_box', 'standing_features', 'standing_raw_features',
    'standing_perspective_factors', 'turn_excluded_reason']
EVENT_FIELDS = ['event_id', 'person_id', 'start_time', 'confirmed_time', 'start_head_angle',
    'turned_head_angle', 'max_axis_diff', 'turn_duration', 'direction_change_deg',
    'rotation_direction', 'active_reference_ray', 'normal_reference_angle',
    'max_reference_diff', 'state_sequence']


def read_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    for section in ('bpjdet', 'head_pose'):
        for key in ('repo', 'weights'):
            config[section][key] = (path.parent/config[section][key]).resolve()
    for key in ('video', 'output_dir'):
        if config.get(key):
            config[key] = (path.parent/config[key]).resolve()
    # Older snapshots predate this filter; replay them without changing their behavior.
    config['standing_filter'] = {**DEFAULTS, 'enabled': False, **config.get('standing_filter', {})}
    config['standing_filter']['weights'] = (path.parent/config['standing_filter']['weights']).resolve()
    if config['standing_filter']['perspective_calibration']:
        config['standing_filter']['perspective_calibration'] = (path.parent/config['standing_filter']['perspective_calibration']).resolve()
    validate_standing_config(config['standing_filter'])
    c = config['temporal']
    if not 0 <= c['normal_threshold_deg'] < c['turn_threshold_deg'] <= 90:
        raise ValueError('thresholds must satisfy 0 <= normal < turn <= 90')
    for key in ('smooth_window_frames', 'trend_window_frames', 'min_transition_samples'):
        if isinstance(c[key], bool) or not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if c['trend_window_frames'] < 2 or c['min_transition_samples'] < 3:
        raise ValueError('trend window >= 2; transition samples >= 3')
    for key in ('min_normal_stable_seconds', 'min_turned_stable_seconds',
                'max_valid_gap_seconds', 'max_frame_direction_jump_deg', 'trend_min_change_deg'):
        if not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not 0 < c['min_smooth_resultant'] <= 1:
        raise ValueError('min_smooth_resultant must be in (0,1]')
    if not math.isfinite(config['reference']['axis_degrees']):
        raise ValueError('reference axis must be finite')
    if not 0 < config['direction']['min_projection_length'] <= 1:
        raise ValueError('projection threshold must be in (0,1]')
    return config


def config_snapshot(config, output):
    """Snapshots are portable: all asset paths are relative to the snapshot file."""
    snapshot = json.loads(json.dumps(config, default=str))
    for section in ('bpjdet', 'head_pose'):
        for key in ('repo', 'weights'):
            snapshot[section][key] = Path(os.path.relpath(config[section][key], output)).as_posix()
    for key in ('video', 'output_dir'):
        if config.get(key):
            snapshot[key] = Path(os.path.relpath(config[key], output)).as_posix()
    snapshot['standing_filter']['weights'] = Path(os.path.relpath(config['standing_filter']['weights'], output)).as_posix()
    if config['standing_filter']['perspective_calibration']:
        snapshot['standing_filter']['perspective_calibration'] = Path(os.path.relpath(config['standing_filter']['perspective_calibration'], output)).as_posix()
    return snapshot


class Pipeline:
    def __init__(self, config):
        self.c = config
        if str(config['device']).startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; use the project CUDA .venv or --device cpu')
        torch.set_num_threads(config['torch_threads'])
        self.detector = BodyHeadDetector(**config['bpjdet']).to(config['device'])
        hp = dict(config['head_pose'])
        self.batch_size = hp.pop('batch_size')
        self.pose = HeadPoseDetector(**hp).to(config['device'])
        self.tracker = PersonTracker(**config['tracking'])
        self.standing = StandingFilter(config['standing_filter'], config['device'],config.get('video')) if config['standing_filter']['enabled'] else None
        self.turns = {}

    def process(self, frame, t):
        people = self.tracker.update(self.detector.predict(frame), t)
        if self.standing is not None:
            self.standing.update(people, frame, t, self.tracker.tracks)
        else:
            for person in people:
                person.update(is_standing=False, posture_label='DISABLED', turn_excluded_reason=None)
        indices = [i for i, p in enumerate(people) if p['head_box'] is not None and not p['is_standing']]
        poses = self.pose.predict_batch([(frame, people[i]['head_box']) for i in indices], self.batch_size)
        for i, pose in zip(indices, poses):
            people[i]['pose'] = pose
        # Expired IDs cannot resume an old direction history.
        self.turns = {ident: turn for ident, turn in self.turns.items() if ident in self.tracker.tracks}
        events = []
        for person in people:
            ident = person['person_id']
            turn = self.turns.setdefault(ident, TurnDetector(ident, self.c['reference']['axis_degrees'], self.c['temporal']))
            direction = direction_from_pose(person['pose'], self.c['direction'])
            if person['is_standing']:
                direction['pose_status'] = 'EXCLUDED_STANDING'
                temporal, event = turn.suspend(t)
            else:
                temporal, event = turn.update(t, direction)
            person.update(direction)
            person.update(temporal)
            person['reference_axis_deg'] = self.c['reference']['axis_degrees']
            person['axis_diff_deg'] = axis_angle_diff_deg(person['head_angle'], person['reference_axis_deg']) if person['head_angle'] is not None else None
            if event:
                events.append(event)
        return people, events


def run(config, video, output, seconds=None):
    if not Path(video).is_file():
        raise FileNotFoundError(video)
    if seconds is not None and (not math.isfinite(seconds) or seconds <= 0):
        raise ValueError('seconds must be positive')
    config = {**config, 'video': Path(video).resolve()}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    targets = ['annotated.mp4', 'frames.csv', 'turn_events.csv', 'turn_report.html', 'summary.json', 'config.snapshot.yaml']
    if any((output/name).exists() for name in targets):
        raise FileExistsError(f'Output already exists: {output}; choose a new --output directory')
    cap = cv2.VideoCapture(str(video))
    writer = None
    event_report = None
    completed = False
    started = time.perf_counter()
    status_counts, state_counts = Counter(), Counter()
    track_ids, rows, event_count, count = set(), 0, 0, 0
    standing_rows, standing_ids = 0, set()
    max_diff, min_diff = 0., 90.
    try:
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open video: {video}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError('Invalid video FPS')
        limit = math.ceil(seconds*fps) if seconds is not None else math.inf
        reported_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        reported_frames = int(reported_frames) if math.isfinite(reported_frames) and reported_frames > 0 else None
        progress_target = str(limit) if seconds is not None else str(reported_frames or 'EOF')
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError('No readable video frames')
        source_shape = list(frame.shape[:2])
        scope = f'first {seconds:g}s ({limit} frames)' if seconds is not None else 'full video (until EOF)'
        print(f'Input: {frame.shape[1]}x{frame.shape[0]}, {fps:g} FPS, {scope}', flush=True)
        snapshot = config_snapshot(config, output)
        (output/'config.snapshot.yaml').write_text(yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False), encoding='utf-8')
        pipeline = Pipeline(config)
        event_report = EventReport(output/'turn_report.html', Path(video).name)
        with (output/'frames.csv').open('w', newline='', encoding='utf-8-sig') as frame_file, (output/'turn_events.csv').open('w', newline='', encoding='utf-8-sig') as event_file:
            frame_table = csv.DictWriter(frame_file, FRAME_FIELDS)
            event_table = csv.DictWriter(event_file, EVENT_FIELDS)
            frame_table.writeheader()
            event_table.writeheader()
            while ok and count < limit:
                t = count/fps
                people, events = pipeline.process(frame, t)
                annotated = draw_frame(frame, people, t, config['reference']['axis_degrees'], config['display'])
                if writer is None:
                    writer = cv2.VideoWriter(str(output/'annotated.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (annotated.shape[1], annotated.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError('Cannot create video writer')
                writer.write(annotated)
                if count in (0, round(2*fps), round(5*fps), limit-1):
                    cv2.imencode('.jpg', annotated)[1].tofile(str(output/f'preview_{count:04d}.jpg'))
                for p in people:
                    row = {k: p.get(k) for k in FRAME_FIELDS}
                    row.update(frame_index=count, time_seconds=t)
                    pose = p['pose'] or {}
                    row.update({k: pose.get(k) for k in ('yaw', 'pitch', 'roll', 'rotation_matrix')})
                    for k in ('body_box', 'head_box', 'rotation_matrix', 'standing_box', 'standing_features',
                              'standing_raw_features', 'standing_perspective_factors'):
                        row[k] = json.dumps(row[k]) if row[k] is not None else None
                    frame_table.writerow(row)
                    rows += 1
                    track_ids.add(p['person_id'])
                    status_counts[p['transition_status']] += 1
                    state_counts[p['state']] += 1
                    if p['is_standing']:
                        standing_rows += 1
                        standing_ids.add(p['person_id'])
                    if p['smoothed_axis_diff_deg'] is not None:
                        max_diff = max(max_diff, p['smoothed_axis_diff_deg'])
                        min_diff = min(min_diff, p['smoothed_axis_diff_deg'])
                for event in events:
                    person = next(p for p in people if p['person_id'] == event['person_id'])
                    event_report.add(event, frame, person, count)
                    event_table.writerow(event)
                    event_count += 1
                count += 1
                if count % max(1, round(fps)) == 0:
                    frame_file.flush()
                    event_file.flush()
                    print(f'{count}/{progress_target} frames, {rows} person rows, {event_count} events, elapsed {time.perf_counter()-started:.1f}s', flush=True)
                if count < limit:
                    ok, frame = cap.read()
        # Also retain the last actual processed frame when duration is unlimited/shorter than requested.
        cv2.imencode('.jpg', annotated)[1].tofile(str(output/f'preview_{count-1:04d}.jpg'))
        summary = dict(video=Path(os.path.relpath(video, ROOT)).as_posix(), requested_seconds=seconds, frames=count,
                       processing_scope='full_video' if seconds is None else 'duration_limit',
                       source_reported_frames=reported_frames,
                       video_fps=fps, processed_seconds=count/fps, source_shape=source_shape,
                       person_rows=rows, person_ids=sorted(track_ids), event_count=event_count,
                       event_report='turn_report.html',
                       standing_filter_enabled=config['standing_filter']['enabled'],
                       standing_excluded_rows=standing_rows, standing_person_ids=sorted(standing_ids),
                       head_pose_person_calls=pipeline.pose.person_calls,
                       transition_status_counts=dict(status_counts), state_counts=dict(state_counts),
                       smoothed_axis_diff_range=[min_diff, max_diff] if rows else None,
                       elapsed_seconds=time.perf_counter()-started, device=config['device'],
                       reference_axis_deg=config['reference']['axis_degrees'])
        (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        completed = True
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if event_report is not None:
            event_report.finish(completed)


def cli():
    parser = argparse.ArgumentParser(description='BPJDet + SemiUHPE fixed-axis head turns')
    parser.add_argument('--config', default=str(ROOT/'config.yaml'))
    parser.add_argument('--video')
    parser.add_argument('--seconds', type=float, default=None,
                        help='Process only the first N seconds; omit to process the full video')
    parser.add_argument('--output')
    parser.add_argument('--device')
    args = parser.parse_args()
    config = read_config(args.config)
    if args.device:
        config['device'] = args.device
    run(config, args.video or config['video'], args.output or config['output_dir']/time.strftime('%Y%m%d_%H%M%S'), args.seconds)
