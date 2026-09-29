"""Independent per-person state, requiring a continuous path from stable normal."""
from collections import deque
import math
from .head_direction import (axis_angle_diff_deg, circular_angle_diff_deg,
                             signed_angle_change_deg, smooth_direction)


class TurnDetector:
    def __init__(self, person_id, axis, config):
        self.person_id, self.axis, self.c = person_id, axis, config
        self.history = deque(maxlen=config['smooth_window_frames'])
        self.trend = deque(maxlen=config['trend_window_frames'])
        self.state, self.state_start_time = 'UNKNOWN', 0.0
        self.last_time = self.last_raw = self.last_smooth = None
        self.normal_since = self.high_since = None
        self.active_reference_ray = None
        # Freeze the actual normal heading for this episode, including gaps.
        self.normal_reference_angle = None
        self.path = None
        self.event_id = None
        self.events = []
        self.broken = False

    def _state(self, state, t):
        if self.state != state:
            self.state, self.state_start_time = state, t
            if self.path is not None:
                self.path['state_sequence'].append(state)

    def _break(self):
        # Keep displayed state on invalid frames; cancel continuity and duration evidence.
        self.history.clear()
        self.trend.clear()
        self.normal_since = self.high_since = None
        self.path = None
        self.broken = True

    def suspend(self, t):
        """Standing interrupts this observation episode; retain event numbering only."""
        self._break()
        self.last_time = self.last_raw = self.last_smooth = None
        self.event_id = self.active_reference_ray = None
        self.normal_reference_angle = None
        self.normal_sample = None
        self._state('UNKNOWN', t)
        return self.update(t, dict(direction_valid=False, pose_status='EXCLUDED_STANDING'))

    def update(self, t, direction):
        row = dict(smoothed_head_angle=None, smoothed_axis_diff_deg=None,
                   smoothed_reference_diff_deg=None, normal_reference_angle=self.normal_reference_angle,
                   smoothed_direction_x=None, smoothed_direction_y=None,
                   direction_jump_deg=None, raw_direction_jump_deg=None,
                   transition_status=direction['pose_status'], state=self.state,
                   state_start_time=self.state_start_time,
                   active_reference_ray=self.active_reference_ray, event_id=self.event_id)
        if not direction['direction_valid']:
            self._break()
            return row, None
        if self.last_time is not None and t-self.last_time > self.c['max_valid_gap_seconds']+1e-8:
            self._break()
            self.last_raw = self.last_smooth = None
        angle = direction['head_angle']
        raw_jump = circular_angle_diff_deg(angle, self.last_raw) if self.last_raw is not None else 0.0
        row['raw_direction_jump_deg'] = raw_jump
        candidate = list(self.history) + [(direction['head_direction_x'], direction['head_direction_y'])]
        smoothed = smooth_direction(candidate[-self.c['smooth_window_frames']:], self.c['min_smooth_resultant'])
        if smoothed is None:
            self._break()
            row['transition_status'] = 'INVALID_SMOOTH'
            return row, None
        sx, sy, sa = smoothed
        axis_diff = axis_angle_diff_deg(sa, self.axis)
        diff = (circular_angle_diff_deg(sa, self.normal_reference_angle)
                if self.normal_reference_angle is not None else axis_diff)
        jump = circular_angle_diff_deg(sa, self.last_smooth) if self.last_smooth is not None else 0.0
        row.update(smoothed_head_angle=sa, smoothed_axis_diff_deg=axis_diff,
                   smoothed_reference_diff_deg=diff,
                   smoothed_direction_x=sx, smoothed_direction_y=sy, direction_jump_deg=jump)
        if max(raw_jump, jump) > self.c['max_frame_direction_jump_deg']:
            # Reject without moving the last accepted direction. Persistent new pose must
            # pass gap recovery, then reacquire NORMAL; a jump cannot start a turn event.
            self._break()
            row['transition_status'] = 'POSE_JUMP'
            return row, None
        if self.broken:
            if self.event_id is None:
                self._state('UNKNOWN', t)
            self.broken = False
        self.history.append((direction['head_direction_x'], direction['head_direction_y']))
        self.last_raw, self.last_smooth, self.last_time = angle, sa, t
        self.trend.append(diff)
        trend = self.trend[-1]-self.trend[0]
        row['transition_status'] = 'VALID'
        event = None
        normal = diff <= self.c['normal_threshold_deg']
        if normal:
            if self.normal_since is None:
                self.normal_since = t
        else:
            self.normal_since = None
        normal_stable = self.normal_since is not None and t-self.normal_since+1e-8 >= self.c['min_normal_stable_seconds']
        if normal_stable:
            if self.state != 'NORMAL':
                # Only renew at a stable normal heading, never follow head drift.
                self.normal_reference_angle = sa
                diff = 0.0
                self.trend.clear()
                self.trend.append(diff)
            self._state('NORMAL', t)
            self.event_id = None
            self.high_since = None
            self.path = None
            rays = [self.axis % 360, (self.axis+180) % 360]
            self.active_reference_ray = min(rays, key=lambda r: circular_angle_diff_deg(sa, r))
        elif self.state == 'NORMAL' and not normal and trend >= self.c['trend_min_change_deg']:
            # Include the last NORMAL sample, retaining the observed near-axis start.
            self.path = dict(start_time=self.normal_sample[0], start_head_angle=self.normal_sample[1],
                             max_axis_diff=axis_diff, max_reference_diff=diff,
                             samples=1, signed_change=0.0,
                             previous_angle=self.normal_sample[1], state_sequence=['NORMAL'])
            self._state('TURNING', t)
        elif self.state == 'TURNED' and trend <= -self.c['trend_min_change_deg']:
            self._state('RETURNING', t)
        elif self.state == 'RETURNING' and diff >= self.c['turn_threshold_deg'] and trend >= self.c['trend_min_change_deg']:
            self._state('TURNED', t)
        if self.state == 'NORMAL' and normal:
            self.normal_sample = (t, sa)
        if self.state == 'TURNING' and self.path is not None:
            p = self.path
            p['samples'] += 1
            p['max_axis_diff'] = max(p['max_axis_diff'], axis_diff)
            p['max_reference_diff'] = max(p['max_reference_diff'], diff)
            p['signed_change'] += signed_angle_change_deg(sa, p['previous_angle'])
            p['previous_angle'] = sa
            if diff >= self.c['turn_threshold_deg']:
                if self.high_since is None:
                    self.high_since = t
            else:
                self.high_since = None
            if (self.high_since is not None and t-self.high_since+1e-8 >= self.c['min_turned_stable_seconds']
                    and p['samples'] >= self.c['min_transition_samples']):
                self._state('TURNED', t)
                self.event_id = f'P{self.person_id}-E{len(self.events)+1}'
                event = dict(event_id=self.event_id, person_id=self.person_id,
                    start_time=p['start_time'], confirmed_time=t,
                    start_head_angle=p['start_head_angle'], turned_head_angle=sa,
                    max_axis_diff=p['max_axis_diff'], turn_duration=t-p['start_time'],
                    max_reference_diff=p['max_reference_diff'],
                    normal_reference_angle=self.normal_reference_angle,
                    direction_change_deg=p['signed_change'],
                    rotation_direction='CLOCKWISE' if p['signed_change'] >= 0 else 'COUNTERCLOCKWISE',
                    active_reference_ray=self.active_reference_ray,
                    state_sequence=' -> '.join(p['state_sequence']))
                self.events.append(event)
                self.path = None
        row.update(state=self.state, state_start_time=self.state_start_time,
                   normal_reference_angle=self.normal_reference_angle,
                   smoothed_reference_diff_deg=diff,
                   active_reference_ray=self.active_reference_ray, event_id=self.event_id)
        return row, event
