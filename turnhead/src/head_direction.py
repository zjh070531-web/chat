"""Image coordinates: right=0, down=90, left=180, up=270 degrees."""
import math
import numpy as np


def axis_angle_diff_deg(head_angle, axis_angle):
    return abs(((head_angle - axis_angle + 90.0) % 180.0) - 90.0)


def signed_angle_change_deg(current, previous):
    return (current - previous + 180.0) % 360.0 - 180.0


def circular_angle_diff_deg(a, b):
    return abs(signed_angle_change_deg(a, b))


def angle_from_vector(dx, dy):
    return math.degrees(math.atan2(dy, dx)) % 360.0


def smooth_direction(vectors, min_resultant=0.2):
    dx, dy = np.mean(vectors, axis=0)
    length = math.hypot(dx, dy)
    if length < min_resultant:
        return None
    return float(dx/length), float(dy/length), angle_from_vector(dx, dy)


def direction_from_pose(pose, config):
    result = dict(head_direction_x=None, head_direction_y=None, head_angle=None,
                  projection_length=None, direction_valid=False, pose_status='NO_POSE')
    if not pose:
        return result
    matrix = np.asarray(pose['rotation_matrix'], dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix.T @ matrix, np.eye(3), atol=.01)
            or abs(np.linalg.det(matrix)-1) > .01
            or not np.isfinite([pose[k] for k in ('yaw', 'pitch', 'roll')]).all()):
        result['pose_status'] = 'INVALID_POSE'
        return result
    # Exactly the old verified convention. No transpose, sign change or z division.
    vx, vy, _ = matrix @ np.array([0., 0., 1.])
    length = math.hypot(vx, vy)
    result['projection_length'] = length
    if length > 1e-10:
        result.update(head_direction_x=float(vx/length), head_direction_y=float(vy/length),
                      head_angle=angle_from_vector(vx, vy))
    if abs(pose['pitch']) >= config['max_abs_pitch_deg']:
        result['pose_status'] = 'OUT_OF_SCOPE'
    elif length < config['min_projection_length']:
        result['pose_status'] = 'LOW_PROJECTION'
    else:
        result.update(direction_valid=True, pose_status='VALID')
    return result
