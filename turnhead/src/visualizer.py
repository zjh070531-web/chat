import math
import cv2

COLORS = {'UNKNOWN': (160, 160, 160), 'NORMAL': (70, 210, 70),
          'TURNING': (0, 220, 255), 'TURNED': (40, 40, 255), 'RETURNING': (255, 180, 50)}
SUSPICIOUS_STATES = {'TURNING', 'TURNED', 'RETURNING'}


def number(value, decimals=0):
    return f'{float(value):.{decimals}f}' if value is not None and value != '' else '--'


def draw_text(image, text, x, y, color, font_scale, scale):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale*scale, (0, 0, 0), round(3*scale), cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale*scale, color, round(scale), cv2.LINE_AA)


def draw_frame(frame, people, seconds, axis, config):
    image = frame.copy()
    h, w = image.shape[:2]
    scale = w / min(w, config['output_width'])
    non_standing = [p for p in people if not p.get('is_standing', False)]
    visible = [p for p in non_standing if config.get('draw_all_non_standing', True) or p['state'] in SUSPICIOUS_STATES]
    if config.get('show_standing_exclusions', True):
        visible += [p for p in people if p.get('is_standing', False)]
    for person in visible:
        excluded = person.get('is_standing', False)
        color = (140, 140, 140) if excluded else COLORS[person['state']]
        for key in ('body_box', 'head_box'):
            box = person[key]
            if box is not None:
                x1, y1, x2, y2 = map(round, box)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 1 if key == 'body_box' else 2)
        box = person['head_box'] or person['body_box']
        cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
        if person['head_box'] is not None and not excluded:
            reference = person.get('normal_reference_angle')
            r = math.radians(float(reference) if reference is not None else axis)
            dx, dy = scale*config['axis_half_length']*math.cos(r), scale*config['axis_half_length']*math.sin(r)
            if reference is None:
                cv2.line(image, (round(cx-dx), round(cy-dy)), (round(cx+dx), round(cy+dy)), (255, 220, 100), 2)
            else:
                cv2.arrowedLine(image, (round(cx), round(cy)), (round(cx+dx), round(cy+dy)),
                               (255, 220, 100), 2, tipLength=.3)
            # Current raw estimate remains visible even when a pose jump is rejected.
            raw_angle = person.get('head_angle')
            if raw_angle is not None and person.get('pose_status') not in ('LOW_PROJECTION', 'INVALID_POSE', 'NO_POSE'):
                r = math.radians(float(raw_angle))
                length = scale*config['arrow_length']
                cv2.arrowedLine(image, (round(cx), round(cy)),
                               (round(cx+length*math.cos(r)), round(cy+length*math.sin(r))),
                               (230, 230, 230), 1, tipLength=.2)
            if person['transition_status'] == 'VALID' and person['smoothed_head_angle'] is not None:
                r = math.radians(person['smoothed_head_angle'])
                length = scale*config['arrow_length']
                cv2.arrowedLine(image, (round(cx), round(cy)),
                               (round(cx+length*math.cos(r)), round(cy+length*math.sin(r))), color, 2, tipLength=.25)
        angle = person['smoothed_head_angle']
        diff = person.get('smoothed_reference_diff_deg', person['smoothed_axis_diff_deg'])
        diagnostic = '' if person['transition_status'] == 'VALID' else ' '+person['transition_status']
        text = f"P{person['person_id']} H:{number(person.get('head_angle'))} S:{number(angle)} A:{number(diff)} R:{number(person.get('normal_reference_angle'))} {person['state']}{diagnostic}"
        if excluded:
            text = f"P{person['person_id']} STANDING / TURN SKIPPED"
        posture_text = (f"POST:{person.get('posture_label', '--')} {person.get('standing_state', '--')} "
                        f"ST:{number(person.get('standing_smooth_score'), 2)} C:{number(person.get('standing_coverage'), 2)}")
        x, y = max(0, min(w-100, round(box[0]))), max(round(45*scale), min(h-round(20*scale), round(box[1])-round(23*scale)))
        draw_text(image, text, x, y, color, config['font_scale'], scale)
        draw_text(image, posture_text, x, y+round(17*scale), color, config['font_scale'], scale)
    standing_count = len(people)-len(non_standing)
    cv2.putText(image, f'{seconds:.2f}s axis={axis:g} non-standing={len(non_standing)} standing-excluded={standing_count} | white:raw color:smooth',
                (20, round(25*scale)), cv2.FONT_HERSHEY_SIMPLEX, .6*scale, (255, 255, 255), round(2*scale), cv2.LINE_AA)
    width = min(w, config['output_width'])
    height = round(h*width/w)
    return cv2.resize(image, (width//2*2, height//2*2))
