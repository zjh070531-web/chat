"""Validate saved evidence and render readable previews without rerunning models."""
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from src.pipeline import read_config
from src.visualizer import draw_frame

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'outputs/first10s_axis30'


def main():
    config = read_config(OUT/'config.snapshot.yaml')
    with (OUT/'frames.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    with (OUT/'turn_events.csv').open(encoding='utf-8-sig', newline='') as f:
        events = list(csv.DictReader(f))
    frames = {}
    for row in rows:
        for key in ('body_box', 'head_box'):
            row[key] = json.loads(row[key]) if row[key] else None
        for key in ('head_angle', 'smoothed_head_angle', 'smoothed_axis_diff_deg', 'standing_smooth_score', 'standing_coverage'):
            row[key] = float(row[key]) if row.get(key) else None
        row['is_standing'] = row.get('is_standing') == 'True'
        row['person_id'] = int(row['person_id'])
        frames.setdefault(int(row['frame_index']), []).append(row)
        for key in ('axis_diff_deg', 'smoothed_axis_diff_deg'):
            if row[key] not in (None, ''):
                assert 0 <= float(row[key]) <= 90
    assert len(frames) == 200
    assert len({e['event_id'] for e in events}) == len(events)
    checks = []
    for event in events:
        ident = int(event['person_id'])
        start, confirmed = float(event['start_time']), float(event['confirmed_time'])
        path = [r for r in rows if r['person_id'] == ident and start <= float(r['time_seconds']) <= confirmed]
        assert path[0]['state'] == 'NORMAL' and path[-1]['state'] == 'TURNED'
        assert all(r['transition_status'] == 'VALID' for r in path)
        assert all(float(r['direction_jump_deg']) <= config['temporal']['max_frame_direction_jump_deg'] for r in path)
        assert all(float(r['raw_direction_jump_deg']) <= config['temporal']['max_frame_direction_jump_deg'] for r in path)
        assert max(float(b['time_seconds'])-float(a['time_seconds']) for a,b in zip(path,path[1:])) <= config['temporal']['max_valid_gap_seconds']+1e-8
        high_start = confirmed
        for row in reversed(path):
            if row['smoothed_axis_diff_deg'] < config['temporal']['turn_threshold_deg']:
                break
            high_start = float(row['time_seconds'])
        assert confirmed-high_start+1e-8 >= config['temporal']['min_turned_stable_seconds']
        checks.append(dict(event_id=event['event_id'], continuous_valid_samples=len(path),
                           high_stable_seconds=confirmed-high_start,
                           max_raw_jump=max(float(r['raw_direction_jump_deg']) for r in path)))
    cap = cv2.VideoCapture(str(config['video']))
    writer = None
    panels = {e['event_id']: [] for e in events}
    # Event before/during/after samples with original image crops for manual QA.
    targets = {}
    for event in events:
        for t in (float(event['start_time'])-.4, float(event['start_time']),
                  float(event['confirmed_time'])-.25, float(event['confirmed_time']),
                  float(event['confirmed_time'])+.5, float(event['confirmed_time'])+1):
            targets.setdefault(round(t*20), []).append(event)
    try:
        for index in range(200):
            ok, frame = cap.read()
            assert ok
            people = frames[index]
            annotated = draw_frame(frame, people, index/20, config['reference']['axis_degrees'], config['display'])
            if writer is None:
                writer = cv2.VideoWriter(str(OUT/'annotated.readable.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20, (annotated.shape[1], annotated.shape[0]))
                assert writer.isOpened()
            writer.write(annotated)
            if index in (0, 40, 100, 199):
                cv2.imencode('.jpg', annotated)[1].tofile(str(OUT/f'preview_{index:04d}.jpg'))
            for event in targets.get(index, []):
                person = next(p for p in people if p['person_id'] == int(event['person_id']))
                box = person['head_box']
                cx, cy = round((box[0]+box[2])/2), round((box[1]+box[3])/2)
                crop = frame[max(0,cy-120):cy+180,max(0,cx-140):cx+140]
                tile = cv2.resize(crop, (280,300))
                tile = cv2.copyMakeBorder(tile, 0, 50, 0, 0, cv2.BORDER_CONSTANT)
                angle, diff = person['smoothed_head_angle'], person['smoothed_axis_diff_deg']
                cv2.putText(tile, f'{index/20:.2f}s P{person["person_id"]} {person["state"]}', (5,320), cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                if angle is not None:
                    cv2.putText(tile, f'H:{angle:.1f} A:{diff:.1f} {person["transition_status"]}',(5,340),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                panels[event['event_id']].append(tile)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    (OUT/'annotated.readable.mp4').replace(OUT/'annotated.mp4')
    for event_id, tiles in panels.items():
        sheet = np.vstack((np.hstack(tiles[:3]), np.hstack(tiles[3:])))
        cv2.imencode('.jpg', sheet)[1].tofile(str(OUT/f'{event_id}_contact.jpg'))
    verify = cv2.VideoCapture(str(OUT/'annotated.mp4'))
    decoded = 0
    while verify.read()[0]:
        decoded += 1
    metadata = dict(frames=decoded, fps=verify.get(cv2.CAP_PROP_FPS), width=verify.get(cv2.CAP_PROP_FRAME_WIDTH), height=verify.get(cv2.CAP_PROP_FRAME_HEIGHT))
    verify.release()
    assert decoded == 200
    audit = dict(frame_rows=len(rows), frame_count=len(frames), output_video=metadata,
                 event_path_checks=checks, all_axis_diffs_in_range=True)
    (OUT/'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
