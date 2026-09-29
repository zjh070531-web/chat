"""Export the same screenshot report for a previously recorded run, without inference."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import yaml

from src.event_report import EventReport


def export_report(run_dir, output=None):
    run_dir = Path(run_dir).resolve()
    output = Path(output).resolve() if output else run_dir/'turn_report.html'
    if output.exists():
        raise FileExistsError(f'Report already exists: {output}')
    with (run_dir/'turn_events.csv').open(encoding='utf-8-sig', newline='') as file:
        events = list(csv.DictReader(file))
    pending = {}
    for event in events:
        for key in ('start_time', 'confirmed_time'):
            event[key] = float(event[key])
        event['person_id'] = int(event['person_id'])
        key = (event['person_id'], round(event['confirmed_time'], 8))
        if key in pending:
            raise ValueError(f'Duplicate event at {key}')
        pending[key] = event
    frames = {}
    with (run_dir/'frames.csv').open(encoding='utf-8-sig', newline='') as file:
        for row in csv.DictReader(file):
            key = (int(row['person_id']), round(float(row['time_seconds']), 8))
            event = pending.pop(key, None)
            if event is not None:
                person = {name: json.loads(row[name]) if row[name] else None for name in ('body_box', 'head_box')}
                frames.setdefault(int(row['frame_index']), []).append((event, person))
            if not pending:
                break
    if pending:
        raise ValueError('Some confirmation frames are missing from frames.csv')
    snapshot = yaml.safe_load((run_dir/'config.snapshot.yaml').read_text(encoding='utf-8'))
    video = (run_dir/snapshot['video']).resolve()
    report = EventReport(output, video.name)
    cap = None
    completed = False
    try:
        if frames:
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                raise RuntimeError(f'Cannot open source video: {video}')
            last_frame = max(frames)
            for index in range(last_frame+1):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f'Source video ended before confirmation frame {last_frame}')
                for event, person in frames.get(index, []):
                    report.add(event, frame, person, index)
                    print(f'Saved {event["event_id"]} at {event["confirmed_time"]:.3f}s', flush=True)
        completed = True
    finally:
        if cap is not None:
            cap.release()
        report.finish(completed)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从已有转头CSV和原视频生成完整截图报告，不重新推理')
    parser.add_argument('run_dir', help='包含frames.csv、turn_events.csv和config.snapshot.yaml的输出目录')
    parser.add_argument('--output', help='另存的HTML路径，默认保存到原输出目录')
    args = parser.parse_args()
    print(export_report(args.run_dir, args.output))
