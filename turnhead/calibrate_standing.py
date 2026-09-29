"""Fit a local standing perspective curve from this project's frame CSV."""
import argparse
import csv
import json
import math
from pathlib import Path
from src.standing_perspective import fit_curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv')
    parser.add_argument('--output', default='standing_perspective.json')
    args = parser.parse_args()
    csv_path = Path(args.csv)
    summary = json.loads((csv_path.parent/'summary.json').read_text(encoding='utf-8'))
    samples, seen = [], set()
    with csv_path.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            raw = row.get('standing_raw_features') or row.get('standing_features')
            if not raw or not row.get('standing_box'):
                continue
            feats, box = json.loads(raw), json.loads(row['standing_box'])
            key = (row['frame_index'],tuple(box))
            if key in seen:
                continue
            seen.add(key)
            if (float(row['standing_coverage']) >= .999 and feats.get('knee_angle',0)>=160
                    and feats.get('torso_verticality',0)>=math.cos(math.radians(30))
                    and all(k in feats for k in ('leg_drop_ratio','body_extension','bbox_aspect'))):
                samples.append(dict(y=box[3], **{k:feats[k] for k in ('leg_drop_ratio','body_extension','bbox_aspect')}))
    curve = fit_curve(samples,summary['source_shape'],Path(summary['video']).name)
    Path(args.output).write_text(json.dumps(curve,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"Standing curve: {curve['status']}, {len(samples)} samples, {len(curve['points'])} knots; saved to {args.output}")


if __name__ == '__main__':
    main()
