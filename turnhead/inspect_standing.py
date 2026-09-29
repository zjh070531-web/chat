"""Inspect matched standing keypoints on frame 40 of the default test clip."""
import json
import csv
import math
from pathlib import Path
import cv2
from src.pipeline import read_config, ROOT
from src.standing import StandingFilter, match_pose_boxes

config = read_config(ROOT/'config.yaml')
cap = cv2.VideoCapture(str(config['video']))
for _ in range(41):
    ok, frame = cap.read()
cap.release()
assert ok
with (ROOT/'outputs/standing_filter_first10s_v2/frames.csv').open(encoding='utf-8-sig') as f:
    people = [r for r in csv.DictReader(f) if r['frame_index']=='40']
for p in people:
    p['body_box'] = json.loads(p['body_box'])
poses = StandingFilter(config['standing_filter'],config['device']).predict(frame)
matches = match_pose_boxes(people,poses,config['standing_filter']['match_iou'])
for i,p in enumerate(people):
    if p['person_id'] not in ('1','2','10','27') or i not in matches:
        continue
    pose,_ = matches[i]
    x1,y1,x2,y2 = map(round,p['body_box'])
    crop = frame[max(0,y1-20):min(frame.shape[0],y2+30),max(0,x1-20):min(frame.shape[1],x2+20)].copy()
    for k,(x,y,conf) in enumerate(pose['keypoints']):
        if conf < .35:
            continue
        center = (round(x)-max(0,x1-20),round(y)-max(0,y1-20))
        cv2.circle(crop,center,3,(0,0,255),-1)
        cv2.putText(crop,f'{k}:{conf:.2f}',center,cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,255),1)
    cv2.imencode('.jpg',crop)[1].tofile(str(ROOT/f'outputs/standing_filter_first10s_v2/P{p["person_id"]}_skeleton.jpg'))
    print(p['person_id'],[(j,round(float(pose['keypoints'][j,2]),3)) for j in (5,6,11,12,13,14,15,16)])
