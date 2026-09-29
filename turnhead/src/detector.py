"""复用 BPJDet 权重加载、预处理、NMS 及作者的人体-头部关联。"""
import os
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from .upstream import load_functions


class BodyHeadDetector:
    def __init__(self, repo, weights, image_size=768, confidence=.35, nms_iou=.75,
                 match_inside_ratio=.6):
        sys.path.insert(0, str(repo))
        from models.experimental import attempt_load
        from utils.general import non_max_suppression, scale_coords, check_img_size
        from utils.augmentations import letterbox
        original_load = torch.load
        def load_official_checkpoint(*args, **kwargs):
            # 官方 BPJDet 保存完整 nn.Module，PyTorch 2.6+ 需要显式声明。
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        with patch('torch.load', load_official_checkpoint):
            self.model = attempt_load(str(weights), map_location='cpu').eval()
        for module in self.model.modules():
            if isinstance(module, torch.nn.Upsample) and not hasattr(module, 'recompute_scale_factor'):
                module.recompute_scale_factor = None
        self.stride = int(self.model.stride.max())
        self.image_size = check_img_size(image_size, s=self.stride)
        self.nms, self.letterbox = non_max_suppression, letterbox
        _, self.associate = load_functions(repo / 'val.py', ['cal_inside_iou', 'post_process_batch'],
            {'np': np, 'torch': torch, 'Path': Path, 'osp': os.path, 'scale_coords': scale_coords})
        self.confidence, self.nms_iou = confidence, nms_iou
        self.data = dict(dataset='CrowdHuman', part_type='head', num_offsets=2,
                         match_iou_thres=match_inside_ratio)

    def to(self, device):
        self.model.to(device)
        return self

    @torch.inference_mode()
    def predict(self, frame):
        resized = self.letterbox(frame, self.image_size, stride=self.stride, auto=True)[0]
        tensor = torch.from_numpy(np.ascontiguousarray(resized[:, :, ::-1].transpose(2, 0, 1)))
        tensor = tensor.to(next(self.model.parameters()).device).float()[None] / 255
        predictions = self.model(tensor, augment=False)[0]
        body = self.nms(predictions.clone(), self.confidence, self.nms_iou, classes=[0], num_offsets=2)
        head = self.nms(predictions.clone(), self.confidence, self.nms_iou, classes=[1], num_offsets=2)
        boxes, points, scores, *_ = self.associate(self.data, tensor, [], [[frame.shape[:2]]], body, head)
        results = []
        for box, point, score in zip(boxes, points, scores):
            head_score = float(point[0][2])
            results.append(dict(body_box=box.tolist(), body_confidence=float(score),
                head_box=point[0][3:].tolist() if head_score > 0 else None,
                head_confidence=head_score, pose=None))
        return results
