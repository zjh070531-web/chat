"""SemiUHPE 头部朝向；不是眼球视线追踪。"""
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from scipy.spatial.transform import Rotation
from PIL import Image
from .upstream import load_functions


class HeadPoseDetector:
    def __init__(self, repo, weights, min_head_pixels=12, edges_scale=-0.05):
        self.min_head_pixels, self.edges_scale = min_head_pixels, edges_scale
        # 调用上游构建函数，跳过 ImageNet 初始化下载，严格加载完整任务权重。
        factories = SimpleNamespace(efficientnet_v2_s=lambda **kw: models.efficientnet_v2_s(weights=None))
        factory, = load_functions(repo / 'src/networks.py', ['get_EfficientNet_V2'],
                                  {'models': factories, 'nn': nn})
        self.model = factory(SimpleNamespace(num_classes=9), model_name='S')
        checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
        # 与 video.py -> agent.net -> load_ckpt 的 student 权重键一致。
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        del checkpoint
        self.model.eval()
        self.batch_calls = 0
        self.person_calls = 0
        self.to_rotation, = load_functions(repo / 'src/fisher/fisher_utils.py',
            ['batch_torch_A_to_R'], {'torch': torch}, cpu_rotation=True)
        self.limit_angle, = load_functions(repo / 'src/utils.py', ['limit_angle'], {})
        self.transform = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def to(self, device):
        self.model.to(device)
        return self

    def _prepare(self, frame, box):
        """沿用单头预处理；过小/空裁剪不进入批次。"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        if min(x2-x1, y2-y1) < self.min_head_pixels:
            return None
        cx, cy = (x1+x2)/2, (y1+y2)/2
        radius = (0.5-self.edges_scale)*max(x2-x1, y2-y1)
        x1, x2 = max(0, int(cx-radius)), min(w-1, int(cx+radius))
        y1, y2 = max(0, int(cy-radius)), min(h-1, int(cy+radius))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        image = Image.fromarray(crop[:, :, ::-1]).resize((224, 224))
        return self.transform(image), [x1, y1, x2, y2]

    @torch.inference_mode()
    def predict_batch(self, items, batch_size=8):
        """items为(frame, head_box)列表，结果与输入逐项对应，无效头返回None。"""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('head_batch_size须为正整数')
        results = [None] * len(items)
        pending = []

        def flush():
            # 每批一次GPU前向、一次CPU旋转矩阵转换；不累计整段视频。
            tensors = torch.stack([x[1] for x in pending]).to(next(self.model.parameters()).device)
            matrices = self.to_rotation(self.model(tensors)).numpy()
            self.batch_calls += 1
            self.person_calls += len(pending)
            for (index, _, crop_box), matrix in zip(pending, matrices, strict=True):
                angles = Rotation.from_matrix(matrix.T).as_euler('xyz', degrees=True)
                roll, pitch, yaw = map(self.limit_angle, [angles[2], angles[0]-180, angles[1]])
                if not np.isfinite([yaw, pitch, roll]).all():
                    raise RuntimeError('头部朝向推理出现非有限值')
                results[index] = dict(yaw=float(yaw), pitch=float(pitch), roll=float(roll),
                    rotation_matrix=matrix.tolist(), crop_box=crop_box)
            pending.clear()

        for index, (frame, box) in enumerate(items):
            prepared = self._prepare(frame, box)
            if prepared is not None:
                pending.append((index, *prepared))
                if len(pending) == batch_size:
                    flush()
        if pending:
            flush()
        return results

    def predict(self, frame, box):
        # 保留校准脚本和旧单人入口。
        return self.predict_batch([(frame, box)], batch_size=1)[0]
