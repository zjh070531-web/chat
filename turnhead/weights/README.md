本目录包含从旧工程复制的原始模型权重：

- `ch_head_s_1536_e150_best_mMR.pt`：BPJDet 人体与头部检测。
- `DAD-WildHead-EffNetV2-S-best.pth`：SemiUHPE 头姿估计。
- `yolo26s-pose.pt`：从站立发布包复制，仅用于站立过滤的17点姿态提取。

根目录 `config.yaml` 使用项目内相对路径加载。无需旧工程、下载或训练。
