# 03 · YOLO + ByteTrack + TensorRT — KITTI 2D 检测 / 跟踪 / 部署量化

## 我做了什么
- KITTI → YOLO 格式转换，规范 train/val 切分（5985 / 1496，**独立验证集**，非 train 当 val）。
- 训练 **YOLO11n** 2D 检测；**ByteTrack** 在 KITTI Tracking 序列做多目标跟踪。
- 部署优化：best.pt → ONNX → **TensorRT**，导出并对比 **FP32 / FP16 / INT8**。
- 附加：**YOLO26n vs RT-DETR-l**（CNN vs Transformer 检测器）精度 / 速度 / 体积权衡。

## 结果（KITTI 独立验证集）
- 检测 YOLO11n：**mAP50 85.0% / mAP50-95 57.0%**（Car 95.1 / Ped 77.3 / Cyc 82.6 的 mAP50）。
- TensorRT：**FP16 mAP50 0.839（≈FP32 0.848，近无损），引擎 11→6MB**；INT8 0.826。
- YOLO26n 0.833（2.6M, ~0.5ms） vs RT-DETR-l 0.931（32M, 4.5ms）。

详见 `results/`。权重 (*.pt/*.engine) 与数据集未纳入版本管理。
