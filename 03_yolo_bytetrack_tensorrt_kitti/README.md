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
<img width="604" height="400" alt="27581c4666f1fbea9731219f2e44b14c" src="https://github.com/user-attachments/assets/836165a2-8973-4084-8844-37f73d12faa5" />

端到端压测脚本已实现，并完成 1501 帧正式实测。
- 脚本：[benchmark_yolo_bytetrack_e2e.py](D:\\GuangFU\\PV Detection-LNN\\tools\\benchmark_yolo_bytetrack_e2e.py)
- 回归测试：[test_benchmark_yolo_bytetrack_e2e.py](D:\\GuangFU\\PV Detection-LNN\\tests\\test_benchmark_yolo_bytetrack_e2e.py)
- 正式报告：summary.json：results\summary.json、逐帧 CSV:results\per_frame_latency.csv
测试口径：排除 100 帧 warmup；每帧从视频解码开始，包含预处理、TensorRT 推理、NMS、ByteTrack 和绘制 ID/框；不含可选的视频编码写盘。

<img width="904" height="390" alt="image" src="https://github.com/user-attachments/assets/19bbe361-6f0f-4cb8-92f5-8fab2feb0542" />
阶段均值：解码 0.806 ms，检测+NMS+ByteTrack 3.978 ms，绘制 0.901 ms。
复跑命令：
C:\Python313\python.exe tools\benchmark_yolo_bytetrack_e2e.py `
  --output-dir "runs\detect\benchmarks\e2e_fp16_bytetrack_demo1"
若要额外保存跟踪视频：
C:\Python313\python.exe tools\benchmark_yolo_bytetrack_e2e.py --save
--save 的视频编码耗时会单独记录在 CSV 的 write_ms_excluded，不会污染实时端到端 p50/p95。三项统计/预热回归测试均通过。

