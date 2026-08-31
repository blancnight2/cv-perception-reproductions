# 03 · YOLO + ByteTrack + TensorRT — KITTI 2D 检测 / 跟踪 / 部署量化

## 我做了什么
- KITTI → YOLO 格式转换，规范 train/val 切分（5985 / 1496，**独立验证集**，非 train 当 val）。
- 训练 **YOLO11n** 2D 检测；**ByteTrack** 在 KITTI Tracking 序列做多目标跟踪。
- 部署优化：best.pt → ONNX → **TensorRT**，导出并对比 **FP32 / FP16 / INT8**，engine profile , p50/p95/p99压测。
- 附加：**YOLO26n vs RT-DETR-l**（CNN vs Transformer 检测器）精度 / 速度 / 体积权衡。

## 结果（KITTI 独立验证集）
- 检测 YOLO11n：**mAP50 83.9% / mAP50-95 57.1%**（Car 95.1 / Ped 77.3 / Cyc 82.6 的 mAP50）。
- TensorRT：**FP16 mAP50 0.839（≈FP32 0.848，近无损），引擎 11→6MB**；INT8 0.826。
- YOLO26n 0.833（2.6M, ~0.5ms） vs RT-DETR-l 0.931（32M, 4.5ms）。

详见 `results/`。权重 (*.pt/*.engine) 与数据集未纳入版本管理。
<img width="604" height="400" alt="27581c4666f1fbea9731219f2e44b14c" src="https://github.com/user-attachments/assets/836165a2-8973-4084-8844-37f73d12faa5" />

针对KITTI数据集优化后

训练调优口径是 7481 张 KITTI 训练集按 8:2 划分，独立验证集 1496 张。e3 YOLO11s 在 960、rect、batch 8 的组合下，第 46 轮达到 mAP50 90.39%、mAP50-95 65.10%、Precision 89.55%、Recall 82.66%。
部署口径单独说明：静态 640 FP16 TRT 11.2 engine 的纯 engine mean/p95 为 0.408/0.409 ms；demo1.mp4 1501 帧、排除 100 帧预热、含解码到绘制的端到端 p50/p95 为 5.707/6.502 ms，平均处理能力 175.50 FPS。

Engine profile：

<img width="871" height="480" alt="image" src="https://github.com/user-attachments/assets/b2cbf2a2-b2d3-412e-8085-697e3f837fdc" />


用 TensorRT 11.2 对 FP16 ONNX 构建了静态 batch=1、640×640 的 engine，并以 trtexec 预热 500 次、压测 30 秒。纯 GPU engine 推理平均延迟 0.408 ms，p95 为 0.409 ms，吞吐 2448 qps。这个数字不包含视频解码、预处理、主机到显存传输、后处理/NMS、ByteTrack 和渲染，因此不能直接等同于端到端 FPS；端到端性能需要再单独压测。

端到端压测，1501 帧正式实测

- 脚本：code\benchmark_yolo_bytetrack_e2e.py
- 回归测试：code\test_benchmark_yolo_bytetrack_e2e.py
- 正式报告：summary.json：results\summary.json、逐帧 CSV:results\per_frame_latency.csv
测试口径：排除 100 帧 warmup；每帧从视频解码开始，包含预处理、TensorRT 推理、NMS、ByteTrack 和绘制 ID/框；不含可选的视频编码写盘。

<img width="904" height="390" alt="image" src="https://github.com/user-attachments/assets/19bbe361-6f0f-4cb8-92f5-8fab2feb0542" />

阶段均值：解码 0.806 ms，检测+NMS+ByteTrack 3.978 ms，绘制 0.901 ms。

复跑命令：
C:\Python313\python.exe tools\benchmark_yolo_bytetrack_e2e.py `--output-dir "runs\detect\benchmarks\e2e_fp16_bytetrack_demo1"

若要额外保存跟踪视频：
C:\Python313\python.exe tools\benchmark_yolo_bytetrack_e2e.py --save--save 

视频编码耗时会单独记录在 CSV 的 write_ms_excluded，不会污染实时端到端 p50/p95。三项统计/预热回归测试均通过。

