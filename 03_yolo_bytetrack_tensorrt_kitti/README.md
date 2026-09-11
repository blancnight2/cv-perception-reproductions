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


针对KITTI数据集的优化


① 数据与协议:先把 KITTI 标签转成 YOLO 格式,筛选/映射到 Car / Pedestrian / Cyclist 三类;关键是我用了独立验证集(train/val = 5985/1496),不是拿训练集当验证——保证 mAP 真实、可复现。

② 输入尺寸 + 矩形训练:KITTI 图像是宽幅(约 1242×375,接近 3.3:1),用方形 640 会 padding 掉大量像素、还压缩了远处小目标。所以我开 rect=True 矩形训练、把 imgsz 加到 960,保住远处行人/骑行者的细节。

③ 场景相关的数据增强:驾驶场景重力方向固定,不能上下翻,所以 flipud=0;左右翻合理保留。避免破坏几何结构的增强。

④ 模型与部署:从 YOLO11n 基线(mAP50 85.0%) 换到 YOLO11s + 960 + rect,mAP50 提到 90.39% / mAP50-95 65.10%;再 ONNX→TensorRT 做 FP32/FP16/INT8 部署对比

<img width="985" height="220" alt="image" src="https://github.com/user-attachments/assets/c5993c77-ece6-4e13-9c52-075642183e9e" />

① 基线 YOLO11n@640 是 85.0;

② 模型不变,只把输入改成 960+矩形训练 → 88.8,mAP50-95 从 57 涨到 63 —— 这是最大的单一提升,因为 KITTI 宽幅、远处目标小,高分辨率对小目标和定位帮助最大(所以严格的 mAP50-95 涨得更多);

③ 在此基础上只把 n 换成 s → 90.4,模型容量再加约 1.6 个点,是次要贡献;

④ 我还试过更强的 scale 增强 + 提高 cls 权重,结果掉到 87.3,说明过强增强不划算,就没采用。
所以主力是输入端调优,模型升级是锦上添花

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


---


---

## 多目标跟踪评测（KITTI Tracking，21 段训练序列）

此前这个项目只有「ByteTrack 已跑通」，**没有任何跟踪指标**——而 ByteTrack 的全部价值就在
ID 一致性，不测等于没做。这里补上完整评测。

**协议**（对齐 KITTI 官方跟踪 benchmark 的关键几条）：

- 逐类评测，匹配 IoU 阈值 0.5；21 段 training 序列（testing 无公开真值）
- `Van` 对 Car、`Person_sitting` 对 Pedestrian、`DontCare` 一律记为 **ignore**
- **难度过滤**：框高 < 25 px、遮挡等级 > 2、截断 > 0.5 的真值降级为 ignore
- ignore 区域的压制用 **IoA（检测框被覆盖的比例）而非 IoU**——KITTI 的 `DontCare`
  常是覆盖一片远处目标的大框，小检测框落在里面 IoU 很低，用 IoU 压不住

### 结果（YOLO11n + ByteTrack，conf 0.25，imgsz 640）

| 类别 | MOTA | MOTP(IoU) | IDF1 | IDSW | MT | ML |
|---|---|---|---|---|---|---|
| Car | **68.58%** | 0.849 | **81.28%** | 214 | 367 | 28 |
| Pedestrian | 40.32% | 0.747 | 48.68% | 456 | 36 | 41 |
| Cyclist | 26.66% | 0.750 | 49.89% | 27 | 10 | 8 |

### 误差归因：跟踪不是瓶颈，检测召回才是

| 类别 | 真值数 | 漏检 FN | 误检 FP | **ID 切换** |
|---|---|---|---|---|
| Car | 19551 | 18.9% | 11.4% | **1.1%** |
| Pedestrian | 11070 | **42.6%** | 13.0% | **4.1%** |
| Cyclist | 1819 | **49.9%** | 22.0% | **1.5%** |

> **ID 切换在三类里都只占 1–4%，而漏检占 19–50%。**
> 也就是说 **ByteTrack 本身工作正常，MOTA 低几乎全部来自检测器没看见目标**——
> 近一半的行人和骑行者从未被检出。
> **结论：继续调跟踪器（缓冲帧数、匹配阈值）收益极小，该投入的是检测召回**
> ——更高输入分辨率、更强主干、或针对小目标的训练策略。
> 这与本项目另一处发现互相印证：960 + 矩形训练把 mAP50 从 85.0 拉到 88.8，
> 正是因为 KITTI 宽幅图像下远处小目标吃亏。

### 一个额外的归因实验

怀疑「把骑车人误判成行人」是行人 FP 的主因，于是加了一档
`--neutral-person`（Cyclist ↔ Pedestrian 互记 ignore）单独量化：

| 类别 | 官方口径 | 人/骑车人互记 ignore | 差 |
|---|---|---|---|
| Pedestrian | 40.32% | 40.50% | +0.18 |
| Cyclist | 26.66% | 27.10% | +0.44 |

> **假设被否掉**：类别混淆只值 0.2–0.4 个 MOTA 点。
> 在 3 段子集上看曾经很像主因（行人预测里 53 个压在 Cyclist 真值上、只有 36 个压在行人上），
> 但全量 21 段推翻了它——**那个子集里 Cyclist 是 Pedestrian 的 7 倍，严重不具代表性**。
> 教训：小子集上的归因结论必须回到全量验证。

### 评测过程中发现并修掉的四个坑

1. **`persist=False` 导致跟踪器每帧重建**。ultralytics 把图片文件夹当成一堆互相独立的图，
   ID 每帧从 1 重排，于是每个目标每帧都被记一次 ID 切换——
   **IDSW 虚高到 9484、Car MOTA 虚低到 35.97%**。改 `persist=True` 后 IDSW 降到 214。
   代价是跨序列必须手动 `trackers[i].reset()`，否则上一段的 ID 会串到下一段。
2. **难度过滤缺失**：不过滤小目标/重遮挡，FN 与 FP 同时虚高。
3. **ignore 压制误用 IoU**：改成 IoA 后，Car 的 FP 从 88 降到 37（3 段子集上）。
4. **`motmetrics` 的 `motp` 在 `generate_overall` 下返回 nan**：改为直接从事件表里
   取 `Type=='MATCH'` 的距离均值，自己换算回 IoU。

### ⚠️ 一个必须说明的局限

**KITTI 的目标检测数据集与跟踪数据集来自同一批原始采集序列。**
本项目的检测器是在 KITTI 检测划分上训练的，因此**很可能见过这些跟踪序列中的部分帧**，
上面的数字应视为**乐观估计**。要得到无污染的结论，需按原始采集片段（drive）重新划分
训练/评测，或直接在 KITTI tracking 的 test 集上提交官方评测。

### 复现

```bash
python code/eval_track.py                      # 官方口径，21 段
python code/eval_track.py --neutral-person     # 归因口径
python code/eval_track.py --seqs 0000,0003     # 只跑几段
```

需 KITTI tracking 真值 `data_tracking_label_2`（官网单独下载），
解压后把 `label_02/` 放到 `data_tracking_image_2/` 下。

### 检验归因：只换检测器，跟踪器一行不动

上面断言「瓶颈在检测召回而非跟踪关联」。直接检验——把检测器从 YOLO11n@640（mAP50 85.0）
换成 YOLO11s@960+rect（mAP50 90.4），**ByteTrack 配置完全不变**：

| 类别 | MOTA | IDF1 | IDSW | 漏检 FN | MT | ML |
|---|---|---|---|---|---|---|
| Car | 68.58 → **71.90** (+3.3) | 81.28 → **82.72** | 214 → **169** | 3703 → **3211** | 367 → **395** | 28 → **22** |
| Pedestrian | 40.32 → **47.80** (+7.5) | 48.68 → **56.59** | 456 → **311** | 4717 → **4199** | 36 → **52** | 41 → **30** |
| Cyclist | 26.66 → **32.71** (+6.1) | 49.89 → **60.23** | 27 → 23 | 907 → **784** | 10 → **16** | 8 → **5** |

> **归因成立**：漏检全线下降，MOTA 随之上升，跟踪器未做任何改动。
>
> **附带发现（比主结果更有意思）：ID 切换也大幅下降**——Car −21%、Pedestrian −32%，
> 而跟踪器一行没改。说明 **IDSW 不是纯粹的跟踪器属性：漏检会把一条轨迹打断成几段，
> 从而制造出「切换」**。所以看到 IDSW 高时，先查检测召回，再怀疑关联算法。

### 关于与官方排行榜的可比性

- **本项目的数字全部来自 21 段 training 序列**，而排行榜是 29 段 test 序列（真值不公开）。
  training 通常更容易，且本项目检测器与这些序列**数据同源**（见上文局限），故应视为乐观估计。
- **排行榜主排序指标是 HOTA**，本项目算的是 CLEAR MOT（MOTA/MOTP）与 IDF1——
  所用的 `motmetrics` 不实现 HOTA，需改用官方 `TrackEval`。
- **KITTI 跟踪官方只评 Car 与 Pedestrian**，Cyclist 无官方参照，本项目的该项仅供自比。
- **无法提交 test 评测**：KITTI 明文规定「只允许具有显著新颖性、并将发表于同行评议
  会议或期刊的方法提交；对已有算法的小改动和学生研究项目不允许」。本项目是现成
  检测器 + 现成跟踪器的组合，按此规定不具备提交资格。
- 同为单目 2D 的参照：**CenterTrack（DLA-34）KITTI test Car MOTA 89.44 / val 88.7**。
  本项目 71.90（train split、YOLO11s）明显更低——差距主要在检测主干规模与输入分辨率。
