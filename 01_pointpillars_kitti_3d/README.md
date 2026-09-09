# 01 · PointPillars — KITTI 3D 目标检测复现（OpenPCDet）

## 我做了什么
- 在 **Blackwell RTX 5070 (sm_120)** 自建 OpenPCDet：CUDA 12.8 / PyTorch 2.11+cu128 / spconv-cu124 / conda gcc-13。
- 完成 KITTI 3D 数据预处理（kitti_infos / gt_database）、PointPillars **训练 80 epoch** 与评估全流程。
- **环境适配与修复（亮点）**：新版 **numba 0.67 + numpy 2.2** 与 OpenPCDet 自带 `rotate_iou.py`（numba.cuda 旋转 IoU）不兼容致 KITTI eval 崩溃；我将其**改写为 PyTorch 封装的 `boxes_overlap_bev_gpu`**（令 heading = -angle 使几何与原实现完全一致），绕开 numba，精度不变。另修复 torch 2.6+ 加载 checkpoint 需 `weights_only=False`。

## 结果（KITTI val, AP_R40, 3D）
| 类别 | Easy | Moderate | Hard |
|---|---|---|---|
| Car (IoU0.70) | 85.59 | **75.60** | 72.76 |
| Pedestrian (IoU0.50) | 50.79 | **43.75** | 39.12 |
| Cyclist (IoU0.50) | 79.72 | **62.05** | 57.71 |

Car / Cyclist 接近 OpenPCDet 官方基线（≈77 / ≈63）。完整结果见 `results/`。

## 复现
- 训练：`python tools/train.py --cfg_file cfgs/kitti_models/pointpillar.yaml`
- 评估：`python tools/test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt <checkpoint_epoch_80.pth>`

---

## 改进实验：两个假设的受控验证

复现完基线后，针对「与官方基线的差距」做了两轮单变量实验。

### Run 1 — 有效批量假设（被自己的实验否掉）
官方用多卡、有效 batch 更大。用**梯度累积**在单卡上复现有效 batch 32（`patch_gradaccum.py`，改 `train_utils.py`）：

| 类别 (R40, 3D Mod) | 基线 batch 4 | Run 1 有效 batch 32 |
|---|---|---|
| Car | 75.60 | 75.68 |
| Pedestrian | 43.75 | 43.53 |
| Cyclist | 62.05 | 59.92 |

> **结论：有效批量不是差距来源**——三类基本持平，Cyclist 反而降了。假设被自己的实验否掉。

### Run 2 — 分辨率假设（成立）
`pointpillar_vox008.yaml` 与官方 `pointpillar.yaml` **只差 3 行**：
`VOXEL_SIZE: [0.08, 0.08, 4]`、`train: 64000`、`test: 128000`。

| 类别 (R40, 3D) | Easy | **Moderate** | Hard |
|---|---|---|---|
| Car | 86.61 | **76.88** | 73.01 |
| Pedestrian | 56.94 | **49.01** | 43.25 |
| Cyclist | 82.21 | **63.49** | 59.24 |

相对自建基线：**Car +1.28 / Pedestrian +5.26 / Cyclist +1.44**（R40 Moderate），**三类全面提升**。

> ⚠️ **协议说明（重要）**：以上全部是 **AP_R40**；OpenPCDet 官方 README 的表格是 **AP_R11**，两者不可直接比较。
> 本仓库所有对比都在**自建的同协议基线**之间进行。跨协议的「超过官方」不作为结论。

小 pillar 收益在**小目标上最大**（Pedestrian +5.26），符合预期——0.16m pillar 对行人尺度过粗。
