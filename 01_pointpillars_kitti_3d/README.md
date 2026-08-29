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
