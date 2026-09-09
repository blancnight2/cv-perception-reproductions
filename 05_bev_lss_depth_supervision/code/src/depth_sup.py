# -*- coding: utf-8 -*-
"""
深度监督消融的核心模块。

实验问题：**BEV 感知到底需要多少激光深度真值？**
  BEVDepth(2022) 证明了「给 LSS 的深度分布加激光监督有用」，
  但没人系统回答「需要多少」——而这直接对应工业界的传感器选型决策
  （32 线 vs 128 线，矿区车对成本尤其敏感）。

做法：把 LiDAR 点随机降采样到 r ∈ {0, 10%, 25%, 50%, 100%}，
      其余完全不变（同一份代码、同一组超参、同一个种子），只改这一个变量。

⚠️ 最容易错的一步：深度真值必须应用与图像相同的数据增强变换。
   LSS 的 get_geometry 里是 orig = post_rot⁻¹(aug - post_tran)，
   所以正向是 aug = post_rot @ orig + post_tran。漏了这步，
   深度真值会与增强后的图像系统性错位，而 loss 照样在降——很难发现。
"""
import numpy as np
import torch
import torch.nn.functional as F


def lidar_to_depth(pts_ego, rot, tran, intrin, post_rot, post_tran,
                   fH, fW, downsample, dbound, ratio=1.0, rng=None):
    """把 ego 系点云投到某个相机，产出特征分辨率的稀疏深度图。

    pts_ego  : (N,3)  ego 系点云（LSS 的 get_lidar_data 直接给 ego 系）
    rot,tran : 相机外参 cam->ego
    intrin   : (3,3) 内参
    post_*   : 数据增强的 2D 仿射（3x3 / 3,）
    ratio    : 稀疏度消融的降采样比例
    返回      : (fH,fW) float32，0 表示该格子无监督
    """
    out = np.zeros((fH, fW), dtype=np.float32)
    if len(pts_ego) == 0:
        return out

    # --- 稀疏度消融：只在这里动手，别的地方一律不变 ---
    if ratio < 1.0:
        n = max(1, int(len(pts_ego) * ratio))
        rng = rng or np.random
        pts_ego = pts_ego[rng.choice(len(pts_ego), n, replace=False)]

    # --- ego -> cam：p_cam = rotᵀ (p_ego - tran) ---
    p = (pts_ego - tran[None, :]) @ rot          # (N,3)，rot 右乘等价于 rotᵀ 左乘
    z = p[:, 2]
    m = z > 1e-3                                  # 相机后方的点丢掉
    if not m.any():
        return out
    p, z = p[m], z[m]

    # --- 投影到原始像素平面 ---
    uvw = p @ intrin.T
    uv = uvw[:, :2] / uvw[:, 2:3]

    # --- 应用数据增强（关键步骤，见文件头说明）---
    uv = uv @ post_rot[:2, :2].T + post_tran[:2][None, :]

    # --- 落到特征网格 ---
    fx = np.floor(uv[:, 0] / downsample).astype(np.int64)
    fy = np.floor(uv[:, 1] / downsample).astype(np.int64)
    m2 = (fx >= 0) & (fx < fW) & (fy >= 0) & (fy < fH) & \
         (z >= dbound[0]) & (z < dbound[1])
    if not m2.any():
        return out
    fx, fy, z = fx[m2], fy[m2], z[m2]

    # --- 同一格子取最近的面（min-pool），与 BEVDepth 一致 ---
    flat = fy * fW + fx
    order = np.argsort(-z)                        # 远的先写，近的后写覆盖
    buf = np.zeros(fH * fW, dtype=np.float32)
    buf[flat[order]] = z[order]
    return buf.reshape(fH, fW)


def depth_loss(pred_dist, depth_gt, dbound, eps=1e-8):
    """预测的深度分布 vs 激光深度真值，只在有监督的像素上算。

    pred_dist : (B,N,D,fH,fW)  已 softmax 的深度分布
    depth_gt  : (B,N,fH,fW)    米制深度，0 = 无监督
    """
    B, N, D, fH, fW = pred_dist.shape
    lo, _, step = dbound
    valid = depth_gt > 0
    if valid.sum() == 0:
        return pred_dist.sum() * 0.0                # 保持计算图，返回 0

    idx = torch.clamp(((depth_gt - lo) / step).long(), 0, D - 1)   # (B,N,fH,fW)
    logp = torch.log(pred_dist.clamp_min(eps))                      # (B,N,D,fH,fW)
    picked = torch.gather(logp, 2, idx.unsqueeze(2)).squeeze(2)     # (B,N,fH,fW)
    return -(picked[valid]).mean()


def depth_metrics(pred_dist, depth_gt, dbound):
    """诊断用：深度分布的期望值 vs 真值，报 MAE 和 bin 命中率。"""
    B, N, D, fH, fW = pred_dist.shape
    lo, _, step = dbound
    valid = depth_gt > 0
    if valid.sum() == 0:
        return float('nan'), float('nan')
    centers = torch.arange(D, device=pred_dist.device, dtype=pred_dist.dtype) * step + lo
    exp_d = (pred_dist * centers.view(1, 1, D, 1, 1)).sum(2)        # (B,N,fH,fW)
    mae = (exp_d[valid] - depth_gt[valid]).abs().mean().item()
    gt_bin = torch.clamp(((depth_gt - lo) / step).long(), 0, D - 1)
    hit = (pred_dist.argmax(2)[valid] == gt_bin[valid]).float().mean().item()
    return mae, hit
