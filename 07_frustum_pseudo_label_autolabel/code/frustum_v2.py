# -*- coding: utf-8 -*-
"""
视锥法 v2 —— 针对"只看得到一个面"的两处改进

v1 的问题：
  1) 最小外接矩形贴的是可见点云而非车辆真实轮廓 -> 尺寸系统性偏小 (l -0.71m)
  2) 只看到单面时矩形主方向 != 车辆朝向 -> 朝向误差均值 34.7°

改进：
  A. L-shape fitting（Zhang et al. closeness 准则）替代最小面积矩形，
     搜索使点云最贴合矩形两条边的朝向 —— 专为单面/L 形可见设计。
  B. 类别先验尺寸 + 沿视线方向的中心补偿：
     可见面只是车的一侧，几何中心要沿观测方向后推，补出被遮挡的半个车身。
"""
import numpy as np
from frustum_lib import (Calib, remove_ground_ransac, dbscan, pick_cluster,
                         velo_box_to_kitti)

# KITTI 训练集类别平均尺寸 (h, w, l)
PRIOR = {
    'Car':        (1.53, 1.63, 3.88),
    'Pedestrian': (1.76, 0.66, 0.84),
    'Cyclist':    (1.74, 0.60, 1.76),
}


def lshape_yaw(xy, n_ang=90, d0=0.05):
    """
    L-shape fitting，closeness 准则。
    对每个候选朝向，算每点到"最近矩形边"的距离，距离越小说明点越贴边，
    单面/L 形点云会在真实朝向上取得最高得分。
    返回 [0, pi/2) 内的最优角度。
    """
    best_a, best_s = 0.0, -np.inf
    for k in range(n_ang):
        a = k * (np.pi / 2) / n_ang
        e1 = np.array([np.cos(a), np.sin(a)])
        e2 = np.array([-np.sin(a), np.cos(a)])
        c1, c2 = xy @ e1, xy @ e2
        d1 = np.minimum(c1 - c1.min(), c1.max() - c1)
        d2 = np.minimum(c2 - c2.min(), c2.max() - c2)
        d = np.maximum(np.minimum(d1, d2), d0)
        s = float((1.0 / d).sum())
        if s > best_s:
            best_s, best_a = s, a
    return best_a


def fit_box3d_v2(pts_velo, cls_name='Car', use_prior=True):
    """簇点云 -> velodyne 系框字典"""
    if len(pts_velo) < 8:
        return None
    xy = pts_velo[:, :2]

    a = lshape_yaw(xy)
    ph0, pw0, pl0 = PRIOR.get(cls_name, PRIOR['Car'])

    # 90 度歧义消解：只看得到单面时，"跨度更大"并不代表那是车长方向。
    # 物理判据 —— 主可见面的跨度更接近车宽(≈1.6m)说明看到的是车头/车尾，
    # 此时车长轴垂直于该面；更接近车长(≈3.9m)则看到的是车侧，长轴沿该面。
    cands = []
    for yaw in (a, a + np.pi / 2):
        e1 = np.array([np.cos(yaw), np.sin(yaw)])
        e2 = np.array([-np.sin(yaw), np.cos(yaw)])
        c1, c2 = xy @ e1, xy @ e2
        cands.append((float(c1.max() - c1.min()), yaw, e1, e2, c1, c2))

    span_max = max(c[0] for c in cands)
    if abs(span_max - pw0) < abs(span_max - pl0):
        # 主可见面是车头/车尾 -> 车长轴取跨度较小的那个方向
        pick = min(cands, key=lambda t: t[0])
    else:
        # 主可见面是车侧 -> 车长轴沿跨度较大的方向
        pick = max(cands, key=lambda t: t[0])
    span, yaw, e1, e2, c1, c2 = pick

    obs_l = float(c1.max() - c1.min())
    obs_w = float(c2.max() - c2.min())
    z_min, z_max = float(pts_velo[:, 2].min()), float(pts_velo[:, 2].max())
    obs_h = z_max - z_min

    ph, pw, pl = ph0, pw0, pl0
    if use_prior:
        # 观测尺寸只是下界（看不见的部分没有点），用先验补齐
        L = max(obs_l, pl)
        W = max(obs_w, pw)
        H = max(obs_h, ph * 0.85)
    else:
        L, W, H = max(obs_l, 0.5), max(obs_w, 0.5), max(obs_h, 0.5)

    # 中心补偿：观测到的是靠近传感器那一侧，
    # 沿"远离原点"的方向把中心推到补齐后的框中心
    cen1 = (c1.min() + c1.max()) / 2
    cen2 = (c2.min() + c2.max()) / 2
    obs_center = e1 * cen1 + e2 * cen2

    view = obs_center / max(np.linalg.norm(obs_center), 1e-6)   # 视线方向(向外)
    # 沿 e1 / e2 各自补偿未观测到的一半
    for e, obs, full in ((e1, obs_l, L), (e2, obs_w, W)):
        gap = (full - obs) / 2.0
        if gap > 0:
            obs_center = obs_center + e * gap * np.sign(e @ view)

    return dict(h=H, w=W, l=L,
                cx=float(obs_center[0]), cy=float(obs_center[1]),
                z_bottom=z_min, yaw=float(yaw))


def frustum_to_box_v2(points, calib, box2d, cls_name='Car',
                      img_wh=None, eps=0.6, min_samples=8, use_prior=True):
    xyz = points[:, :3]
    uv, depth = calib.velo_to_img(xyz)
    keep = depth > 0.5
    if img_wh is not None:
        keep &= (uv[:, 0] >= 0) & (uv[:, 0] < img_wh[0]) \
              & (uv[:, 1] >= 0) & (uv[:, 1] < img_wh[1])
    px = 0.05 * (box2d[2] - box2d[0]); py = 0.05 * (box2d[3] - box2d[1])
    keep &= (uv[:, 0] >= box2d[0] - px) & (uv[:, 0] <= box2d[2] + px) \
          & (uv[:, 1] >= box2d[1] - py) & (uv[:, 1] <= box2d[3] + py)

    idx = np.where(keep)[0]
    if len(idx) < 10:
        return None
    fp, fuv = xyz[idx], uv[idx]

    ng = remove_ground_ransac(fp)
    if ng.sum() < 10:
        return None
    fp, fuv = fp[ng], fuv[ng]

    labels = dbscan(fp, eps=eps, min_samples=min_samples)
    m = pick_cluster(fp, labels, fuv, box2d)
    if m is None or m.sum() < 8:
        return None

    box = fit_box3d_v2(fp[m], cls_name, use_prior)
    if box is None:
        return None
    return velo_box_to_kitti(box, calib)
