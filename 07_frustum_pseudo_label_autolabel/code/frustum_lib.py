# -*- coding: utf-8 -*-
"""
KITTI 视锥法 3D 伪标签生成 —— 几何核心

坐标系约定（KITTI）：
  velodyne : x 前, y 左, z 上
  rect cam : x 右, y 下, z 前
  变换链   : x_cam = R0_rect @ Tr_velo_to_cam @ x_velo
             x_img = P2 @ x_cam   (齐次, 再除以深度)
  label_2  : h,w,l, x,y,z(相机系底面中心), ry(绕相机 Y 轴)
"""
import numpy as np


# ============================ 标定 ============================
class Calib:
    def __init__(self, path):
        d = {}
        for line in open(path):
            line = line.strip()
            if not line or ':' not in line:
                continue
            k, v = line.split(':', 1)
            d[k.strip()] = np.array([float(x) for x in v.split()])
        self.P2 = d['P2'].reshape(3, 4)
        R = np.eye(4); R[:3, :3] = d['R0_rect'].reshape(3, 3)
        self.R0 = R
        T = np.eye(4); T[:3, :4] = d['Tr_velo_to_cam'].reshape(3, 4)
        self.V2C = T

    def velo_to_rect(self, pts):
        """(N,3) velodyne -> (N,3) rect camera"""
        n = pts.shape[0]
        h = np.hstack([pts, np.ones((n, 1))])
        return (self.R0 @ self.V2C @ h.T).T[:, :3]

    def rect_to_velo(self, pts):
        n = pts.shape[0]
        h = np.hstack([pts, np.ones((n, 1))])
        inv = np.linalg.inv(self.R0 @ self.V2C)
        return (inv @ h.T).T[:, :3]

    def rect_to_img(self, pts):
        """(N,3) rect camera -> (N,2) 像素 + (N,) 深度"""
        n = pts.shape[0]
        h = np.hstack([pts, np.ones((n, 1))])
        p = (self.P2 @ h.T).T
        depth = p[:, 2]
        uv = p[:, :2] / np.maximum(depth[:, None], 1e-6)
        return uv, depth

    def velo_to_img(self, pts):
        r = self.velo_to_rect(pts)
        return self.rect_to_img(r)


# ============================ 地面 ============================
def remove_ground_ransac(pts, n_iter=120, thr=0.2, seed=0):
    """RANSAC 拟合地面，返回非地面点掩码。pts 为 velodyne 系 (N,3)"""
    if len(pts) < 20:
        return np.ones(len(pts), bool)
    rng = np.random.RandomState(seed)
    # 只在低处点里找地面，避免把车顶拟合成平面
    cand = np.where(pts[:, 2] < np.percentile(pts[:, 2], 50))[0]
    if len(cand) < 10:
        return np.ones(len(pts), bool)
    best_in, best_plane = -1, None
    for _ in range(n_iter):
        idx = rng.choice(cand, 3, replace=False)
        p0, p1, p2 = pts[idx]
        nrm = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(nrm)
        if nn < 1e-6:
            continue
        nrm = nrm / nn
        if abs(nrm[2]) < 0.8:          # 地面法向应接近竖直
            continue
        d = -nrm @ p0
        dist = np.abs(pts @ nrm + d)
        cnt = int((dist < thr).sum())
        if cnt > best_in:
            best_in, best_plane = cnt, (nrm, d)
    if best_plane is None:
        return pts[:, 2] > (np.percentile(pts[:, 2], 5) + 0.3)
    nrm, d = best_plane
    return np.abs(pts @ nrm + d) >= thr


# ============================ 聚类 ============================
def dbscan(pts, eps=0.6, min_samples=8):
    """返回标签数组，-1 为噪声。用 sklearn，没有则退化为单簇"""
    try:
        from sklearn.cluster import DBSCAN
        return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
    except ImportError:
        return np.zeros(len(pts), int)


def pick_cluster(pts, labels, uv, box2d):
    """选最可能是目标的簇：优先投影落在 2D 框中心附近、且点数多、且距离近"""
    best, best_score = None, -1e9
    cx = (box2d[0] + box2d[2]) / 2
    cy = (box2d[1] + box2d[3]) / 2
    bw = max(box2d[2] - box2d[0], 1)
    bh = max(box2d[3] - box2d[1], 1)
    for l in set(labels.tolist()):
        if l == -1:
            continue
        m = labels == l
        if m.sum() < 8:
            continue
        p, u = pts[m], uv[m]
        # 投影中心与 2D 框中心的归一化偏差
        du = abs(u[:, 0].mean() - cx) / bw
        dv = abs(u[:, 1].mean() - cy) / bh
        dist = np.linalg.norm(p[:, :2].mean(axis=0))
        score = np.log(m.sum() + 1) - 3.0 * (du + dv) - 0.02 * dist
        if score > best_score:
            best_score, best = score, m
    return best


# ============================ 3D 框拟合 ============================
def min_area_rect(xy):
    """BEV 最小面积外接矩形（旋转卡壳）-> (cx, cy, l, w, yaw_velo)"""
    try:
        from scipy.spatial import ConvexHull
        hull = xy[ConvexHull(xy).vertices]
    except Exception:
        hull = xy
    best = None
    n = len(hull)
    for i in range(n):
        p0, p1 = hull[i], hull[(i + 1) % n]
        e = p1 - p0
        norm = np.linalg.norm(e)
        if norm < 1e-6:
            continue
        e = e / norm
        R = np.array([[e[0], e[1]], [-e[1], e[0]]])
        q = xy @ R.T
        mn, mx = q.min(0), q.max(0)
        area = (mx[0] - mn[0]) * (mx[1] - mn[1])
        if best is None or area < best[0]:
            c = R.T @ ((mn + mx) / 2)
            best = (area, c[0], c[1], mx[0] - mn[0], mx[1] - mn[1],
                    np.arctan2(e[1], e[0]))
    if best is None:
        c = xy.mean(0)
        return c[0], c[1], 1.0, 1.0, 0.0
    return best[1], best[2], best[3], best[4], best[5]


def fit_box3d(pts_velo, cls_name='Car'):
    """簇点云 -> KITTI 相机系 3D 框参数 (h, w, l, cx, cy, cz, ry)。pts 为 velodyne 系"""
    if len(pts_velo) < 8:
        return None
    cx, cy, L, W, yaw = min_area_rect(pts_velo[:, :2])
    if L < W:                       # 长边定义为 l
        L, W = W, L
        yaw += np.pi / 2
    z_min, z_max = pts_velo[:, 2].min(), pts_velo[:, 2].max()
    h = float(z_max - z_min)

    # 类别先验：视锥里只看得到目标的一面，尺寸容易偏小，做下限约束
    prior = {'Car': (1.53, 1.63, 3.88), 'Pedestrian': (1.76, 0.66, 0.84),
             'Cyclist': (1.74, 0.60, 1.76)}.get(cls_name, (1.5, 1.6, 3.9))
    h = max(h, prior[0] * 0.6)
    W = max(W, prior[1] * 0.6)
    L = max(L, prior[2] * 0.6)

    return dict(h=float(h), w=float(W), l=float(L),
                cx=float(cx), cy=float(cy), z_bottom=float(z_min), yaw=float(yaw))


def velo_box_to_kitti(box, calib):
    """velodyne 系框 -> KITTI label_2 的 (h,w,l,x,y,z,ry)，x/y/z 为相机系底面中心"""
    bottom = np.array([[box['cx'], box['cy'], box['z_bottom']]])
    cam = calib.velo_to_rect(bottom)[0]
    # velodyne yaw(绕 z, x 前 y 左) -> 相机 ry(绕 y, z 前 x 右)
    ry = -box['yaw'] - np.pi / 2
    ry = (ry + np.pi) % (2 * np.pi) - np.pi
    return (box['h'], box['w'], box['l'],
            float(cam[0]), float(cam[1]), float(cam[2]), float(ry))


# ============================ 主流程 ============================
def frustum_to_box(points, calib, box2d, cls_name='Car',
                   img_wh=None, eps=0.6, min_samples=8):
    """
    points : (N,4) velodyne xyz+intensity
    box2d  : [x1,y1,x2,y2] 像素
    返回   : (h,w,l,x,y,z,ry) 或 None
    """
    xyz = points[:, :3]
    uv, depth = calib.velo_to_img(xyz)
    keep = depth > 0.5
    if img_wh is not None:
        keep &= (uv[:, 0] >= 0) & (uv[:, 0] < img_wh[0]) \
              & (uv[:, 1] >= 0) & (uv[:, 1] < img_wh[1])
    # 落在 2D 框内（略放宽，容忍检测框偏差）
    pad_x = 0.05 * (box2d[2] - box2d[0])
    pad_y = 0.05 * (box2d[3] - box2d[1])
    keep &= (uv[:, 0] >= box2d[0] - pad_x) & (uv[:, 0] <= box2d[2] + pad_x) \
          & (uv[:, 1] >= box2d[1] - pad_y) & (uv[:, 1] <= box2d[3] + pad_y)

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

    box = fit_box3d(fp[m], cls_name)
    if box is None:
        return None
    return velo_box_to_kitti(box, calib)
