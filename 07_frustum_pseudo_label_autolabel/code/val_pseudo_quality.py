# -*- coding: utf-8 -*-
"""
离线评估 3D 伪标签质量（对照真值 3D 框），不训练。

为什么要有这一步：训一次 PointPillars 要 2.5 小时。三个 conf 阈值就是 7.5 小时。
用 BEV IoU 的精确率/召回率先离线选出最好的那个，只训一次。

⚠️ 判据取向：伪标签场景下**精确率比召回率重要**。
   漏检只是少几个训练目标；误检会变成**错误的训练目标**，直接毒化模型。

用法：
    python val_pseudo_quality.py ~/autolabel/pseudo_gt_train            # 基线
    python val_pseudo_quality.py ~/autolabel/pseudo_yoloe-conf0.15_train
"""
import sys, os, glob, math
import numpy as np
from shapely.geometry import Polygon

KITTI = os.path.expanduser('~/OpenPCDet/data/kitti/training/label_2')
IDS = os.path.expanduser('~/OpenPCDet/data/kitti/ImageSets/train.txt')


def read_kitti(path, cls='Car'):
    """返回 [(h, w, l, x, y, z, ry), ...]（相机坐标）"""
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path):
        v = line.split()
        if len(v) < 15 or v[0] != cls:
            continue
        h, w, l = float(v[8]), float(v[9]), float(v[10])
        x, y, z = float(v[11]), float(v[12]), float(v[13])
        ry = float(v[14])
        if h <= 0 or w <= 0 or l <= 0:
            continue
        out.append((h, w, l, x, y, z, ry))
    return out


def bev_poly(b):
    """KITTI 相机坐标下的 BEV footprint（x-z 平面）"""
    _, w, l, x, _, z, ry = b
    c, s = math.cos(ry), math.sin(ry)
    xc = [l / 2, l / 2, -l / 2, -l / 2]
    zc = [w / 2, -w / 2, -w / 2, w / 2]
    pts = [(x + c * xc[i] + s * zc[i], z - s * xc[i] + c * zc[i]) for i in range(4)]
    p = Polygon(pts)
    return p if p.is_valid else p.buffer(0)


def iou(a, b):
    pa, pb = bev_poly(a), bev_poly(b)
    if pa.area <= 0 or pb.area <= 0:
        return 0.0
    inter = pa.intersection(pb).area
    u = pa.area + pb.area - inter
    return inter / u if u > 0 else 0.0


PRED_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/autolabel/pseudo_gt_train')
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0

ids = [l.strip() for l in open(IDS) if l.strip()]
if LIMIT:
    ids = ids[:LIMIT]

TP5 = TP7 = NP = NG = 0
ious = []
frames_no_pred = 0

for fid in ids:
    gt = read_kitti(os.path.join(KITTI, fid + '.txt'))
    pr = read_kitti(os.path.join(PRED_DIR, fid + '.txt'))
    NG += len(gt)
    NP += len(pr)
    if not pr:
        frames_no_pred += 1
    if not gt or not pr:
        continue

    M = np.zeros((len(pr), len(gt)))
    for i, p in enumerate(pr):
        for j, g in enumerate(gt):
            M[i, j] = iou(p, g)

    # 贪心一对一匹配
    usedg = set()
    for i in np.argsort(-M.max(axis=1)):
        j = int(np.argmax([M[i, k] if k not in usedg else -1 for k in range(len(gt))]))
        v = M[i, j]
        if j in usedg or v <= 0:
            continue
        usedg.add(j)
        ious.append(v)
        if v >= 0.5:
            TP5 += 1
        if v >= 0.7:
            TP7 += 1

ious = np.array(ious) if ious else np.array([0.0])
print('=' * 70)
print('伪标签目录 : %s' % PRED_DIR)
print('帧数       : %d   （%d 帧无任何预测）' % (len(ids), frames_no_pred))
print('=' * 70)
print('真值 Car 3D 框 : %d' % NG)
print('伪标签框       : %d  （占真值 %.1f%%）' % (NP, NP / max(NG, 1) * 100))
print('-' * 70)
print('%-22s %10s %10s' % ('判据', 'IoU>=0.5', 'IoU>=0.7'))
print('-' * 70)
print('%-22s %10d %10d' % ('匹配上的框数', TP5, TP7))
print('%-22s %9.1f%% %9.1f%%' % ('精确率(占伪标签)', TP5 / max(NP, 1) * 100, TP7 / max(NP, 1) * 100))
print('%-22s %9.1f%% %9.1f%%' % ('召回率(占真值)  ', TP5 / max(NG, 1) * 100, TP7 / max(NG, 1) * 100))
f1_5 = 2 * TP5 / max(NP + NG, 1) * 100
print('%-22s %9.1f%%' % ('F1 @0.5', f1_5))
print('-' * 70)
print('匹配对的 BEV IoU: 中位 %.3f  均值 %.3f  P25 %.3f  P75 %.3f'
      % (np.median(ious), ious.mean(), np.percentile(ious, 25), np.percentile(ious, 75)))
print('=' * 70)
print('选阈值看：**精确率优先**（误检会变成错误训练目标），F1 作参考。')
