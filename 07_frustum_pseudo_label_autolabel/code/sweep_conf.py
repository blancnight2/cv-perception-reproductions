# -*- coding: utf-8 -*-
"""
选自动标注的工作点 —— 刷 AP 和做标注要的阈值不是一回事。

AP 场景：低分误检排在最后，几乎不扣分。
标注场景：每个误检 -> 一个错误的 3D 伪标签 -> 污染训练集。
所以这里按 精确率/召回率/F1 选阈值，不看 AP。

GT 口径：Car = 正样本；Van / Truck / DontCare = 忽略区（命中不算误检，
        因为那确实是车，只是 KITTI 没标成 Car）。

用法：python sweep_conf.py [preds.jsonl]
"""
import os, sys, json
import numpy as np

PRED = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.expanduser('~/openvocab/preds_yoloe-train3712.jsonl')
LBL = os.path.expanduser('~/OpenPCDet/data/kitti/training/label_2')
IDS = os.path.expanduser('~/OpenPCDet/data/kitti/ImageSets/train.txt')
IGNORE = {'Van', 'Truck', 'DontCare', 'Misc', 'Tram'}
THRS = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]


def iou(a, b):
    """a:(4,) b:(N,4) xyxy"""
    if len(b) == 0:
        return np.zeros(0)
    b = np.asarray(b, dtype=np.float64)
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2]); y2 = np.minimum(a[3], b[:, 3])
    w = np.clip(x2 - x1, 0, None); h = np.clip(y2 - y1, 0, None)
    inter = w * h
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(aa + ab - inter, 1e-9, None)


def read_gt(fid):
    car, ign = [], []
    p = os.path.join(LBL, fid + '.txt')
    if not os.path.exists(p):
        return np.zeros((0, 4)), np.zeros((0, 4))
    for line in open(p):
        v = line.split()
        if len(v) < 8:
            continue
        box = [float(v[4]), float(v[5]), float(v[6]), float(v[7])]
        if v[0] == 'Car':
            car.append(box)
        elif v[0] in IGNORE:
            ign.append(box)
    return (np.array(car).reshape(-1, 4), np.array(ign).reshape(-1, 4))


# ---------- 读预测 ----------
preds = {}
with open(PRED) as f:
    for line in f:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        keep = [(b, s) for b, s, l in zip(r['boxes'], r['scores'], r['labels']) if l == 0]
        preds[r['stem']] = keep

ids = [l.strip() for l in open(IDS) if l.strip()]
print('预测覆盖 %d / %d 帧' % (sum(1 for i in ids if i in preds), len(ids)))

gts = {i: read_gt(i) for i in ids}
n_gt = sum(len(g[0]) for g in gts.values())
print('GT Car 2D 框合计 %d 个\n' % n_gt)

print('%-7s %8s %8s %8s %8s %8s  %s' %
      ('conf', '预测框', 'TP', 'FP', '精确率', '召回率', 'F1'))
print('-' * 66)
rows = []
for thr in THRS:
    TP = FP = 0
    matched_total = 0
    for i in ids:
        car, ign = gts[i]
        cand = [b for b, s in preds.get(i, []) if s >= thr]
        used = np.zeros(len(car), dtype=bool)
        for b in cand:
            b = np.asarray(b, dtype=np.float64)
            o = iou(b, car)
            if len(o) and o.max() >= 0.5:
                j = int(o.argmax())
                if not used[j]:
                    used[j] = True; TP += 1
                else:
                    TP += 1          # 重复命中同一目标，仍算检出（不惩罚）
                continue
            oi = iou(b, ign)
            if len(oi) and oi.max() >= 0.5:
                continue             # 落在忽略区，不算误检
            FP += 1
        matched_total += int(used.sum())
    prec = TP / max(TP + FP, 1)
    rec = matched_total / max(n_gt, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    rows.append((thr, TP + FP, TP, FP, prec, rec, f1))
    print('%-7.2f %8d %8d %8d %8.3f %8.3f  %.3f'
          % (thr, TP + FP, TP, FP, prec, rec, f1))

best_f1 = max(rows, key=lambda r: r[6])
best_p = max((r for r in rows if r[5] >= 0.60), key=lambda r: r[4], default=None)
print()
print('F1 最高      : conf=%.2f (P %.3f / R %.3f / F1 %.3f)'
      % (best_f1[0], best_f1[4], best_f1[5], best_f1[6]))
if best_p:
    print('召回>=0.60 中精确率最高 : conf=%.2f (P %.3f / R %.3f)'
          % (best_p[0], best_p[4], best_p[5]))
print()
print('选点建议：自动标注宁可漏、不可错 —— 错框会变成错误 3D 标签污染训练；')
print('          漏检只是少几个样本。所以偏向精确率一侧，别取 F1 最高点。')
