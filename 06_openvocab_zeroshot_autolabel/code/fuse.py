# -*- coding: utf-8 -*-
"""
多提示词投票融合 —— 给零样本框造出真实置信度

动机：开放词汇/MLLM 检测器不输出可靠的 confidence，而 AP 依赖按分排序画 PR 曲线。
做法：同一张图用 K 组不同提示词各推一遍，同一目标被越多提示词同时框到 = 越可信。
      融合分 = (命中该目标的提示词数 / K) ** alpha  ×  各变体置信度的加权均值
      融合框 = 按置信度加权平均（类 WBF）

用法:
    python fuse.py --tags V1 V2 V3 V4 V5 --iou 0.55 --alpha 1.0
"""
import argparse, os, glob, json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision

VAL_IMG = os.environ.get('VAL_IMG',
    '/mnt/d/GuangFU/PV Detection-LNN/YOLO 检测 + ByteTrack 跟踪/kitti_data/yolo/images/val')
VAL_LBL = os.environ.get('VAL_LBL',
    '/mnt/d/GuangFU/PV Detection-LNN/YOLO 检测 + ByteTrack 跟踪/kitti_data/yolo/labels/val')
NAMES = ['Car', 'Pedestrian', 'Cyclist']


def load_gt(stem, W, H):
    f = os.path.join(VAL_LBL, stem + '.txt')
    boxes, labels = [], []
    if os.path.exists(f):
        for line in open(f):
            v = line.split()
            if len(v) < 5:
                continue
            c = int(v[0]); cx, cy, w, h = (float(x) for x in v[1:5])
            boxes.append([(cx - w/2)*W, (cy - h/2)*H, (cx + w/2)*W, (cy + h/2)*H])
            labels.append(c)
    return (torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            torch.tensor(labels, dtype=torch.long))


def iou_mat(a, b):
    """a:(N,4) b:(M,4) -> (N,M)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def fuse_one(items, K, iou_th, alpha):
    """items: list of (box[4], score, label, variant_id) -> 融合后的 (boxes, scores, labels)"""
    ob, os_, ol = [], [], []
    for cls in set(i[2] for i in items):
        sub = [i for i in items if i[2] == cls]
        sub.sort(key=lambda x: -x[1])
        B = np.array([s[0] for s in sub], dtype=np.float64)
        S = np.array([s[1] for s in sub], dtype=np.float64)
        V = np.array([s[3] for s in sub])
        used = np.zeros(len(sub), bool)
        for i in range(len(sub)):
            if used[i]:
                continue
            ious = iou_mat(B[i:i+1], B)[0]
            grp = (ious >= iou_th) & (~used)
            used |= grp
            gb, gs, gv = B[grp], S[grp], V[grp]
            w = gs / max(gs.sum(), 1e-9)
            box = (gb * w[:, None]).sum(0)
            agree = len(set(gv.tolist())) / K          # 有多少个提示词投了这一票
            ob.append(box.tolist())
            os_.append(float((agree ** alpha) * gs.mean()))
            ol.append(int(cls))
    return ob, os_, ol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', required=True)
    ap.add_argument('--iou', type=float, default=0.55)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--min-agree', type=int, default=1, help='至少几个提示词命中才保留')
    args = ap.parse_args()

    K = len(args.tags)
    store = {}
    for vid, tag in enumerate(args.tags):
        p = os.path.expanduser('~/openvocab/preds_%s.jsonl' % tag)
        n = 0
        for line in open(p):
            r = json.loads(line)
            store.setdefault(r['stem'], []).extend(
                (b, s, l, vid) for b, s, l in zip(r['boxes'], r['scores'], r['labels']))
            n += 1
        print('%s: %d 张' % (tag, n))

    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox',
                                  class_metrics=True, backend='faster_coco_eval')
    imgs = sorted(glob.glob(os.path.join(VAL_IMG, '*')))
    n_box = 0
    for path in tqdm(imgs, ncols=80):
        stem = os.path.splitext(os.path.basename(path))[0]
        with Image.open(path) as im:
            W, H = im.size
        items = store.get(stem, [])
        b, s, l = fuse_one(items, K, args.iou, args.alpha) if items else ([], [], [])
        if args.min_agree > 1:
            keep = [i for i in range(len(s)) if s[i] >= (args.min_agree / K) ** args.alpha * 1e-9]
            b = [b[i] for i in keep]; s = [s[i] for i in keep]; l = [l[i] for i in keep]
        n_box += len(b)
        preds = [dict(boxes=torch.tensor(b, dtype=torch.float32).reshape(-1, 4),
                      scores=torch.tensor(s, dtype=torch.float32),
                      labels=torch.tensor(l, dtype=torch.long))]
        gb, gl = load_gt(stem, W, H)
        metric.update(preds, [dict(boxes=gb, labels=gl)])

    res = metric.compute()
    print('\n' + '=' * 56)
    print('  融合变体   : %s (K=%d)' % (','.join(args.tags), K))
    print('  IoU阈值    : %.2f    alpha: %.1f' % (args.iou, args.alpha))
    print('  平均框数   : %.1f / 图' % (n_box / len(imgs)))
    print('  mAP50      : %.4f' % res['map_50'].item())
    print('  mAP50-95   : %.4f' % res['map'].item())
    for i, v in enumerate(res['map_per_class'].tolist()):
        if i < len(NAMES):
            print('    %-12s: %.4f' % (NAMES[i], v))
    print('=' * 56)


if __name__ == '__main__':
    main()
