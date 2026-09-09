# -*- coding: utf-8 -*-
"""
生成 KITTI 3D 伪标签（视锥法 v3）

两种 2D 框来源：
  --src gt     用 label_2 里的 2D 框（弱监督设定：只标 2D 不标 3D）
  --src yoloe  用 YOLOE 零样本检测的 2D 框（完全零人工标注）

输出 KITTI label_2 格式，可直接喂给 OpenPCDet。
"""
import os, sys, json, argparse, time
import numpy as np
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frustum_lib import Calib
from frustum_v2 import frustum_to_box_v2

ROOT = '/home/blancnight/OpenPCDet/data/kitti/training'
IMG_WH = None          # 每帧单独读，KITTI 尺寸不完全一致


def read_gt2d(fid, cls):
    p = os.path.join(ROOT, 'label_2', fid + '.txt')
    out = []
    if not os.path.exists(p):
        return out
    for line in open(p):
        v = line.split()
        if len(v) < 15 or v[0] != cls:
            continue
        out.append(dict(box2d=[float(x) for x in v[4:8]],
                        trunc=float(v[1]), occ=int(v[2])))
    return out


def load_yoloe_boxes(path):
    """读 eval_zeroshot.py 产出的 preds_*.jsonl，返回 {stem: [box2d,...]}"""
    d = {}
    for line in open(path):
        r = json.loads(line)
        keep = [b for b, l in zip(r['boxes'], r['labels']) if l == 0]   # 0 = Car
        d[r['stem']] = keep
    return d


def img_size(fid):
    try:
        from PIL import Image
        with Image.open(os.path.join(ROOT, 'image_2', fid + '.png')) as im:
            return im.size
    except Exception:
        return (1242, 375)


def proc_one(job):
    fid, boxes2d, cls, min_pts = job
    vp = os.path.join(ROOT, 'velodyne', fid + '.bin')
    cp = os.path.join(ROOT, 'calib', fid + '.txt')
    if not (os.path.exists(vp) and os.path.exists(cp)) or not boxes2d:
        return fid, [], len(boxes2d)
    calib = Calib(cp)
    pts = np.fromfile(vp, dtype=np.float32).reshape(-1, 4)
    wh = img_size(fid)

    lines = []
    for b in boxes2d:
        r = frustum_to_box_v2(pts, calib, b, cls, img_wh=wh)
        if r is None:
            continue
        h, w, l, x, y, z, ry = r
        # 基本合理性过滤：尺寸与距离
        if not (0.5 < h < 4 and 0.5 < w < 4 and 0.5 < l < 12):
            continue
        if not (0 < z < 80):
            continue
        alpha = ry - np.arctan2(x, z)
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
        lines.append('%s 0.00 0 %.2f %.2f %.2f %.2f %.2f '
                     '%.2f %.2f %.2f %.2f %.2f %.2f %.2f'
                     % (cls, alpha, b[0], b[1], b[2], b[3],
                        h, w, l, x, y, z, ry))
    return fid, lines, len(boxes2d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', choices=['gt', 'yoloe'], default='gt')
    ap.add_argument('--yoloe-preds', default=None)
    ap.add_argument('--split', default='train')
    ap.add_argument('--cls', default='Car')
    ap.add_argument('--out', default=None)
    ap.add_argument('--workers', type=int, default=max(1, cpu_count() - 2))
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    out = args.out or os.path.expanduser('~/autolabel/pseudo_%s_%s' % (args.src, args.split))
    os.makedirs(out, exist_ok=True)

    ids = [l.strip() for l in open(
        '/home/blancnight/OpenPCDet/data/kitti/ImageSets/%s.txt' % args.split)]
    if args.limit:
        ids = ids[:args.limit]

    if args.src == 'gt':
        src2d = {f: [g['box2d'] for g in read_gt2d(f, args.cls)] for f in ids}
    else:
        if not args.yoloe_preds:
            sys.exit('--src yoloe 需要 --yoloe-preds 指向 preds_*.jsonl')
        yb = load_yoloe_boxes(args.yoloe_preds)
        src2d = {f: yb.get(f, []) for f in ids}

    n_2d = sum(len(v) for v in src2d.values())
    print('划分 %s：%d 帧，2D 框来源 %s，共 %d 个 %s 框'
          % (args.split, len(ids), args.src, n_2d, args.cls))
    print('输出 -> %s   进程数 %d' % (out, args.workers))

    jobs = [(f, src2d[f], args.cls, 8) for f in ids]
    t0 = time.time()
    n_box = n_empty = 0
    with Pool(args.workers) as pool:
        for i, (fid, lines, n_in) in enumerate(
                pool.imap_unordered(proc_one, jobs, chunksize=8), 1):
            with open(os.path.join(out, fid + '.txt'), 'w') as f:
                f.write('\n'.join(lines) + ('\n' if lines else ''))
            n_box += len(lines)
            if not lines:
                n_empty += 1
            if i % 250 == 0 or i == len(jobs):
                el = time.time() - t0
                print('  %4d/%d  已生成 %d 框  用时 %.1f 分  预计剩余 %.1f 分'
                      % (i, len(jobs), n_box, el / 60,
                         el / 60 / i * (len(jobs) - i)))

    print()
    print('=' * 56)
    print('  总帧数      : %d' % len(ids))
    print('  输入 2D 框  : %d' % n_2d)
    print('  产出 3D 框  : %d  (转化率 %.1f%%)'
          % (n_box, 100.0 * n_box / max(n_2d, 1)))
    print('  空标注帧    : %d (%.1f%%)' % (n_empty, 100.0 * n_empty / len(ids)))
    print('  耗时        : %.1f 分钟' % ((time.time() - t0) / 60))
    print('  输出目录    : %s' % out)
    print('=' * 56)


if __name__ == '__main__':
    main()
