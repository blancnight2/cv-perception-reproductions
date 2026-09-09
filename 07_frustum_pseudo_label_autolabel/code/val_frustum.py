# -*- coding: utf-8 -*-
"""用 GT 的 2D 框跑视锥管线，与 GT 3D 框对比 —— 测几何管线的上限"""
import os, sys, glob, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frustum_lib import Calib, frustum_to_box
from frustum_v2 import frustum_to_box_v2

ROOT = '/home/blancnight/OpenPCDet/data/kitti/training'


def read_label(p):
    out = []
    for line in open(p):
        v = line.split()
        if len(v) < 15:
            continue
        out.append(dict(
            cls=v[0], trunc=float(v[1]), occ=int(v[2]),
            box2d=[float(x) for x in v[4:8]],
            h=float(v[8]), w=float(v[9]), l=float(v[10]),
            x=float(v[11]), y=float(v[12]), z=float(v[13]), ry=float(v[14])))
    return out


def bev_corners(x, z, l, w, ry):
    """相机系 BEV(x-z 平面) 四角"""
    c, s = np.cos(ry), np.sin(ry)
    dx = np.array([l / 2, l / 2, -l / 2, -l / 2])
    dz = np.array([w / 2, -w / 2, -w / 2, w / 2])
    return list(zip(x + c * dx + s * dz, z - s * dx + c * dz))


def bev_iou(a, b):
    from shapely.geometry import Polygon
    pa = Polygon(bev_corners(a['x'], a['z'], a['l'], a['w'], a['ry']))
    pb = Polygon(bev_corners(b['x'], b['z'], b['l'], b['w'], b['ry']))
    if not pa.is_valid or not pb.is_valid:
        return 0.0
    inter = pa.intersection(pb).area
    u = pa.area + pb.area - inter
    return inter / u if u > 1e-9 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--cls', default='Car')
    ap.add_argument('--max-dist', type=float, default=45.0)
    ap.add_argument('--v2', action='store_true', help='用 v2（L-shape + 先验尺寸 + 中心补偿）')
    ap.add_argument('--no-prior', action='store_true')
    args = ap.parse_args()

    ids = [l.strip() for l in
           open('/home/blancnight/OpenPCDet/data/kitti/ImageSets/val.txt')][:args.n]

    ious, dcen, dh, dw, dl, dyaw = [], [], [], [], [], []
    n_gt = n_ok = 0

    for fid in ids:
        lp = os.path.join(ROOT, 'label_2', fid + '.txt')
        vp = os.path.join(ROOT, 'velodyne', fid + '.bin')
        cp = os.path.join(ROOT, 'calib', fid + '.txt')
        if not (os.path.exists(lp) and os.path.exists(vp)):
            continue
        calib = Calib(cp)
        pts = np.fromfile(vp, dtype=np.float32).reshape(-1, 4)

        for g in read_label(lp):
            if g['cls'] != args.cls or g['z'] > args.max_dist:
                continue
            if g['trunc'] > 0.5 or g['occ'] > 1:      # 只看清晰目标，测上限
                continue
            n_gt += 1
            if args.v2:
                r = frustum_to_box_v2(pts, calib, g['box2d'], args.cls,
                                      img_wh=(1242, 375),
                                      use_prior=not args.no_prior)
            else:
                r = frustum_to_box(pts, calib, g['box2d'], args.cls,
                                   img_wh=(1242, 375))
            if r is None:
                continue
            n_ok += 1
            h, w, l, x, y, z, ry = r
            p = dict(h=h, w=w, l=l, x=x, y=y, z=z, ry=ry)
            ious.append(bev_iou(p, g))
            dcen.append(np.hypot(x - g['x'], z - g['z']))
            dh.append(h - g['h']); dw.append(w - g['w']); dl.append(l - g['l'])
            e = abs(((ry - g['ry']) + np.pi) % (2 * np.pi) - np.pi)
            dyaw.append(min(e, abs(np.pi - e)))       # 朝向 180 度模糊不惩罚

    if not ious:
        print('没有产出任何框'); return
    ious = np.array(ious)
    print('=' * 58)
    tag = 'v2' + ('(无先验)' if args.no_prior else '') if args.v2 else 'v1'
    print('  [%s] 帧数 %d   %s 目标 %d   成功出框 %d (%.1f%%)'
          % (tag, len(ids), args.cls, n_gt, n_ok, 100.0 * n_ok / max(n_gt, 1)))
    print('-' * 58)
    print('  BEV IoU  均值 %.3f   中位 %.3f' % (ious.mean(), np.median(ious)))
    for t in (0.25, 0.5, 0.7):
        print('           IoU>%.2f 占比 %.1f%%' % (t, 100.0 * (ious > t).mean()))
    print('  中心误差 均值 %.2f m   中位 %.2f m'
          % (np.mean(dcen), np.median(dcen)))
    print('  尺寸偏差 h %+.2f  w %+.2f  l %+.2f (m, 均值)'
          % (np.mean(dh), np.mean(dw), np.mean(dl)))
    print('  朝向误差 均值 %.1f°  中位 %.1f°'
          % (np.degrees(np.mean(dyaw)), np.degrees(np.median(dyaw))))
    print('=' * 58)


if __name__ == '__main__':
    main()
