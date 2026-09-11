# -*- coding: utf-8 -*-
"""KITTI Tracking 上评测 YOLO + ByteTrack 的 MOTA / IDF1 / ID switch。

此前这个项目只写了"ByteTrack 已跑通"，从未测过任何跟踪指标——
而 ByteTrack 的全部价值就在 ID 一致性，不测等于没做。

评测协议（对齐 KITTI 官方跟踪 benchmark 的关键几条）：
  · 逐类评测 Car / Pedestrian，IoU 阈值 0.5
  · Van 对 Car 记为 ignore，Person_sitting 对 Pedestrian 记为 ignore，DontCare 一律 ignore
    （官方做法：这些区域里的预测既不算 TP 也不罚 FP）
  · 只用 training 的 21 个序列（testing 无公开真值）

用法：
  python eval_track.py --seqs 0000,0001,0002       # 先试几段
  python eval_track.py                              # 全部 21 段
"""
import os
import sys
import json
import time
import argparse
import numpy as np

TRACK_ROOT = ('/mnt/d/GuangFU/PV Detection-LNN/YOLO 检测 + ByteTrack 跟踪/'
              'data_tracking_image_2/training')
LABEL_DIR = os.environ.get('KITTI_TRACK_LABEL', os.path.join(
    os.path.dirname(TRACK_ROOT), 'label_02'))
# 模型：优先用已导出的 TensorRT 引擎，否则回退 .pt
DEFAULT_W = '/mnt/d/GuangFU/PV Detection-LNN/runs/detect/train/weights/best.pt'

# KITTI 类别 -> 我们的三类；ignore 类在匹配时既不算命中也不罚误检
NAME2CLS = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
IGNORE_FOR = {0: {'Van', 'DontCare'}, 1: {'Person_sitting', 'DontCare'}, 2: {'DontCare'}}


def load_gt(seq):
    """label_02/<seq>.txt -> {frame: [(track_id, cls_name, x1,y1,x2,y2), ...]}"""
    f = os.path.join(LABEL_DIR, seq + '.txt')
    if not os.path.exists(f):
        sys.exit('[缺真值] 找不到 %s\n'
                 '  KITTI 跟踪真值要单独下载 data_tracking_label_2.zip（约 4 MB），\n'
                 '  解压后把 label_02/ 放到 %s' % (f, os.path.dirname(LABEL_DIR)))
    gt = {}
    for line in open(f):
        v = line.split()
        if len(v) < 17:
            continue
        fr, tid, name = int(v[0]), int(v[1]), v[2]
        box = [float(x) for x in v[6:10]]
        gt.setdefault(fr, []).append((tid, name, box))
    return gt


def iou_mat(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float), np.asarray(b, float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=DEFAULT_W)
    ap.add_argument('--seqs', default='')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou-thr', type=float, default=0.5)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--tracker', default='bytetrack.yaml')
    ap.add_argument('--out', default=os.path.expanduser(
        '~/cv-perception-reproductions/03_yolo_bytetrack_tensorrt_kitti/results/tracking.json'))
    args = ap.parse_args()

    import motmetrics as mm
    from ultralytics import YOLO

    seqs = ([s.strip() for s in args.seqs.split(',') if s.strip()] or
            sorted(os.listdir(os.path.join(TRACK_ROOT, 'image_02'))))
    print('=' * 78)
    print('KITTI Tracking 评测 · %d 个序列 · %s · conf=%.2f · IoU=%.1f'
          % (len(seqs), os.path.basename(args.weights), args.conf, args.iou_thr))
    print('=' * 78)

    model = YOLO(args.weights)
    accs = {c: {} for c in (0, 1, 2)}       # cls -> {seq: accumulator}
    t0 = time.time()

    for si, seq in enumerate(seqs):
        d = os.path.join(TRACK_ROOT, 'image_02', seq)
        gt = load_gt(seq)
        acc = {c: mm.MOTAccumulator(auto_id=False) for c in (0, 1, 2)}
        # ultralytics 的 track 对文件夹会按序读，persist 由它内部维护
        res = model.track(source=d, tracker=args.tracker, conf=args.conf,
                          imgsz=args.imgsz, stream=True, verbose=False, persist=False)
        n_fr = 0
        for fr, r in enumerate(res):
            n_fr += 1
            b = r.boxes
            if b is None or b.id is None:
                pb, pid, pcl = np.zeros((0, 4)), np.array([]), np.array([])
            else:
                pb = b.xyxy.cpu().numpy()
                pid = b.id.cpu().numpy().astype(int)
                pcl = b.cls.cpu().numpy().astype(int)
            g = gt.get(fr, [])
            for c in (0, 1, 2):
                gname = [k for k, v in NAME2CLS.items() if v == c][0]
                gsel = [(t, bx) for t, nm, bx in g if nm == gname]
                ign = [bx for t, nm, bx in g if nm in IGNORE_FOR[c]]
                msk = pcl == c
                pbb, pii = pb[msk], pid[msk]
                # 落在 ignore 区域里的预测：既不计 TP 也不罚 FP -> 直接剔除
                if len(ign) and len(pbb):
                    keep = iou_mat(pbb, ign).max(1) < args.iou_thr
                    pbb, pii = pbb[keep], pii[keep]
                gids = [t for t, _ in gsel]
                gbb = [bx for _, bx in gsel]
                dist = iou_mat(gbb, pbb)
                dist = np.where(dist >= args.iou_thr, 1 - dist, np.nan)
                acc[c].update(gids, list(pii), dist, frameid=fr)
        for c in (0, 1, 2):
            accs[c][seq] = acc[c]
        print('  [%2d/%d] %s  %4d 帧  真值目标 %d'
              % (si + 1, len(seqs), seq, n_fr, sum(len(v) for v in gt.values())))

    print('\n耗时 %.1f 分钟\n' % ((time.time() - t0) / 60))
    mh = mm.metrics.create()
    METS = ['mota', 'motp', 'idf1', 'num_switches', 'mostly_tracked',
            'mostly_lost', 'num_false_positives', 'num_misses', 'num_objects']
    out = {}
    print('%-12s %7s %7s %7s %7s %5s %5s %8s %8s' %
          ('类别', 'MOTA', 'MOTP', 'IDF1', 'IDSW', 'MT', 'ML', 'FP', 'FN'))
    print('-' * 78)
    for c, nm in ((0, 'Car'), (1, 'Pedestrian'), (2, 'Cyclist')):
        names = [s for s in seqs if accs[c][s].events['Type'].notna().any()]
        if not names:
            print('%-12s (无真值目标，跳过)' % nm)
            continue
        s = mh.compute_many([accs[c][s] for s in names], metrics=METS,
                            names=names, generate_overall=True).loc['OVERALL']
        # MOTP 在 motmetrics 里是距离(1-IoU)，转成 IoU 更直观
        print('%-12s %6.2f%% %6.3f %6.2f%% %7d %5d %5d %8d %8d'
              % (nm, 100 * s['mota'], 1 - s['motp'], 100 * s['idf1'],
                 int(s['num_switches']), int(s['mostly_tracked']),
                 int(s['mostly_lost']), int(s['num_false_positives']),
                 int(s['num_misses'])))
        out[nm] = {'MOTA': float(100 * s['mota']), 'MOTP_IoU': float(1 - s['motp']),
                   'IDF1': float(100 * s['idf1']), 'IDSW': int(s['num_switches']),
                   'MT': int(s['mostly_tracked']), 'ML': int(s['mostly_lost']),
                   'FP': int(s['num_false_positives']), 'FN': int(s['num_misses']),
                   'num_objects': int(s['num_objects'])}
    out['_config'] = {'weights': args.weights, 'tracker': args.tracker,
                      'conf': args.conf, 'iou_thr': args.iou_thr,
                      'imgsz': args.imgsz, 'n_seqs': len(seqs), 'seqs': seqs}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, 'w'), indent=2, ensure_ascii=False)
    print('\n已存 -> %s' % args.out)
    print('\n注：MOTP 已由 motmetrics 的距离(1−IoU)换算回 IoU，越大越好。')


if __name__ == '__main__':
    main()
