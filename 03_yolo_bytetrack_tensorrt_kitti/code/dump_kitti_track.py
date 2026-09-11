# -*- coding: utf-8 -*-
"""把 YOLO+ByteTrack 的跟踪结果导成 KITTI tracking 官方格式，喂给 TrackEval 算 HOTA。

为什么要这一步：
  排行榜主排序指标是 HOTA，而 motmetrics 不实现它（只有 CLEAR MOT 与 Identity）。
  HOTA 会把检测与关联拆成 DetA / AssA 两个数 —— 正好把本项目
  「瓶颈在检测召回、关联没问题」的结论从间接论证变成直接测量。
  附带好处：TrackEval 是 KITTI 官方所用实现，它算出的 MOTA 可交叉验证我手写的那套评测。

输出格式（KITTI tracking，18 列）：
  frame id type truncated occluded alpha x1 y1 x2 y2 h w l X Y Z ry score
  纯 2D 跟踪没有 3D 量，按 KITTI 惯例填 -1 / -10 / -1000。
"""
import os
import sys
import argparse
import numpy as np

TRACK_ROOT = ('/mnt/d/GuangFU/PV Detection-LNN/YOLO 检测 + ByteTrack 跟踪/'
              'data_tracking_image_2/training')
CLS_NAME = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--name', required=True, help='tracker 名字，作为输出子目录')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--tracker', default='bytetrack.yaml')
    ap.add_argument('--te-root', default=os.path.expanduser('~/TrackEval'))
    args = ap.parse_args()

    from ultralytics import YOLO

    out_dir = os.path.join(args.te_root, 'data/trackers/kitti/kitti_2d_box_train',
                           args.name, 'data')
    os.makedirs(out_dir, exist_ok=True)
    seqs = sorted(os.listdir(os.path.join(TRACK_ROOT, 'image_02')))
    print('导出 %d 段 -> %s' % (len(seqs), out_dir))

    model = YOLO(args.weights)
    lengths = {}
    for si, seq in enumerate(seqs):
        d = os.path.join(TRACK_ROOT, 'image_02', seq)
        # 跨序列必须重置，否则上一段的 ID 会串到下一段
        tks = getattr(getattr(model, 'predictor', None), 'trackers', None)
        if tks:
            for tk in tks:
                tk.reset()
        res = model.track(source=d, tracker=args.tracker, conf=args.conf,
                          imgsz=args.imgsz, stream=True, verbose=False, persist=True)
        lines, n = [], 0
        for fr, r in enumerate(res):
            n += 1
            b = r.boxes
            if b is None or b.id is None:
                continue
            xyxy = b.xyxy.cpu().numpy()
            ids = b.id.cpu().numpy().astype(int)
            cls = b.cls.cpu().numpy().astype(int)
            cf = b.conf.cpu().numpy()
            for k in range(len(ids)):
                nm = CLS_NAME.get(int(cls[k]))
                if nm is None:
                    continue
                x1, y1, x2, y2 = xyxy[k]
                lines.append('%d %d %s -1 -1 -10 %.4f %.4f %.4f %.4f '
                             '-1000 -1000 -1000 -1000 -1000 -1000 -10 %.6f'
                             % (fr, int(ids[k]), nm, x1, y1, x2, y2, float(cf[k])))
        lengths[seq] = n
        open(os.path.join(out_dir, seq + '.txt'), 'w').write('\n'.join(lines) + '\n')
        print('  [%2d/%d] %s  %4d 帧  %d 条轨迹记录' % (si + 1, len(seqs), seq, n, len(lines)))

    # ---- 顺带备好 GT 与 seqmap（TrackEval 要求的目录结构）----
    gt_fol = os.path.join(args.te_root, 'data/gt/kitti/kitti_2d_box_train')
    os.makedirs(gt_fol, exist_ok=True)
    lbl_src = os.path.join(os.path.dirname(TRACK_ROOT), 'label_02')
    lbl_dst = os.path.join(gt_fol, 'label_02')
    if not os.path.exists(lbl_dst):
        os.symlink(lbl_src, lbl_dst)
        print('\nGT 已软链: %s -> %s' % (lbl_dst, lbl_src))
    smap = os.path.join(gt_fol, 'evaluate_tracking.seqmap.training')
    with open(smap, 'w') as f:
        for s in seqs:
            f.write('%s empty %06d %06d\n' % (s, 0, lengths[s]))
    print('seqmap 已写: %s（%d 段）' % (smap, len(seqs)))
    print('\n下一步：python %s/scripts/run_kitti.py --TRACKERS_TO_EVAL %s'
          % (args.te_root, args.name))


if __name__ == '__main__':
    main()
