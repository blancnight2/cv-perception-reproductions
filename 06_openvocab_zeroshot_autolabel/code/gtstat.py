# -*- coding: utf-8 -*-
import glob, os, collections
LBL = '/mnt/d/GuangFU/PV Detection-LNN/YOLO 检测 + ByteTrack 跟踪/kitti_data/yolo/labels/val'
NAMES = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'}

files = sorted(glob.glob(os.path.join(LBL, '*.txt')))
print('=== 前 8 张图的真值 ===')
for f in files[:8]:
    c = collections.Counter(int(l.split()[0]) for l in open(f) if l.split())
    print('%-14s %s' % (os.path.basename(f),
                        '  '.join('%s=%d' % (NAMES[k], v) for k, v in sorted(c.items())) or '(空)'))

tot = collections.Counter()
for f in files:
    for l in open(f):
        v = l.split()
        if v:
            tot[int(v[0])] += 1
print('\n=== 全部 %d 张的类别分布 ===' % len(files))
s = sum(tot.values())
for k in sorted(tot):
    print('%-12s %6d  (%.1f%%)' % (NAMES[k], tot[k], tot[k] / s * 100))
print('%-12s %6d' % ('合计', s))
