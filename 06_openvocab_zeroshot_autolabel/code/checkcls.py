# -*- coding: utf-8 -*-
"""看原始 KITTI 标注里有哪些类别，以及转换后保留了哪几类"""
import glob, os, collections

ORIG = '/mnt/d/GuangFU/PV Detection-LNN/runs/kitti_labels'
YOLO = '/mnt/d/GuangFU/PV Detection-LNN/YOLO 检测 + ByteTrack 跟踪/kitti_data/yolo/labels/val'

c = collections.Counter()
files = sorted(glob.glob(os.path.join(ORIG, '*.txt')))
print('原始标注文件数:', len(files))
for f in files[:2000]:
    for line in open(f):
        v = line.split()
        if v:
            c[v[0]] += 1
print('\n=== 原始 KITTI 类别 (前2000个文件) ===')
for k, v in c.most_common():
    print('  %-16s %6d' % (k, v))

cy = collections.Counter()
for f in glob.glob(os.path.join(YOLO, '*.txt')):
    for line in open(f):
        v = line.split()
        if v:
            cy[v[0]] += 1
print('\n=== 转换后 YOLO val 保留的类别 id ===')
for k, v in sorted(cy.items()):
    print('  id=%s  %6d' % (k, v))
print('\n>> 原始里有但 YOLO 标注中【没有】的类别 = 模型检到就算误检')
