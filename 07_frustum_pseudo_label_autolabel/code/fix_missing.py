# -*- coding: utf-8 -*-
"""补齐因 while-read 漏掉最后一行而缺失的标签文件"""
import os, shutil

P = '/home/blancnight/OpenPCDet/data/kitti_pseudo'
SRC = '/home/blancnight/OpenPCDet/data/kitti'
PSEUDO = '/home/blancnight/autolabel/pseudo_gt_train'
L = os.path.join(P, 'training/label_2')
DONTCARE = ('DontCare -1 -1 -10 0.00 0.00 0.00 0.00 '
            '-1 -1 -1 -1000 -1000 -1000 -10\n')


def ids(name):
    with open(os.path.join(P, 'ImageSets', name)) as f:
        return [l.strip() for l in f if l.strip()]


tr, va = ids('train.txt'), ids('val.txt')
print('train.txt %d 个ID   val.txt %d 个ID   合计 %d' % (len(tr), len(va), len(tr) + len(va)))
print('label_2 现有 %d 个文件' % len(os.listdir(L)))

n_tr = n_va = 0
for fid in tr:
    p = os.path.join(L, fid + '.txt')
    if os.path.exists(p):
        continue
    src = os.path.join(PSEUDO, fid + '.txt')
    if os.path.exists(src) and os.path.getsize(src) > 0:
        shutil.copyfile(src, p)
    else:
        open(p, 'w').write(DONTCARE)
    n_tr += 1
    print('  补 train %s' % fid)

for fid in va:
    p = os.path.join(L, fid + '.txt')
    if os.path.exists(p):
        continue
    shutil.copyfile(os.path.join(SRC, 'training/label_2', fid + '.txt'), p)
    n_va += 1
    print('  补 val   %s' % fid)

# 顺带再扫一遍空文件
n_empty = 0
for fn in os.listdir(L):
    p = os.path.join(L, fn)
    if os.path.getsize(p) == 0:
        open(p, 'w').write(DONTCARE)
        n_empty += 1

print()
print('补 train %d 个, val %d 个, 空文件 %d 个' % (n_tr, n_va, n_empty))
print('label_2 现有 %d 个文件（应为 %d）' % (len(os.listdir(L)), len(tr) + len(va)))

# 校验：train 帧不应含 GT 才有的类别
import collections
c = collections.Counter()
for fid in tr[:500]:
    for line in open(os.path.join(L, fid + '.txt')):
        v = line.split()
        if v:
            c[v[0]] += 1
print('抽查 500 个 train 帧的类别分布:', dict(c))
print('（应只有 Car 和 DontCare —— 出现 Pedestrian/Van 等说明混进了真值）')
