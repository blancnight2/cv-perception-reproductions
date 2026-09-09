# -*- coding: utf-8 -*-
"""
搭建「完全零人工标注」训练集：kitti_pseudo_yoloe
  train 帧 -> YOLOE 零样本 2D 框经视锥管线生成的 3D 伪标签
  val   帧 -> 真值标签（评测必须用真值，否则作弊）

一体化：建目录 -> 组装 label_2 -> 写两份 yaml -> infos -> gt_database -> 校验

踩过的坑（都已规避）：
  · ImageSets 用 Python 读，不用 `while read`（会漏掉没有结尾换行的最后一行）
  · create_kitti_infos 入口把 data_path 写死成 data/kitti，必须自己调函数
  · kitti_dataset.py 的 `Path` 只在 __main__ 里 import，作为模块调用会 NameError
    -> 提前注入 kd.Path
  · 空标签文件写一行 DontCare，不留 0 字节
"""
import os, io, re, sys, shutil, yaml
from pathlib import Path
from easydict import EasyDict

R = '/home/blancnight/OpenPCDet'
SRC = os.path.join(R, 'data/kitti')
DST = os.path.join(R, 'data/kitti_pseudo_yoloe')
PSEUDO = os.path.expanduser(os.environ.get('PSEUDO_DIR', '~/autolabel/pseudo_yoloe0.05_train'))
DS_NAME = 'kitti_pseudo_yoloe_dataset'
MODEL_NAME = 'pointpillar_car_pseudo_yoloe'
DCFG = os.path.join(R, 'tools/cfgs/dataset_configs')
MCFG = os.path.join(R, 'tools/cfgs/kitti_models')
DONTCARE = ('DontCare -1 -1 -10 0.00 0.00 0.00 0.00 '
            '-1 -1 -1 -1000 -1000 -1000 -10\n')


def ids(name):
    with io.open(os.path.join(SRC, 'ImageSets', name), encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]


# ---------- 1) 目录 + 软链 ----------
print('=' * 66)
if os.path.exists(DST):
    shutil.rmtree(DST)
os.makedirs(os.path.join(DST, 'training'))
os.makedirs(os.path.join(DST, 'testing'))
for d in ('image_2', 'velodyne', 'calib'):
    os.symlink(os.path.join(SRC, 'training', d), os.path.join(DST, 'training', d))
for d in ('velodyne', 'calib'):
    os.symlink(os.path.join(SRC, 'testing', d), os.path.join(DST, 'testing', d))
shutil.copytree(os.path.join(SRC, 'ImageSets'), os.path.join(DST, 'ImageSets'))
print('目录就绪（大文件走软链，不复制）:', DST)

# ---------- 2) 组装 label_2 ----------
L = os.path.join(DST, 'training/label_2')
os.makedirs(L)
tr, va = ids('train.txt'), ids('val.txt')
n_ps = n_empty = 0
for fid in tr:
    s = os.path.join(PSEUDO, fid + '.txt')
    d = os.path.join(L, fid + '.txt')
    if os.path.exists(s) and os.path.getsize(s) > 0:
        shutil.copyfile(s, d); n_ps += 1
    else:
        io.open(d, 'w', encoding='utf-8', newline='\n').write(DONTCARE); n_empty += 1
for fid in va:
    shutil.copyfile(os.path.join(SRC, 'training/label_2', fid + '.txt'),
                    os.path.join(L, fid + '.txt'))

n_box = 0
for fid in tr:
    for line in io.open(os.path.join(L, fid + '.txt'), encoding='utf-8'):
        if line.split() and line.split()[0] == 'Car':
            n_box += 1
print('label_2: train %d 帧（%d 帧有伪标签 / %d 帧空），val %d 帧真值，共 %d 个文件'
      % (len(tr), n_ps, n_empty, len(va), len(os.listdir(L))))
print('train 伪标签 Car 框总数: %d' % n_box)
assert len(os.listdir(L)) == len(tr) + len(va), '标签文件数对不上！'

# ---------- 3) 两份 yaml ----------
base_ds = io.open(os.path.join(DCFG, 'kitti_dataset.yaml'), encoding='utf-8').read()
s = base_ds.replace("DATA_PATH: '../data/kitti'", "DATA_PATH: '../data/kitti_pseudo_yoloe'")
s = s.replace("filter_by_min_points: ['Car:5', 'Pedestrian:5', 'Cyclist:5']",
              "filter_by_min_points: ['Car:5']")
s = re.sub(r"SAMPLE_GROUPS:\s*\[[^\]]*\]", "SAMPLE_GROUPS: ['Car:15']", s)
DS_YAML = os.path.join(DCFG, DS_NAME + '.yaml')
io.open(DS_YAML, 'w', encoding='utf-8', newline='\n').write(s)

# 模型配置：直接抄已验证过的 pointpillar_car_pseudo，只换 _BASE_CONFIG_
base_m = io.open(os.path.join(MCFG, 'pointpillar_car_pseudo.yaml'), encoding='utf-8').read()
m = base_m.replace('cfgs/dataset_configs/kitti_pseudo_dataset.yaml',
                   'cfgs/dataset_configs/%s.yaml' % DS_NAME)
M_YAML = os.path.join(MCFG, MODEL_NAME + '.yaml')
io.open(M_YAML, 'w', encoding='utf-8', newline='\n').write(m)
y = yaml.safe_load(m)
print('配置: %s.yaml / %s.yaml   CLASS_NAMES=%s   anchor组数=%d'
      % (DS_NAME, MODEL_NAME, y['CLASS_NAMES'],
         len(y['MODEL']['DENSE_HEAD']['ANCHOR_GENERATOR_CONFIG'])))
assert y['CLASS_NAMES'] == ['Car']
assert '%s.yaml' % DS_NAME in m

# ---------- 4) infos + gt_database ----------
print('=' * 66)
print('生成 infos + gt_database ...')
sys.path.insert(0, R)
import pcdet.datasets.kitti.kitti_dataset as kd
kd.Path = Path                     # 上游把 Path 写在 __main__ 里，模块调用会 NameError

cfg = EasyDict(yaml.safe_load(io.open(DS_YAML, encoding='utf-8')))
kd.create_kitti_infos(dataset_cfg=cfg, class_names=['Car'],
                      data_path=Path(DST), save_path=Path(DST))

# ---------- 5) 校验 ----------
import pickle
print('=' * 66)
print('=== 校验 ===')
for sp in ('train', 'val'):
    p = os.path.join(DST, 'kitti_infos_%s.pkl' % sp)
    d = pickle.load(open(p, 'rb'))
    n = sum(sum(1 for x in i['annos']['name'] if x == 'Car') for i in d)
    print('  infos %-5s : %d 帧, Car %d 个' % (sp, len(d), n))
gp = os.path.join(DST, 'gt_database')
print('  gt_database : %d 个样本' % (len(os.listdir(gp)) if os.path.isdir(gp) else -1))
db = pickle.load(open(os.path.join(DST, 'kitti_dbinfos_train.pkl'), 'rb'))
for k, v in db.items():
    print('  dbinfos %-8s: %d 个' % (k, len(v)))
print('=' * 66)
print('就绪。训练命令：')
print('  python train.py --cfg_file cfgs/kitti_models/%s.yaml --batch_size 4 '
      '--epochs 80 --extra_tag v1' % MODEL_NAME)
