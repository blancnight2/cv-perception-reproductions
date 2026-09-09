# -*- coding: utf-8 -*-
"""
只生成 gt_database（infos 已经好了）。

上游问题：kitti_dataset.py 把 `from pathlib import Path` 写在
`if __name__ == '__main__':` 里，而 create_groundtruth_database 函数体用到了 Path，
所以只能经由它自带的 CLI 入口调用；作为模块 import 时会 NameError。
这里手动把 Path 注入模块命名空间绕开。
"""
import sys, yaml
from pathlib import Path
from easydict import EasyDict

sys.path.insert(0, '/home/blancnight/OpenPCDet')
import pcdet.datasets.kitti.kitti_dataset as kd
kd.Path = Path                      # 注入缺失的名字

CFG = '/home/blancnight/OpenPCDet/tools/cfgs/dataset_configs/kitti_pseudo_dataset.yaml'
ROOT = Path('/home/blancnight/OpenPCDet/data/kitti_pseudo')

cfg = EasyDict(yaml.safe_load(open(CFG)))
ds = kd.KittiDataset(dataset_cfg=cfg, class_names=['Car'],
                     root_path=ROOT, training=False)
ds.set_split('train')
ds.create_groundtruth_database(ROOT / 'kitti_infos_train.pkl', split='train')
print('gt_database 完成')
