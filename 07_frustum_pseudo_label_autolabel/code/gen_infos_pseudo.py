# -*- coding: utf-8 -*-
"""
为伪标签数据集生成 infos + gt_database。

OpenPCDet 自带的 create_kitti_infos 入口把 data_path 写死成 data/kitti，
完全忽略 yaml 里的 DATA_PATH，所以自定义数据目录必须自己调函数。
"""
import sys, yaml
from pathlib import Path
from easydict import EasyDict

sys.path.insert(0, '/home/blancnight/OpenPCDet')
from pcdet.datasets.kitti.kitti_dataset import create_kitti_infos

CFG = '/home/blancnight/OpenPCDet/tools/cfgs/dataset_configs/kitti_pseudo_dataset.yaml'
ROOT = Path('/home/blancnight/OpenPCDet/data/kitti_pseudo')

cfg = EasyDict(yaml.safe_load(open(CFG)))
print('DATA_PATH in cfg :', cfg.DATA_PATH)
print('实际使用 data_path:', ROOT)
print()

create_kitti_infos(dataset_cfg=cfg,
                   class_names=['Car'],      # 只做 Car
                   data_path=ROOT,
                   save_path=ROOT)
