# -*- coding: utf-8 -*-
"""带激光深度真值的 nuScenes 数据集（按线数降采样版）。

v2 改动：自变量从「随机丢点比例」换成「激光线数」。
  原因：特征网格仅 8×22=176 格/相机，min-pool 后每格一个点就算有监督，
  随机丢 75% 的点覆盖率只从 97.7% 掉到 95.0% —— 三个实验组实际上是同一个。
  按线数降采样是结构化稀疏（整条仰角环消失），覆盖率 93.3%→54.0%→24.9%，
  且直接对应「32 线 vs 8 线」的真实传感器选型。

⚠️ 深度真值必须用 get_image_data 实际返回的 post_rots/post_trans 来对齐——
   sample_augmentation() 每张图随机，不能事后重算一遍。
"""
import numpy as np
import torch

from .data import NuscData
from .beam_sub import load_lidar_beams
from .depth_sup import lidar_to_depth


class DepthSegData(NuscData):
    def __init__(self, *args, n_beams=32, downsample=16, **kwargs):
        super(DepthSegData, self).__init__(*args, **kwargs)
        self.n_beams = n_beams
        self.downsample = downsample
        fH, fW = self.data_aug_conf['final_dim']
        self.fH, self.fW = fH // downsample, fW // downsample

    def get_depth_gt(self, rec, ncam, rots, trans, intrins, post_rots, post_trans):
        out = np.zeros((ncam, self.fH, self.fW), dtype=np.float32)
        if self.n_beams <= 0:
            return torch.from_numpy(out)          # 基线组：全 0 = 无深度监督
        pts_ego = load_lidar_beams(self.nusc, rec, self.n_beams)
        if len(pts_ego) == 0:
            return torch.from_numpy(out)
        for i in range(ncam):
            out[i] = lidar_to_depth(
                pts_ego,
                rots[i].numpy().astype(np.float64),
                trans[i].numpy().astype(np.float64),
                intrins[i].numpy().astype(np.float64),
                post_rots[i].numpy().astype(np.float64),
                post_trans[i].numpy().astype(np.float64),
                self.fH, self.fW, self.downsample,
                self.grid_conf['dbound'])
        return torch.from_numpy(out)

    def __getitem__(self, index):
        rec = self.ixes[index]
        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        binimg = self.get_binimg(rec)
        depth = self.get_depth_gt(rec, len(cams), rots, trans,
                                  intrins, post_rots, post_trans)
        return imgs, rots, trans, intrins, post_rots, post_trans, binimg, depth


def compile_depth_data(version, dataroot, data_aug_conf, grid_conf, bsz,
                       nworkers, n_beams):
    """与原版 compile_data 同构，只是数据集换成 DepthSegData。
    ⚠️ 验证集固定用 32 线满配——评的是 BEV 分割，深度只作诊断，
       各组必须用同一份验证深度才可比。"""
    from nuscenes.nuscenes import NuScenes
    import os
    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=os.path.join(dataroot, version), verbose=False)
    traindata = DepthSegData(nusc, is_train=True, data_aug_conf=data_aug_conf,
                             grid_conf=grid_conf, n_beams=n_beams)
    valdata = DepthSegData(nusc, is_train=False, data_aug_conf=data_aug_conf,
                           grid_conf=grid_conf, n_beams=32)

    def _wi(x):
        np.random.seed(13 + x)

    trainloader = torch.utils.data.DataLoader(
        traindata, batch_size=bsz, shuffle=True, num_workers=nworkers,
        drop_last=True, worker_init_fn=_wi)
    valloader = torch.utils.data.DataLoader(
        valdata, batch_size=bsz, shuffle=False, num_workers=nworkers)
    return trainloader, valloader
