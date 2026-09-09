# -*- coding: utf-8 -*-
"""
按激光线数降采样（替代原来的随机丢点）。

为什么换：特征网格只有 8×22=176 格/相机，而一帧激光约 3 万点，min-pool 后
每格只要落到一个点就算有监督 —— 随机丢掉 75% 的点，覆盖率只从 97.7% 掉到 95.0%，
1.0/0.5/0.25 三组几乎是同一个实验。

按线数降采样产生的是**结构化稀疏**（整条仰角环消失），既对应真实的
「32 线 vs 16 线 vs 8 线」传感器选型，覆盖率也会真正塌下去。

nuScenes 的 .pcd.bin 每点 5 个 float：x, y, z, intensity, ring_index（0-31）。
devkit 的 LidarPointCloud.from_file 只取前 4 维会把 ring 丢掉，所以这里自己读。
"""
import os
import numpy as np
from pyquaternion import Quaternion

N_RINGS = 32          # nuScenes 用 32 线激光


def load_lidar_beams(nusc, rec, n_beams):
    """读关键帧点云，按线数降采样，返回 ego 系 (N,3)。

    n_beams: 32 / 16 / 8 / 4 ...；<=0 表示不要深度监督
    """
    if n_beams <= 0:
        return np.zeros((0, 3))

    sd = nusc.get('sample_data', rec['data']['LIDAR_TOP'])
    path = os.path.join(nusc.dataroot, sd['filename'])
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)   # x,y,z,intensity,ring

    if n_beams < N_RINGS:
        # 均匀抽环：32->16 取偶数环，32->8 取每 4 环，以此类推
        step = N_RINGS // n_beams
        pts = pts[(pts[:, 4].astype(np.int32) % step) == 0]

    xyz = pts[:, :3].astype(np.float64)

    # 去掉打在自车上的点（与 devkit 的 min_distance=2.2 一致）
    d = np.hypot(xyz[:, 0], xyz[:, 1])
    xyz = xyz[d >= 2.2]

    # lidar -> ego（单帧时 get_lidar_data 的变换链就退化成这一步）
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    R = Quaternion(cs['rotation']).rotation_matrix
    t = np.array(cs['translation'], dtype=np.float64)
    return xyz @ R.T + t[None, :]
