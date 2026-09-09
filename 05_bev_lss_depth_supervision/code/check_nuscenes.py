# -*- coding: utf-8 -*-
"""
解压后跑这个：一次查清路径层级、devkit 加载、以及深度投影管线是否真的对齐。

    python check_nuscenes.py            # 默认 ~/nuscenes, mini
    python check_nuscenes.py trainval

最后一项最重要：把 LiDAR 投到图像上画出来，人眼确认深度真值贴在物体上、
不是整体偏移。增强对齐错了 loss 照样降，只有看图能发现。
"""
import os, sys
import numpy as np

VER = sys.argv[1] if len(sys.argv) > 1 else 'mini'
ROOT = os.path.expanduser('~/nuscenes')
DR = os.path.join(ROOT, VER)
OUT = os.path.expanduser('~/lss_runs/depth_check')

print('=' * 60)
print('dataroot :', DR)

# ---------- ① 目录层级 ----------
need = ['samples', 'sweeps', 'v1.0-%s' % VER]
ok = True
for d in need:
    p = os.path.join(DR, d)
    if os.path.isdir(p):
        n = len(os.listdir(p))
        print('  OK   %-14s (%d 项)' % (d, n))
    else:
        print('  缺   %-14s <- 没有这个目录' % d)
        ok = False
if not ok:
    print()
    print('层级不对。最常见的是多套了一层，比如解成了')
    print('  %s/v1.0-mini/samples/...' % DR)
    print('应该是')
    print('  %s/samples/...' % DR)
    print('修法：mv %s/v1.0-mini/{samples,sweeps,maps} %s/ 之后再把 json 那层留在 v1.0-mini/' % (DR, DR))
    sys.exit(1)

# ---------- ② devkit 加载 ----------
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-%s' % VER, dataroot=DR, verbose=False)
print()
print('  场景数 %d   关键帧 %d' % (len(nusc.scene), len(nusc.sample)))

# ---------- ③ 深度投影对齐（关键）----------
sys.path.insert(0, os.path.expanduser('~/lift-splat-shoot'))
from src.data_depth import DepthSegData

grid_conf = {'xbound': [-50.0, 50.0, 0.5], 'ybound': [-50.0, 50.0, 0.5],
             'zbound': [-10.0, 10.0, 20.0], 'dbound': [4.0, 45.0, 1.0]}
data_aug_conf = {
    'resize_lim': (0.193, 0.225), 'final_dim': (128, 352),
    'rot_lim': (-5.4, 5.4), 'H': 900, 'W': 1600, 'rand_flip': True,
    'bot_pct_lim': (0.0, 0.22),
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 5,
}
ds = DepthSegData(nusc, is_train=False, data_aug_conf=data_aug_conf,
                  grid_conf=grid_conf, depth_ratio=1.0)
imgs, rots, trans, intrins, prots, ptrans, binimg, depth = ds[0]
print()
print('  imgs   ', tuple(imgs.shape), '  (相机数, 3, 128, 352)')
print('  depth  ', tuple(depth.shape), '  (相机数, 8, 22)')
print('  binimg ', tuple(binimg.shape))
cov = (depth > 0).float().mean().item()
print('  深度监督覆盖率 %.1f%% 的特征格子' % (cov * 100))
print('  深度范围 %.1f ~ %.1f m' % (depth[depth > 0].min(), depth[depth > 0].max()))

# ---------- ④ 稀疏度消融是否真的生效 ----------
print()
print('  --- 降采样比例 vs 覆盖率（应单调下降）---')
for r in [1.0, 0.5, 0.25, 0.1]:
    d2 = DepthSegData(nusc, is_train=False, data_aug_conf=data_aug_conf,
                      grid_conf=grid_conf, depth_ratio=r)
    dd = d2[0][7]
    print('    ratio %-5.2f  覆盖 %5.1f%%' % (r, (dd > 0).float().mean().item() * 100))

# ---------- ⑤ 画图人工核对 ----------
os.makedirs(OUT, exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
n = imgs.shape[0]
fig, axes = plt.subplots(2, n, figsize=(4 * n, 5))
for i in range(n):
    im = imgs[i].numpy().transpose(1, 2, 0) * std + mean
    axes[0, i].imshow(np.clip(im, 0, 1)); axes[0, i].axis('off')
    d = depth[i].numpy()
    axes[1, i].imshow(np.clip(im, 0, 1))
    ys, xs = np.nonzero(d)
    axes[1, i].scatter(xs * 16 + 8, ys * 16 + 8, c=d[ys, xs], s=14, cmap='jet')
    axes[1, i].axis('off')
p = os.path.join(OUT, 'depth_align.png')
plt.tight_layout(); plt.savefig(p, dpi=110); plt.close()
print()
print('=' * 60)
print('对齐图已存: %s' % p)
print('⚠️ 必须人眼看：深度点应当贴在车辆/建筑上，近处偏蓝(深度小)、远处偏红(深度大)，jet 色图低值为蓝。')
print('   若整体偏移或压在一角 -> 增强对齐错了，别开训练。')
