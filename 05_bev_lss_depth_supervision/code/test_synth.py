# -*- coding: utf-8 -*-
"""
不依赖 nuScenes 的合成测试：把投影数学和增强对齐单独验掉。

这两处是整个实验最容易错、又最难发现的地方——错了 loss 照样降。
用解析可算的例子先钉死，等真数据到了只剩「看图确认」一件事。
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser('~/lift-splat-shoot'))
from src.depth_sup import lidar_to_depth, depth_loss, depth_metrics

DB = [4.0, 45.0, 1.0]        # dbound
DS = 16                       # downsample
fH, fW = 128 // DS, 352 // DS  # 8 x 22
F, CX, CY = 500.0, 176.0, 64.0
INTRIN = np.array([[F, 0, CX], [0, F, CY], [0, 0, 1.0]])
I3, Z3 = np.eye(3), np.zeros(3)
fail = 0


def chk(name, cond, extra=''):
    global fail
    print('  %-46s %s %s' % (name, 'PASS' if cond else '**FAIL**', extra))
    if not cond:
        fail += 1


print('=' * 68)
print('① 投影数学：相机在 ego 原点、无旋转，点应落在主点所在格子')
# 相机系 (0,0,10) -> 像素 (CX,CY) -> 格子 (CY//16, CX//16) = (4,11)
pts = np.array([[0.0, 0.0, 10.0]])
d = lidar_to_depth(pts, I3, Z3, INTRIN, np.eye(3), np.zeros(3), fH, fW, DS, DB)
chk('落在格子 (4,11)', d[4, 11] == 10.0, '实际非零格 %s' % np.argwhere(d > 0).tolist())
chk('其余格子为 0', (d > 0).sum() == 1)

print()
print('② 偏移点：相机系 (2, 0, 10) -> u = 500*2/10 + 176 = 276 -> 列 276//16 = 17')
pts = np.array([[2.0, 0.0, 10.0]])
d = lidar_to_depth(pts, I3, Z3, INTRIN, np.eye(3), np.zeros(3), fH, fW, DS, DB)
chk('落在格子 (4,17)', d[4, 17] == 10.0, '实际 %s' % np.argwhere(d > 0).tolist())

print()
print('③ 增强对齐：post_rot=0.5I -> 像素减半 -> (88,32) -> 格子 (2,5)')
pr = np.eye(3) * 0.5; pr[2, 2] = 1.0
pts = np.array([[0.0, 0.0, 10.0]])
d = lidar_to_depth(pts, I3, Z3, INTRIN, pr, np.zeros(3), fH, fW, DS, DB)
chk('缩放后落在 (2,5)', d[2, 5] == 10.0, '实际 %s' % np.argwhere(d > 0).tolist())

print()
print('④ 增强平移：post_tran=(+32,+16) -> (208,80) -> 格子 (5,13)')
d = lidar_to_depth(pts, I3, Z3, INTRIN, np.eye(3), np.array([32.0, 16.0, 0.0]),
                   fH, fW, DS, DB)
chk('平移后落在 (5,13)', d[5, 13] == 10.0, '实际 %s' % np.argwhere(d > 0).tolist())

print()
print('⑤ 相机后方的点必须丢弃（z<0）')
d = lidar_to_depth(np.array([[0.0, 0.0, -10.0]]), I3, Z3, INTRIN,
                   np.eye(3), np.zeros(3), fH, fW, DS, DB)
chk('身后的点被剔除', (d > 0).sum() == 0)

print()
print('⑥ dbound 之外的点必须丢弃（<4m 或 >=45m）')
for z, why in [(2.0, '太近'), (60.0, '太远')]:
    d = lidar_to_depth(np.array([[0.0, 0.0, z]]), I3, Z3, INTRIN,
                       np.eye(3), np.zeros(3), fH, fW, DS, DB)
    chk('%s (%.0fm) 被剔除' % (why, z), (d > 0).sum() == 0)

print()
print('⑦ 同格子取最近的面（min-pool）')
pts = np.array([[0.0, 0.0, 30.0], [0.0, 0.0, 10.0], [0.0, 0.0, 20.0]])
d = lidar_to_depth(pts, I3, Z3, INTRIN, np.eye(3), np.zeros(3), fH, fW, DS, DB)
chk('保留最近的 10m 而非 30m', d[4, 11] == 10.0, '实际 %.1f' % d[4, 11])

print()
print('⑧ 外参真的生效：相机沿 ego x 平移 2m，点应反向偏移')
# tran=(2,0,0) 表示相机原点在 ego 的 x=2 处；ego 原点前方的点在相机系里 x=-2
pts = np.array([[0.0, 0.0, 10.0]])
d = lidar_to_depth(pts, I3, np.array([2.0, 0.0, 0.0]), INTRIN,
                   np.eye(3), np.zeros(3), fH, fW, DS, DB)
nz = np.argwhere(d > 0).tolist()
chk('外参改变了落点', nz != [[4, 11]] and len(nz) >= 0, '实际 %s' % nz)

print()
print('⑨ 稀疏度消融：覆盖率应随 ratio 单调下降')
rng = np.random.RandomState(0)
big = np.stack([rng.uniform(-8, 8, 4000), rng.uniform(-3, 3, 4000),
                rng.uniform(5, 40, 4000)], 1)
prev, mono = 1e9, True
for r in [1.0, 0.5, 0.25, 0.1]:
    d = lidar_to_depth(big, I3, Z3, INTRIN, np.eye(3), np.zeros(3),
                       fH, fW, DS, DB, ratio=r, rng=np.random.RandomState(0))
    c = (d > 0).mean()
    print('    ratio %-5.2f 覆盖 %5.1f%%' % (r, c * 100))
    if c > prev:
        mono = False
    prev = c
chk('覆盖率单调下降', mono)

print()
print('⑩ 损失与诊断指标')
B, N, D = 2, 5, 41
pred = torch.softmax(torch.randn(B, N, D, fH, fW), dim=2)
gt = torch.zeros(B, N, fH, fW)
gt[0, 0, 4, 11] = 10.0
gt[1, 2, 3, 7] = 25.0
l = depth_loss(pred, gt, DB)
mae, hit = depth_metrics(pred, gt, DB)
chk('loss 有限且为正', torch.isfinite(l) and l.item() > 0, 'loss=%.4f' % l.item())
chk('MAE 有限', np.isfinite(mae), 'MAE=%.2f m  bin命中=%.3f' % (mae, hit))
chk('全零真值时 loss=0', depth_loss(pred, torch.zeros_like(gt), DB).item() == 0.0)

# 完美预测应给出接近 0 的 loss
perfect = torch.full((B, N, D, fH, fW), 1e-8)
perfect[0, 0, 6, 4, 11] = 1.0        # 10m -> bin (10-4)/1 = 6
perfect[1, 2, 21, 3, 7] = 1.0        # 25m -> bin 21
lp = depth_loss(perfect, gt, DB)
chk('完美预测 loss≈0', lp.item() < 1e-3, 'loss=%.2e' % lp.item())

print()
print('⑪ 打过补丁的模型：forward 应返回 (BEV, depth)')
from src.models import compile_model
gc = {'xbound': [-50., 50., 0.5], 'ybound': [-50., 50., 0.5],
      'zbound': [-10., 10., 20.], 'dbound': DB}
ac = {'final_dim': (128, 352), 'cams': ['a'] * 6, 'Ncams': 5,
      'resize_lim': (0.193, 0.225), 'rot_lim': (-5.4, 5.4),
      'H': 900, 'W': 1600, 'rand_flip': True, 'bot_pct_lim': (0., 0.22)}
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
m = compile_model(gc, ac, outC=1).to(dev).eval()
Bx, Nx = 1, 5
with torch.no_grad():
    out, dep = m(torch.randn(Bx, Nx, 3, 128, 352).to(dev),
                 torch.eye(3).repeat(Bx, Nx, 1, 1).to(dev),
                 torch.zeros(Bx, Nx, 3).to(dev),
                 torch.tensor(INTRIN, dtype=torch.float32).repeat(Bx, Nx, 1, 1).to(dev),
                 torch.eye(3).repeat(Bx, Nx, 1, 1).to(dev),
                 torch.zeros(Bx, Nx, 3).to(dev))
chk('BEV 输出 (1,1,200,200)', tuple(out.shape) == (Bx, 1, 200, 200), str(tuple(out.shape)))
chk('depth 输出 (1,5,41,8,22)', tuple(dep.shape) == (Bx, Nx, 41, fH, fW), str(tuple(dep.shape)))
chk('depth 是概率分布（沿 D 和为 1）',
    torch.allclose(dep.sum(2), torch.ones_like(dep.sum(2)), atol=1e-3))
if dev == 'cuda':
    print('    推理显存 %.2f GB' % (torch.cuda.max_memory_allocated() / 1e9))

print()
print('=' * 68)
print('全部通过' if fail == 0 else '**有 %d 项失败，别开训练**' % fail)
sys.exit(1 if fail else 0)
