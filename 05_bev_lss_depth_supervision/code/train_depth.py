# -*- coding: utf-8 -*-
"""
深度监督稀疏度消融的训练脚本。

    python train_depth.py --version=mini --dataroot=~/nuscenes --depth-ratio=0.25

唯一的自变量是 --depth-ratio ∈ {0, 0.1, 0.25, 0.5, 1.0}：
    0    = 原版 LSS（隐式深度，无监督），基线
    >0   = 用降采样到该比例的 LiDAR 点监督深度分布

⚠️ 为保证各组可比，以下全部固定：随机种子、数据顺序、增强、超参、训练步数。
   验证集深度真值一律用 100%（评的是 BEV 分割，深度只作诊断）。
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models import compile_model
from src.tools import SimpleLoss, get_batch_iou
from src.data_depth import compile_depth_data
from src.depth_sup import depth_loss, depth_metrics

ap = argparse.ArgumentParser()
ap.add_argument('--version', default='mini', choices=['mini', 'trainval'])
ap.add_argument('--dataroot', default=os.path.expanduser('~/nuscenes'))
ap.add_argument('--beams', type=int, required=True)
ap.add_argument('--depth-weight', type=float, default=1.0)
ap.add_argument('--nepochs', type=int, default=20)
ap.add_argument('--bsz', type=int, default=4)
ap.add_argument('--nworkers', type=int, default=6)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--weight-decay', type=float, default=1e-7)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--gpuid', type=int, default=0)
ap.add_argument('--logdir', default=None)
args = ap.parse_args()

TAG = 'b%d' % args.beams
LOGDIR = args.logdir or os.path.expanduser('~/lss_runs/%s' % TAG)
os.makedirs(LOGDIR, exist_ok=True)

# ---- 固定一切随机性，各组唯一差别只能是 depth_ratio ----
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

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

trainloader, valloader = compile_depth_data(
    args.version, args.dataroot, data_aug_conf, grid_conf,
    bsz=args.bsz, nworkers=args.nworkers, n_beams=args.beams)

device = torch.device('cuda:%d' % args.gpuid)
model = compile_model(grid_conf, data_aug_conf, outC=1).to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
seg_loss_fn = SimpleLoss(2.13).to(device)

print('=' * 62)
print('激光线数     : %s线%s' % (TAG, '（基线：无深度监督）' if args.beams <= 0 else ''))
print('训练集 %d 批 / 验证集 %d 批   epoch %d   bsz %d'
      % (len(trainloader), len(valloader), args.nepochs, args.bsz))
print('=' * 62)


def evaluate():
    model.eval()
    ti = un = 0.0
    sl = dm = dh = n = 0.0
    with torch.no_grad():
        for imgs, rots, trans, intrins, prots, ptrans, binimgs, dgt in valloader:
            preds, dpred = model(imgs.to(device), rots.to(device), trans.to(device),
                                 intrins.to(device), prots.to(device), ptrans.to(device))
            binimgs = binimgs.to(device)
            sl += seg_loss_fn(preds, binimgs).item()
            i, u, _ = get_batch_iou(preds, binimgs)
            ti += i; un += u
            mae, hit = depth_metrics(dpred, dgt.to(device), grid_conf['dbound'])
            if mae == mae:          # 非 NaN
                dm += mae; dh += hit; n += 1
    model.train()
    return {'iou': float(ti) / max(float(un), 1.0),
            'seg_loss': sl / max(len(valloader), 1),
            'depth_mae': dm / max(n, 1), 'depth_hit': dh / max(n, 1)}


best = -1.0
hist = []
step = 0
for epoch in range(args.nepochs):
    np.random.seed(args.seed * 1000 + epoch)       # 每组同 epoch 用同一份增强
    t0 = time.time()
    for imgs, rots, trans, intrins, prots, ptrans, binimgs, dgt in trainloader:
        opt.zero_grad()
        preds, dpred = model(imgs.to(device), rots.to(device), trans.to(device),
                             intrins.to(device), prots.to(device), ptrans.to(device))
        binimgs = binimgs.to(device)
        ls = seg_loss_fn(preds, binimgs)
        if args.beams > 0:
            ld = depth_loss(dpred, dgt.to(device), grid_conf['dbound'])
            loss = ls + args.depth_weight * ld
        else:
            ld = torch.zeros((), device=device)
            loss = ls
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        step += 1
        if step % 50 == 0:
            print('  step %-6d seg %.4f  depth %.4f' % (step, ls.item(), ld.item()), flush=True)

    v = evaluate()
    hist.append(dict(epoch=epoch, **v))
    print('[epoch %2d] val IoU %.4f | seg_loss %.4f | depth MAE %.2f m | bin命中 %.3f | %.1f 分钟'
          % (epoch, v['iou'], v['seg_loss'], v['depth_mae'], v['depth_hit'],
             (time.time() - t0) / 60), flush=True)
    if v['iou'] > best:
        best = v['iou']
        torch.save(model.state_dict(), os.path.join(LOGDIR, 'best.pt'))
    json.dump({'n_beams': args.beams, 'best_iou': best, 'hist': hist},
              open(os.path.join(LOGDIR, 'result.json'), 'w'), indent=2)

print('=' * 62)
print('线数=%s   最佳 val BEV 分割 IoU = %.4f' % (TAG, best))
print('结果 -> %s/result.json' % LOGDIR)
