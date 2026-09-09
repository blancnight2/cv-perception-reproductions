# -*- coding: utf-8 -*-
"""
PointPillars 小目标优化：显存/速度冒烟测试。

在 12GB 上跑不了的配置，设计得再好也没用。先测清楚 pillar 尺寸与 batch 的可行组合，
再决定 Run 1 能同时上多大的杠杆。

每个组合只跑 30 个 iteration，报峰值显存和 s/iter。
"""
import os, sys, time, copy, argparse
import torch

sys.path.insert(0, os.path.expanduser('~/OpenPCDet'))
sys.path.insert(0, os.path.expanduser('~/OpenPCDet/tools'))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

CFG = os.path.expanduser('~/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml')
NITER = 30

# (pillar 尺寸, batch, 训练时最大 voxel 数)
# voxel 变细 -> 单帧 voxel 数暴涨，MAX_NUMBER_OF_VOXELS 不跟着提会丢点
COMBOS = [
    (0.16, 4,  16000),   # 基线，对照
    (0.16, 8,  16000),   # 只提 batch
    (0.16, 12, 16000),   # batch 更大
    (0.08, 4,  64000),   # 只细化分辨率（4x pillar）
    (0.08, 2,  64000),   # 细化 + 减半 batch 保命
]


def run(vox, bsz, max_vox, logger):
    c = copy.deepcopy(cfg)
    # --- 改 pillar 尺寸 ---
    for p in c.DATA_CONFIG.DATA_PROCESSOR:
        if p.NAME == 'transform_points_to_voxels':
            p.VOXEL_SIZE = [vox, vox, 4]
            p.MAX_NUMBER_OF_VOXELS = {'train': max_vox, 'test': max_vox * 2}
    c.MODEL.BACKBONE_3D = c.MODEL.get('BACKBONE_3D', None)

    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        loader_set = build_dataloader(
            dataset_cfg=c.DATA_CONFIG, class_names=c.CLASS_NAMES,
            batch_size=bsz, dist=False, workers=4, logger=logger, training=True)
        dataset, loader = loader_set[0], loader_set[1]
        model = build_network(model_cfg=c.MODEL, num_class=len(c.CLASS_NAMES),
                              dataset=dataset).cuda().train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)

        it = iter(loader)
        t0 = None
        for i in range(NITER):
            batch = next(it)
            from pcdet.models import load_data_to_gpu
            load_data_to_gpu(batch)
            ret, tb, disp = model(batch)
            loss = ret['loss'] if isinstance(ret, dict) else ret
            opt.zero_grad(); loss.backward(); opt.step()
            if i == 9:                      # 前 10 iter 是预热，不计时
                torch.cuda.synchronize(); t0 = time.time()
        torch.cuda.synchronize()
        spi = (time.time() - t0) / (NITER - 10)
        peak = torch.cuda.max_memory_allocated() / 1024**3
        grid = dataset.grid_size
        return dict(ok=True, mem=peak, spi=spi, grid=tuple(int(g) for g in grid))
    except RuntimeError as e:
        msg = str(e)
        return dict(ok=False, err='OOM' if 'out of memory' in msg.lower() else msg[:70])
    finally:
        torch.cuda.empty_cache()


def main():
    logger = common_utils.create_logger('/tmp/smoke_pp.log', rank=0)
    cfg_from_yaml_file(CFG, cfg)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print('=' * 72)
    print('GPU %s  %.1f GB' % (torch.cuda.get_device_name(0), total_gb))
    print('=' * 72)
    print('%-8s %-7s %-16s %-10s %-10s %s'
          % ('pillar', 'batch', 'BEV网格', '峰值显存', 's/iter', '80轮预计'))
    print('-' * 72)
    rows = []
    for vox, bsz, mv in COMBOS:
        r = run(vox, bsz, mv, logger)
        if r['ok']:
            # 3712 帧 / batch = iter/epoch
            hrs = r['spi'] * (3712 / bsz) * 80 / 3600
            print('%-8.2f %-7d %-16s %-10.2f %-10.3f %.1f 小时'
                  % (vox, bsz, '%dx%d' % r['grid'][:2], r['mem'], r['spi'], hrs))
            rows.append((vox, bsz, r['mem'], r['spi'], hrs))
        else:
            print('%-8.2f %-7d %-16s %s' % (vox, bsz, '-', r['err']))
        sys.stdout.flush()

    print()
    print('=' * 72)
    fit = [r for r in rows if r[2] < total_gb * 0.85]
    if fit:
        print('可用组合（显存 < 85%%）：')
        for vox, bsz, mem, spi, hrs in fit:
            print('  pillar %.2f / batch %-3d -> %.1f GB, 80 轮约 %.1f 小时' % (vox, bsz, mem, hrs))
    else:
        print('没有组合能安全放下，需要进一步降 batch 或缩小点云范围')


if __name__ == '__main__':
    main()
