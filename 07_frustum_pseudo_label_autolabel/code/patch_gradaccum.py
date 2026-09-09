# -*- coding: utf-8 -*-
"""
给 OpenPCDet 的训练循环加梯度累积。

为什么需要：OpenPCDet 公布的 KITTI 成绩是 8 卡 × batch 4 = 有效 batch 32 训出来的，
单卡 12GB 最多放到 batch 8。梯度累积让单卡也能复刻有效 batch 32，
显存与耗时都不变——这是把「训练条件」这个变量对齐的唯一办法。

用环境变量 GRAD_ACCUM 控制，默认 1 = 完全保持原行为。
原文件备份为 train_utils.py.bak。
"""
import os, shutil

F = os.path.expanduser('~/OpenPCDet/tools/train_utils/train_utils.py')

OLD = """        model.train()
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            loss, tb_dict, disp_dict = model_func(model, batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)"""

NEW = """        model.train()
        # --- 梯度累积（GRAD_ACCUM=1 时与原版逐字等价）---
        _acc = int(os.environ.get('GRAD_ACCUM', '1'))
        if cur_it % _acc == 0:
            optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            loss, tb_dict, disp_dict = model_func(model, batch)

        # 除以累积步数，使等效梯度＝大 batch 的平均梯度而非求和
        scaler.scale(loss / _acc).backward()
        if (cur_it + 1) % _acc != 0:
            # 未到累积边界：只攒梯度，不做 unscale/clip/step
            accumulated_iter += 1
            continue
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)"""

src = open(F, encoding='utf-8').read()

if 'GRAD_ACCUM' in src:
    print('已经打过补丁，跳过')
else:
    if OLD not in src:
        raise SystemExit('!! 找不到待替换片段，上游代码可能变了')
    if not os.path.exists(F + '.bak'):
        shutil.copyfile(F, F + '.bak')
        print('已备份 train_utils.py.bak')
    src = src.replace(OLD, NEW, 1)
    if '\nimport os' not in src and 'import os\n' not in src.split('def ')[0]:
        src = 'import os\n' + src
        print('补了 import os')
    open(F, 'w', encoding='utf-8').write(src)
    print('补丁已打')

# --- 自检 ---
import ast
ast.parse(open(F, encoding='utf-8').read())
print('语法 OK')
s = open(F, encoding='utf-8').read()
for k in ['GRAD_ACCUM', 'loss / _acc', 'continue']:
    print('  含 %-14s %s' % (k, 'YES' if k in s else 'NO'))
print()
print('⚠️ continue 之前已 accumulated_iter += 1，与原版计数一致；')
print('   lr_scheduler 仍按 micro-batch 推进，80 epoch 的 LR 曲线形状不变。')
