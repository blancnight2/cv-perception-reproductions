# -*- coding: utf-8 -*-
"""
固定随机子集重算 —— 让 Qwen3.6 只跑 N 张也能与已有 5 个模型同协议比较。

原理：所有模型的逐图预测都缓存在 ~/openvocab/preds_*.jsonl，
      把它们限制到同一批 stem 上重算指标即可，无需重跑任何模型。

用法：
    python subset_eval.py --n 300                 # 生成子集 + 重算全部已缓存模型
    python subset_eval.py --n 300 --export /root/kitti_sub   # 顺便导出这 300 张图
"""
import os, sys, json, glob, random, argparse, shutil
import torch

sys.path.insert(0, os.path.expanduser('~/openvocab'))
from eval_zeroshot import load_gt, NAMES, VAL_IMG          # noqa: E402
from torchmetrics.detection.mean_ap import MeanAveragePrecision  # noqa: E402

SUBSET_F = os.path.expanduser('~/openvocab/subset_ids.json')

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=300)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--export', default=None, help='把子集图片复制到该目录（传给 AutoDL 用）')
ap.add_argument('--only', default=None, help='只算某个模型（preds_<名字>.jsonl 里的名字）')
args = ap.parse_args()

# ---------- 1) 固定随机子集（种子写死，可复现） ----------
if os.path.exists(SUBSET_F):
    sub = json.load(open(SUBSET_F))
    print('复用已有子集 %s（%d 张，seed=%s）' % (SUBSET_F, len(sub['stems']), sub['seed']))
    stems = set(sub['stems'])
else:
    allimg = sorted(glob.glob(os.path.join(VAL_IMG, '*')))
    allstem = [os.path.splitext(os.path.basename(p))[0] for p in allimg]
    rng = random.Random(args.seed)
    picked = sorted(rng.sample(allstem, min(args.n, len(allstem))))
    json.dump({'seed': args.seed, 'n': len(picked), 'stems': picked},
              open(SUBSET_F, 'w'), indent=1)
    stems = set(picked)
    print('新建子集：从 %d 张里随机抽 %d 张（seed=%d）-> %s'
          % (len(allstem), len(picked), args.seed, SUBSET_F))

# ---------- 2) 可选：导出子集图片 ----------
if args.export:
    os.makedirs(args.export, exist_ok=True)
    n = 0
    for p in sorted(glob.glob(os.path.join(VAL_IMG, '*'))):
        if os.path.splitext(os.path.basename(p))[0] in stems:
            shutil.copy2(p, args.export); n += 1
    sz = sum(os.path.getsize(os.path.join(args.export, f))
             for f in os.listdir(args.export)) / 1e6
    print('已导出 %d 张到 %s（%.0f MB）' % (n, args.export, sz))

# ---------- 3) 逐模型在子集上重算 ----------
rows = []
files = sorted(glob.glob(os.path.expanduser('~/openvocab/preds_*.jsonl')))
for f in files:
    name = os.path.basename(f)[len('preds_'):-len('.jsonl')]
    if args.only and args.only != name:
        continue
    if 'smoke' in name or 'cache-test' in name:
        continue

    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox',
                                  class_metrics=True, backend='faster_coco_eval')
    seen = set()
    for line in open(f):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        s = r['stem']
        if s not in stems or s in seen:
            continue
        seen.add(s)
        preds = [dict(
            boxes=torch.tensor(r['boxes'], dtype=torch.float32).reshape(-1, 4),
            scores=torch.tensor(r['scores'], dtype=torch.float32),
            labels=torch.tensor(r['labels'], dtype=torch.long))]
        gb, gl = load_gt(s, r['W'], r['H'])
        metric.update(preds, [dict(boxes=gb, labels=gl)])

    if not seen:
        print('跳过 %-28s（子集内无缓存）' % name)
        continue
    res = metric.compute()
    per = res.get('map_per_class')
    per = per.tolist() if (per is not None and per.numel() > 1) else [float('nan')] * 3
    rows.append((name, len(seen), res['map_50'].item(), res['map'].item(), per))
    print('算完 %-28s  覆盖 %d/%d 张' % (name, len(seen), len(stems)))

# ---------- 4) 输出对比表 ----------
rows.sort(key=lambda r: -r[2])
print()
print('=' * 88)
print('固定随机子集 %d 张（seed=%d）—— 同协议对比' % (len(stems), args.seed))
print('=' * 88)
print('%-30s %6s %8s %10s %8s %8s %8s' %
      ('模型', '张数', 'mAP50', 'mAP50-95', 'Car', 'Ped', 'Cyc'))
print('-' * 88)
for name, n, m50, m, per in rows:
    p = (list(per) + [float('nan')] * 3)[:3]
    print('%-30s %6d %8.4f %10.4f %8.4f %8.4f %8.4f' % (name, n, m50, m, *p))
print('=' * 88)
print('⚠️ 覆盖张数不等于子集大小的行，说明该模型没跑满，数字不可直接比')

json.dump([{'model': r[0], 'n': r[1], 'map_50': r[2], 'map': r[3],
            'per_class': r[4]} for r in rows],
          open(os.path.expanduser('~/openvocab/subset_result.json'), 'w'),
          indent=2, ensure_ascii=False)
print('已存 ~/openvocab/subset_result.json')
