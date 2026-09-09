# -*- coding: utf-8 -*-
"""
在 3D 标准划分的 train(3712 帧) 上跑 YOLOE 零样本 2D 检测，产出 preds_*.jsonl
供 gen_pseudo.py --src yoloe 使用。

为什么必须补跑：现有 YOLOE 缓存是在 **2D 检测的自建划分**(5985/1496) 上跑的，
只有 713 帧落在 3D train(3712) 里，缺 2999 帧。两套划分不能混用。

⚠️ 关键：一次推理、多阈值导出。
   AP 最优阈值(0.0001) 对伪标签是错的——AP 不惩罚排在末尾的低分误检，
   但伪标签里**每个误检都会变成一个错误的训练目标**。所以要偏向精确率。
   本脚本按 conf 0.05 推理一次，导出多个阈值的 jsonl，后面离线比质量再决定训哪个。

用法：python yoloe_on_train.py
"""
import os, json, glob, time, argparse

ap = argparse.ArgumentParser()
ap.add_argument('--split', default='train')
ap.add_argument('--base-conf', type=float, default=0.05, help='推理阈值，取最低的那档')
ap.add_argument('--thr', default='0.05,0.15,0.30', help='导出阈值，逗号分隔')
ap.add_argument('--limit', type=int, default=0)
args = ap.parse_args()

KITTI = os.path.expanduser('~/OpenPCDet/data/kitti')
IMG = os.path.join(KITTI, 'training/image_2')
OUTDIR = os.path.expanduser('~/openvocab')

# V5 提示词组：消融里最强的单变体（全量 mAP50 0.4374）
PROMPTS = ['parked or moving car',
           'standing or walking person',
           'person on a bike']

ids = [l.strip() for l in open(os.path.join(KITTI, 'ImageSets/%s.txt' % args.split))
       if l.strip()]
if args.limit:
    ids = ids[:args.limit]
print('划分 %s：%d 帧' % (args.split, len(ids)))

from ultralytics import YOLOE
model = YOLOE('yoloe-11l-seg.pt')
model.set_classes(PROMPTS, model.get_text_pe(PROMPTS))
print('提示词：%s' % PROMPTS)

thrs = sorted(float(x) for x in args.thr.split(','))
fs = {t: open(os.path.join(OUTDIR, 'preds_YOLOE-%s-conf%g.jsonl'
                           % (args.split, t)), 'w') for t in thrs}
cnt = {t: 0 for t in thrs}       # Car 框数
frames_with_car = {t: 0 for t in thrs}

t0 = time.time()
for i, fid in enumerate(ids, 1):
    p = os.path.join(IMG, fid + '.png')
    if not os.path.exists(p):
        continue
    r = model.predict(p, conf=args.base_conf, verbose=False)[0]
    H, W = r.orig_shape
    rows = []
    if r.boxes is not None and len(r.boxes):
        for xyxy, cf, cl in zip(r.boxes.xyxy.tolist(),
                                r.boxes.conf.tolist(),
                                r.boxes.cls.tolist()):
            cid = int(cl)
            if cid > 2:
                continue
            rows.append((xyxy, float(cf), cid))

    for t in thrs:
        keep = [(b, s, c) for b, s, c in rows if s >= t]
        rec = {'stem': fid, 'W': W, 'H': H,
               'boxes': [b for b, _, _ in keep],
               'scores': [s for _, s, _ in keep],
               'labels': [c for _, _, c in keep]}
        fs[t].write(json.dumps(rec) + '\n')
        nc = sum(1 for _, _, c in keep if c == 0)
        cnt[t] += nc
        if nc:
            frames_with_car[t] += 1

    if i % 500 == 0:
        el = time.time() - t0
        print('  %4d/%d  %.1f ms/图  剩 %.1f 分'
              % (i, len(ids), el / i * 1000, (len(ids) - i) * el / i / 60))

for f in fs.values():
    f.close()

# ---- 与真值 2D 框数对比，先看个量级 ----
def gt_car_2d(fid):
    p = os.path.join(KITTI, 'training/label_2', fid + '.txt')
    n = 0
    if os.path.exists(p):
        for line in open(p):
            v = line.split()
            if v and v[0] == 'Car':
                n += 1
    return n


gt_total = sum(gt_car_2d(f) for f in ids)

print()
print('=' * 72)
print('%d 帧完成，耗时 %.1f 分钟（%.1f ms/图）'
      % (len(ids), (time.time() - t0) / 60, (time.time() - t0) / len(ids) * 1000))
print('=' * 72)
print('真值 Car 2D 框: %d 个（分母参照）' % gt_total)
print()
print('%-8s %12s %10s %12s' % ('conf', 'Car框数', '占真值%', '有车的帧数'))
print('-' * 48)
for t in thrs:
    print('%-8g %12d %9.1f%% %12d'
          % (t, cnt[t], cnt[t] / max(gt_total, 1) * 100, frames_with_car[t]))
print('=' * 72)
print('产出：')
for t in thrs:
    print('  ~/openvocab/preds_YOLOE-%s-conf%g.jsonl' % (args.split, t))
print()
print('注：框数接近真值不等于质量好（可能又漏又误检）。')
print('    下一步用 gen_pseudo.py 生成 3D 伪标签，再离线比 BEV IoU 精确率/召回率，')
print('    选定一个阈值再训练——避免训 3 次。')
