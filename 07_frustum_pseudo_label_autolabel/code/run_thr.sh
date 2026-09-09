#!/bin/bash
# 三个 conf 阈值各生成一份 3D 伪标签并离线评质量，只训最好的那个。
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pcdet
cd ~/autolabel || exit 1

SRC=~/openvocab/preds_yoloe-train3712.jsonl

for THR in 0.10 0.20 0.30; do
  TAG=yoloe-conf${THR}
  F=~/openvocab/preds_${TAG}.jsonl
  echo ""
  echo "############ conf=$THR ############"

  # 按分数过滤（load_yoloe_boxes 不看 score，必须先切好）
  python - "$SRC" "$F" "$THR" <<'EOF'
import json, sys
src, dst, thr = sys.argv[1], sys.argv[2], float(sys.argv[3])
n_in = n_out = 0
with open(dst, 'w') as w:
    for line in open(src):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        b, s, l = [], [], []
        for bb, ss, ll in zip(r['boxes'], r['scores'], r['labels']):
            n_in += 1
            if ss >= thr:
                b.append(bb); s.append(ss); l.append(ll)
        n_out += len(b)
        w.write(json.dumps({'stem': r['stem'], 'W': r['W'], 'H': r['H'],
                            'boxes': b, 'scores': s, 'labels': l}) + "\n")
print('  过滤 %d -> %d 个框' % (n_in, n_out))
EOF

  OUT=~/autolabel/pseudo_${TAG}_train
  rm -rf "$OUT"
  python -u gen_pseudo.py --src yoloe --yoloe-preds "$F" --out "$OUT" 2>&1 | tail -4
  echo "--- 3D 伪标签质量 ---"
  python -u val_pseudo_quality.py "$OUT" 2>&1 | tail -18
done

echo ""
echo "############ 基线：GT 2D 框（弱监督设定，已训过 3D AP 16.44）############"
python -u val_pseudo_quality.py ~/autolabel/pseudo_gt_train 2>&1 | tail -18
