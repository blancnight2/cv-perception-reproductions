#!/bin/bash
# 深度监督消融：5 档串行，唯一自变量是激光线数，别的一律不动。
#
#   bash run_ablation.sh mini 20        # 管线验证（小数据，快）
#   bash run_ablation.sh trainval 12    # 正式实验
#
# 档位：0(无监督基线) / 4 / 8 / 16 / 32(原生满配)
# 对应特征格覆盖率约：0% / 24.9% / 54.0% / 89.2% / 93.3%
#
# 结果落在 ~/lss_runs/b<线数>/result.json，跑完自动汇总成表。
set -u
# 环境自适应：本地是 conda 环境 bev；AutoDL 上依赖装在 base。
# 激活失败就沿用当前环境，不要中断。
for c in ~/miniconda3 /root/miniconda3 /opt/conda; do
    [ -f "$c/etc/profile.d/conda.sh" ] && . "$c/etc/profile.d/conda.sh" && break
done
conda activate bev 2>/dev/null || true
cd ~/lift-splat-shoot || exit 1

VER=${1:-mini}
EPOCHS=${2:-20}
BEAMS="0 4 8 16 32"

echo "数据集 v1.0-$VER   每档 $EPOCHS epoch   线数档位: $BEAMS"
echo "起始 $(date)"

for B in $BEAMS; do
    DIR=~/lss_runs/b$B
    if [ -f "$DIR/result.json" ]; then
        echo ">>> ${B}线 已有结果，跳过（要重跑先删 $DIR）"
        continue
    fi
    echo ""
    echo "################ 激光线数 = $B ################"
    python -u train_depth.py \
        --version "$VER" \
        --dataroot ~/nuscenes \
        --beams "$B" \
        --nepochs "$EPOCHS" \
        --seed 0 \
        2>&1 | tee ~/lss_runs/train_b$B.log | grep -E "^\[epoch|^激光线数|^训练集|最佳"
done

echo ""
echo "################ 汇总 ################"
python - <<'EOF'
import json, glob, os, re
rows = []
for f in glob.glob(os.path.expanduser('~/lss_runs/b*/result.json')):
    d = json.load(open(f))
    h = d['hist'][-1] if d.get('hist') else {}
    rows.append((int(d['n_beams']), d['best_iou'],
                 h.get('depth_mae', float('nan')), h.get('depth_hit', float('nan'))))
rows.sort()
if not rows:
    print('没有结果'); raise SystemExit

base = dict((r[0], r[1]) for r in rows).get(0)
COV = {0: 0.0, 2: 12.8, 4: 24.9, 8: 54.0, 16: 89.2, 32: 93.3}   # 实测覆盖率
print()
print('%-14s %-10s %-12s %-12s %-12s %s'
      % ('激光线数', '覆盖率', 'BEV分割IoU', '相对基线', '深度MAE(m)', 'bin命中'))
print('-' * 76)
for b, iou, mae, hit in rows:
    tag = '0（无监督）' if b == 0 else '%d 线' % b
    cov = '%.1f%%' % COV.get(b, float('nan'))
    rel = '—' if (base is None or b == 0) else '%+.1f%%' % ((iou - base) / base * 100)
    print('%-14s %-10s %-12.4f %-12s %-12.2f %.3f' % (tag, cov, iou, rel, mae, hit))
print()
print('两个要看的结论：')
print('  ① 加深度监督相对无监督基线提升多少')
print('  ② 从 32 线降到 8 线 / 4 线掉多少 —— 这才是本实验的问题：')
print('     「BEV 感知到底需要多少激光真值」，直接对应传感器选型成本。')
EOF
echo "结束 $(date)"
