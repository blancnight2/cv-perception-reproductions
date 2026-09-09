#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate det
cd ~/openvocab

echo "=== 先跑 20 张验证接口 ==="
python eval_zeroshot.py --backend rtv4 --conf 0.001 --limit 20 \
  --tag "RTv4-smoke" 2>&1 | grep -vE "it/s|it\]" | tail -12

echo ""
echo "=== 全量 1496 张（统一评测协议）==="
python eval_zeroshot.py --backend rtv4 --conf 0.001 \
  --tag "RT-DETRv4-L-kitti" 2>&1 | grep -vE "it/s|it\]" | tail -14
