#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmpercep
export HF_ENDPOINT=https://hf-mirror.com

echo "=== 安装/升级 ultralytics ==="
pip install -q -U ultralytics 2>&1 | tail -3

python - <<'EOF'
import ultralytics
print('ultralytics', ultralytics.__version__)
try:
    from ultralytics import YOLOE
    print('YOLOE 可用')
except ImportError as e:
    print('YOLOE 不可用:', e)
EOF

echo ""
echo "=== YOLOE 文本提示模式 · 全量 1496 张 ==="
cd ~/openvocab
python eval_zeroshot.py --backend yoloe 2>&1 | tail -20
