#!/bin/bash
# 独立检测器环境，不碰 mmpercep（Qwen 正在里面跑）
source ~/miniconda3/etc/profile.d/conda.sh
export HF_ENDPOINT=https://hf-mirror.com

if conda env list | grep -q "^det "; then
  echo "det 环境已存在"
else
  conda create -n det python=3.11 -y -q
fi
conda activate det

echo "=== torch cu128 ==="
python -c "import torch;print(torch.__version__)" 2>/dev/null || \
  pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch;print('torch', torch.__version__)"

echo ""
echo "=== rfdetr ==="
pip install -q rfdetr 2>&1 | tail -3
python -c "import rfdetr;print('rfdetr ok')" 2>&1 | tail -2

echo ""
echo "=== 评测依赖 ==="
pip install -q torchmetrics faster-coco-eval pillow tqdm 2>&1 | tail -2

echo ""
echo "=== 确认 mmpercep 没被动过 ==="
conda activate mmpercep
python -c "import torch;print('mmpercep torch', torch.__version__)"

echo ""
echo "=== 顺便 clone RT-DETRv4 看看权重怎么下 ==="
cd ~ && [ -d RT-DETRv4 ] || git clone -q --depth 1 https://github.com/RT-DETRs/RT-DETRv4.git
grep -riE "drive.google|huggingface|weights|checkpoint" ~/RT-DETRv4/README.md 2>/dev/null | head -12
