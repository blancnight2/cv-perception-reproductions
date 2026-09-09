#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmpercep
cd ~/openvocab

LOG=~/openvocab/qwen_full.log
echo "开始时间: $(date)" > "$LOG"
echo "预计 1496 张 x 21.7s = 约 9 小时" >> "$LOG"
echo "" >> "$LOG"

python -u eval_zeroshot.py --backend qwen --tag "Qwen3-VL-8B-full" >> "$LOG" 2>&1

echo "" >> "$LOG"
echo "结束时间: $(date)" >> "$LOG"
