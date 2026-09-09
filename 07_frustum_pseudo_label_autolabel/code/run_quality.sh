#!/bin/bash
# 等三档伪标签生成完，逐个跑离线质量评估（GT 基线 + 3 个 YOLOE 阈值）
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pcdet
cd ~/autolabel || exit 1

LOG=~/autolabel/quality.log
: > "$LOG"

# 等生成结束
while pgrep -f "gen_pseudo.py --src yoloe" > /dev/null; do
    sleep 20
done

{
    echo "###################### 伪标签质量对比 ######################"
    echo
    for d in pseudo_gt_train pseudo_yoloe0.05_train pseudo_yoloe0.15_train pseudo_yoloe0.3_train; do
        if [ -d ~/autolabel/"$d" ]; then
            python -u val_pseudo_quality.py ~/autolabel/"$d"
            echo
        else
            echo "!! 缺目录 $d"
        fi
    done
    echo QUALITYDONE
} >> "$LOG" 2>&1
