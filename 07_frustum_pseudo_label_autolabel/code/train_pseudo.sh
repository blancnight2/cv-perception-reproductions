#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pcdet
cd ~/OpenPCDet/tools

run () {
    NAME=$1
    LOG=~/autolabel/train_$NAME.log
    : > "$LOG"
    echo "===== $NAME 开始: $(date) =====" >> "$LOG"
    python -u train.py \
        --cfg_file cfgs/kitti_models/$NAME.yaml \
        --batch_size 4 \
        --epochs 80 \
        --extra_tag v1 \
        >> "$LOG" 2>&1
    echo "===== $NAME 结束: $(date) 退出码 $? =====" >> "$LOG"
    echo "[$NAME] 完成，最终 Car AP:"
    grep -a -A4 "Car AP_R40@0.70, 0.70, 0.70" "$LOG" | tail -5
}

# 1) 伪标签训练（零 3D 人工标注）
run pointpillar_car_pseudo

# 2) 真值 Car 单类对照
run pointpillar_car_gt

echo ""
echo "全部完成 $(date)"
