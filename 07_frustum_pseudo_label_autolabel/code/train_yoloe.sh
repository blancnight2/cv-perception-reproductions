#!/bin/bash
# 完全零人工标注设定：YOLOE 零样本 2D 框 -> 视锥伪标签 -> PointPillars
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pcdet
cd ~/OpenPCDet/tools || exit 1

NAME=pointpillar_car_pseudo_yoloe
LOG=~/autolabel/train_$NAME.log
: > "$LOG"
echo "===== $NAME 开始: $(date) =====" >> "$LOG"

python -u train.py \
    --cfg_file cfgs/kitti_models/$NAME.yaml \
    --batch_size 4 \
    --epochs 80 \
    --extra_tag v1 >> "$LOG" 2>&1

echo "===== $NAME 结束: $(date) 退出码 $? =====" >> "$LOG"
