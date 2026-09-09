#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pcdet
cd ~/OpenPCDet/tools || exit 1
LOG=~/autolabel/train_pp_run1_eb32.log
echo "" >> $LOG
echo "===== 续跑（从 epoch 12）: $(date) =====" >> $LOG
GRAD_ACCUM=4 python -u train.py --cfg_file cfgs/kitti_models/pointpillar.yaml \
    --batch_size 8 --epochs 80 --extra_tag run1_eb32 >> $LOG 2>&1
echo "===== 结束: $(date) 退出码 $? =====" >> $LOG
