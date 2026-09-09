#!/bin/bash
# Run 2：pillar 0.16 -> 0.08，其余与「原始基线」完全一致。
# batch 2 + GRAD_ACCUM 2 = 有效 batch 4，与原始那次(Car 75.63/Ped 45.81/Cyc 61.89)相同，
# 所以两者之间只差 pillar 尺寸这一个变量。
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pcdet
cd ~/OpenPCDet/tools || exit 1
LOG=~/autolabel/train_pp_run2_vox008.log
echo "===== Run2 pillar0.08 开始: $(date) =====" >> $LOG
GRAD_ACCUM=2 python -u train.py --cfg_file cfgs/kitti_models/pointpillar_vox008.yaml \
    --batch_size 2 --epochs 80 --extra_tag run2_vox008 >> $LOG 2>&1
echo "===== 结束: $(date) 退出码 $? =====" >> $LOG
