#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pcdet
cd ~/OpenPCDet

echo "########## 为伪标签数据集生成 infos + gt_database ##########"
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos \
    tools/cfgs/dataset_configs/kitti_pseudo_dataset.yaml 2>&1 | tail -25

echo ""
echo "=== 产出 ==="
ls -lh ~/OpenPCDet/data/kitti_pseudo/*.pkl 2>/dev/null
echo "gt_database 样本数: $(ls ~/OpenPCDet/data/kitti_pseudo/gt_database 2>/dev/null | wc -l)"

echo ""
echo "=== 核对：train 用伪标签、val 用真值 ==="
python - <<'EOF'
import pickle, numpy as np
for split in ('train', 'val'):
    p = '/home/blancnight/OpenPCDet/data/kitti_pseudo/kitti_infos_%s.pkl' % split
    d = pickle.load(open(p, 'rb'))
    n_box = sum((np.array(i['annos']['name']) == 'Car').sum() for i in d)
    print('%-5s : %d 帧, Car 框 %d 个' % (split, len(d), n_box))

p = '/home/blancnight/OpenPCDet/data/kitti/kitti_infos_val.pkl'
d = pickle.load(open(p, 'rb'))
n = sum((np.array(i['annos']['name']) == 'Car').sum() for i in d)
print('对照(原始 val 真值): %d 帧, Car 框 %d 个  <- 应与上面 val 一致' % (len(d), n))
EOF
