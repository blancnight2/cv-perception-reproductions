#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pcdet
cd ~/OpenPCDet

LOG=~/autolabel/geninfos.log
python -u ~/autolabel/gen_infos_pseudo.py > "$LOG" 2>&1
echo "退出码: $?"
echo "--- 日志尾部 ---"
grep -a -vE "sample_idx|gt_database sample" "$LOG" | tail -15

echo ""
echo "=== 产出 ==="
ls -lh ~/OpenPCDet/data/kitti_pseudo/*.pkl 2>/dev/null
echo "gt_database: $(ls ~/OpenPCDet/data/kitti_pseudo/gt_database 2>/dev/null | wc -l) 个"

echo ""
echo "=== 核对 ==="
python - <<'EOF'
import pickle, numpy as np, os
def count(p):
    if not os.path.exists(p):
        return None
    d = pickle.load(open(p, 'rb'))
    n = sum(int((np.array(i['annos']['name']) == 'Car').sum()) for i in d)
    return len(d), n
for tag, p in [
    ('kitti_pseudo train', '/home/blancnight/OpenPCDet/data/kitti_pseudo/kitti_infos_train.pkl'),
    ('kitti_pseudo val  ', '/home/blancnight/OpenPCDet/data/kitti_pseudo/kitti_infos_val.pkl'),
    ('原始 train 真值   ', '/home/blancnight/OpenPCDet/data/kitti/kitti_infos_train.pkl'),
    ('原始 val   真值   ', '/home/blancnight/OpenPCDet/data/kitti/kitti_infos_val.pkl'),
]:
    r = count(p)
    print('%s : %s' % (tag, '%4d 帧, Car %5d 个' % r if r else '(缺失)'))
EOF
