#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pcdet
cd ~/OpenPCDet

LOG=~/autolabel/gtdb.log
python -u ~/autolabel/gen_gtdb.py > "$LOG" 2>&1
echo "退出码: $?"
grep -a -vE "gt_database sample" "$LOG" | tail -12

echo ""
echo "=== 产出 ==="
ls -lh ~/OpenPCDet/data/kitti_pseudo/kitti_dbinfos_train.pkl 2>/dev/null
D=~/OpenPCDet/data/kitti_pseudo/gt_database
if [ -d "$D" ]; then
    echo "gt_database: $(ls $D | wc -l) 个样本"
else
    echo "gt_database: 目录不存在"
fi

echo ""
echo "=== dbinfos 内容 ==="
python - <<'EOF'
import pickle, os
p = '/home/blancnight/OpenPCDet/data/kitti_pseudo/kitti_dbinfos_train.pkl'
if os.path.exists(p):
    d = pickle.load(open(p, 'rb'))
    for k, v in d.items():
        print('  %-12s %d 个' % (k, len(v)))
else:
    print('  (缺失)')
EOF
