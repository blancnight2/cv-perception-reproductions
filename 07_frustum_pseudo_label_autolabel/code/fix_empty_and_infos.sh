#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pcdet
set -e

L=~/OpenPCDet/data/kitti_pseudo/training/label_2
echo "=== 给空标注帧补 DontCare（OpenPCDet 的 get_infos 不接受空文件）==="
n=0
for f in "$L"/*.txt; do
    if [ ! -s "$f" ]; then
        echo "DontCare -1 -1 -10 0.00 0.00 0.00 0.00 -1 -1 -1 -1000 -1000 -1000 -10" > "$f"
        n=$((n+1))
    fi
done
echo "补了 $n 个空文件"
echo "label_2 总数: $(ls $L | wc -l)"

echo ""
echo "=== 生成 infos + gt_database ==="
cd ~/OpenPCDet
python ~/autolabel/gen_infos_pseudo.py 2>&1 | grep -vE "gt_database sample" | tail -18

echo ""
echo "=== 产出 ==="
ls -lh ~/OpenPCDet/data/kitti_pseudo/*.pkl 2>/dev/null
echo "gt_database: $(ls ~/OpenPCDet/data/kitti_pseudo/gt_database 2>/dev/null | wc -l) 个"

echo ""
echo "=== 核对 train 用伪标签 / val 用真值 ==="
python - <<'EOF'
import pickle, numpy as np
def count(p):
    d = pickle.load(open(p, 'rb'))
    n = sum(int((np.array(i['annos']['name']) == 'Car').sum()) for i in d)
    return len(d), n
for s in ('train', 'val'):
    f, n = count('/home/blancnight/OpenPCDet/data/kitti_pseudo/kitti_infos_%s.pkl' % s)
    print('kitti_pseudo %-5s : %4d 帧, Car %d 个' % (s, f, n))
f, n = count('/home/blancnight/OpenPCDet/data/kitti/kitti_infos_val.pkl')
print('原始 val 真值      : %4d 帧, Car %d 个  <- 应与上面 val 一致' % (f, n))
f, n = count('/home/blancnight/OpenPCDet/data/kitti/kitti_infos_train.pkl')
print('原始 train 真值    : %4d 帧, Car %d 个  <- 伪标签应少于它' % (f, n))
EOF
