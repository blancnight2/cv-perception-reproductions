#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmpercep
cd ~/openvocab

echo "=== 单变体最优参照: V5 = 0.4374 ==="

for a in 0.0 0.5 1.0 2.0; do
  echo ""
  echo "######## alpha=$a  iou=0.55 ########"
  python fuse.py --tags V1 V2 V3 V4 V5 --iou 0.55 --alpha $a 2>/dev/null \
    | grep -E "mAP50 |mAP50-95|Car |Pedestrian |Cyclist |平均框数"
done

echo ""
echo "######## alpha=1.0  iou=0.45 ########"
python fuse.py --tags V1 V2 V3 V4 V5 --iou 0.45 --alpha 1.0 2>/dev/null \
  | grep -E "mAP50 |mAP50-95|平均框数"

echo ""
echo "######## alpha=1.0  iou=0.70 ########"
python fuse.py --tags V1 V2 V3 V4 V5 --iou 0.70 --alpha 1.0 2>/dev/null \
  | grep -E "mAP50 |mAP50-95|平均框数"
