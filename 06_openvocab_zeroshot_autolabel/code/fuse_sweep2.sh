#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmpercep
cd ~/openvocab

echo "=== 各单变体成绩 ==="
for t in V1 V2 V3 V4 V5; do
  python - <<EOF
import json;d=json.load(open("result_$t.json"))
print("  $t  mAP50 %.4f  mAP50-95 %.4f" % (d["map_50"], d["map"]))
EOF
done

f () {
  echo ""
  echo "######## $1 ########"
  shift
  python fuse.py "$@" 2>/dev/null | grep -E "mAP50 |mAP50-95|Car |Pedestrian |Cyclist |平均框数"
}

f "剔除弱变体V4, alpha=2, iou=0.70" --tags V1 V2 V3 V5 --iou 0.70 --alpha 2.0
f "只用强变体V5+V2+V3, alpha=2, iou=0.70" --tags V2 V3 V5 --iou 0.70 --alpha 2.0
f "全变体 alpha=3, iou=0.70" --tags V1 V2 V3 V4 V5 --iou 0.70 --alpha 3.0
f "全变体 alpha=5, iou=0.70" --tags V1 V2 V3 V4 V5 --iou 0.70 --alpha 5.0
f "剔除V4 alpha=3 iou=0.70" --tags V1 V2 V3 V5 --iou 0.70 --alpha 3.0
