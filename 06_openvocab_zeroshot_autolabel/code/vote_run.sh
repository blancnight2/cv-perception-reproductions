#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmpercep
cd ~/openvocab

CONF=0.001
run () {
  echo "### $1"
  python eval_zeroshot.py --backend yoloe --tag "$1" --conf $CONF --prompts "$2" 2>/dev/null \
    | grep -E "mAP50 |mAP50-95"
}

run V1 "car,pedestrian walking,person riding a bicycle"
run V2 "a photo of a car,a photo of a pedestrian,a photo of a person riding a bicycle"
run V3 "automobile,walking person,cyclist on a bike"
run V4 "a car on the street,a person walking on the sidewalk,a bicyclist"
run V5 "parked or moving car,standing or walking person,person on a bike"

echo ""
echo "=== 预测缓存 ==="
ls -lh ~/openvocab/preds_V*.jsonl
