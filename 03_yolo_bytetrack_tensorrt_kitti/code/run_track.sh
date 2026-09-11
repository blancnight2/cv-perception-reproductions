#!/bin/bash
PY=/home/blancnight/miniconda3/envs/bev/bin/python
CODE=~/cv-perception-reproductions/03_yolo_bytetrack_tensorrt_kitti/code
R=~/cv-perception-reproductions/03_yolo_bytetrack_tensorrt_kitti/results
cd "$CODE"
echo "########## 官方口径 ##########"
$PY -u eval_track.py --out "$R/tracking.json"
echo ""
echo "########## 归因口径：人/骑车人互记 ignore ##########"
$PY -u eval_track.py --neutral-person --out "$R/tracking_neutral.json"
echo ""
echo "ALLDONE"
