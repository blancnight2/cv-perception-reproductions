#!/bin/bash
PY=/home/blancnight/miniconda3/envs/bev/bin/python
CODE=~/cv-perception-reproductions/03_yolo_bytetrack_tensorrt_kitti/code
R=~/cv-perception-reproductions/03_yolo_bytetrack_tensorrt_kitti/results
cd "$CODE"
$PY -u eval_track.py \
  --weights "/mnt/d/GuangFU/PV Detection-LNN/runs/detect/e3_yolo11s/weights/best.pt" \
  --imgsz 960 --out "$R/tracking_yolo11s960.json"
echo "S960DONE"
