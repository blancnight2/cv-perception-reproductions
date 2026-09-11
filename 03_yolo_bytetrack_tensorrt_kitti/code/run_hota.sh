#!/bin/bash
PY=/home/blancnight/miniconda3/envs/bev/bin/python
CODE=~/cv-perception-reproductions/03_yolo_bytetrack_tensorrt_kitti/code
W=/mnt/d/GuangFU/PV\ Detection-LNN/runs/detect
cd "$CODE"
echo "######## 导出 YOLO11n@640 ########"
$PY -u dump_kitti_track.py --weights "$W/train/weights/best.pt" \
    --name yolo11n_640 --imgsz 640 2>&1 | tail -3
echo ""
echo "######## 导出 YOLO11s@960 ########"
$PY -u dump_kitti_track.py --weights "$W/e3_yolo11s/weights/best.pt" \
    --name yolo11s_960 --imgsz 960 2>&1 | tail -3
echo ""
echo "######## TrackEval 官方评测 ########"
cd ~/TrackEval
$PY -u scripts/run_kitti.py --TRACKERS_TO_EVAL yolo11n_640 yolo11s_960 \
    --METRICS HOTA CLEAR Identity --USE_PARALLEL True --NUM_PARALLEL_CORES 8 2>&1 | tail -60
echo "HOTADONE"
