#!/bin/bash
PY=/home/blancnight/miniconda3/envs/bev/bin/python
CODE=~/cv-perception-reproductions/03_yolo_bytetrack_tensorrt_kitti/code
W="/mnt/d/GuangFU/PV Detection-LNN/runs/detect/e3_yolo11s/weights/best.pt"
cd "$CODE"
# conf 与 track_low_thresh 配对：低于 track_low_thresh 的框跟踪器本来也会丢，
# 所以只降 conf 不降 low_thresh 是没用的，必须成对放开。
run () {  # $1=名字 $2=conf $3=tracker.yaml
  echo "#### $1  conf=$2  $3 ####"
  $PY -u dump_kitti_track.py --weights "$W" --name "$1" --imgsz 960 \
      --conf "$2" --tracker "$3" 2>&1 | tail -2
}
run s960_c025 0.25 bytetrack.yaml
run s960_c010 0.10 bytetrack_low010.yaml
run s960_c005 0.05 bytetrack_low005.yaml
run s960_c001 0.01 bytetrack_low001.yaml
echo ""
echo "#### TrackEval ####"
cd ~/TrackEval
$PY -u scripts/run_kitti.py \
  --TRACKERS_TO_EVAL s960_c025 s960_c010 s960_c005 s960_c001 \
  --METRICS HOTA CLEAR Identity --USE_PARALLEL True --NUM_PARALLEL_CORES 8
echo SWEEPDONE
