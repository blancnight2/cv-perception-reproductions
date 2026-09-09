#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmpercep
cd ~/openvocab

R="/mnt/d/GuangFU/PV Detection-LNN/runs/detect"

run () {
  echo ""
  echo "######## $1 ########"
  python eval_zeroshot.py --backend ultra --conf 0.001 \
    --ckpt "$2" --tag "$1" 2>&1 | grep -vE "it/s|it\]" \
    | grep -E "mAP50 |mAP50-95|Car |Pedestrian |Cyclist |图片数"
}

run "RT-DETR-l-kitti"  "$R/rtdetr_kitti/weights/best.pt"
run "YOLO26n-kitti"    "$R/yolo26_kitti/weights/best.pt"
