#!/bin/bash
# 在 WSL 里打包上传件（不影响正在跑的 Qwen3-VL 任务）
set -e
OUT=~/cloud_upload
mkdir -p "$OUT"
SRC="/mnt/d/GuangFU/PV Detection-LNN/YOLO 检测 + ByteTrack 跟踪/kitti_data/yolo"

echo "=== 打包 KITTI val（纯英文路径，避免云端中文乱码）==="
rm -rf /tmp/val && mkdir -p /tmp/val
cp -r "$SRC/images/val" /tmp/val/images
cp -r "$SRC/labels/val" /tmp/val/labels
cd /tmp && tar czf "$OUT/kitti_val.tar.gz" val
echo "图片 $(ls /tmp/val/images | wc -l) 张，标签 $(ls /tmp/val/labels | wc -l) 个"
rm -rf /tmp/val

cp ~/openvocab/eval_zeroshot.py "$OUT/"
cp ~/openvocab/cloud_setup.sh "$OUT/" 2>/dev/null || true
cp ~/openvocab/cloud_run.sh "$OUT/" 2>/dev/null || true

echo ""
echo "=== 待上传清单 ==="
ls -lh "$OUT"
echo ""
echo "上传到 AutoDL 的 /root/autodl-tmp/"
