#!/bin/bash
# 搭建用伪标签训练的 KITTI 数据目录
#   train 帧 -> 伪标签      （模型只能看到伪标签）
#   val   帧 -> 真值标签    （评测必须用真值，否则作弊）
set -e
SRC=~/OpenPCDet/data/kitti
DST=~/OpenPCDet/data/kitti_pseudo
PSEUDO=~/autolabel/pseudo_gt_train

rm -rf "$DST"
mkdir -p "$DST/training" "$DST/testing"

echo "=== 软链大文件（不复制，省 20G）==="
for d in image_2 velodyne calib; do
    ln -sfn "$SRC/training/$d" "$DST/training/$d"
    echo "  training/$d -> 软链"
done
for d in velodyne calib; do
    ln -sfn "$SRC/testing/$d" "$DST/testing/$d"
done
cp -r "$SRC/ImageSets" "$DST/ImageSets"

echo ""
echo "=== 组装 label_2：train 用伪标签，val 用真值 ==="
mkdir -p "$DST/training/label_2"
NT=0; NV=0
while read -r fid; do
    [ -z "$fid" ] && continue
    if [ -f "$PSEUDO/$fid.txt" ]; then
        cp "$PSEUDO/$fid.txt" "$DST/training/label_2/$fid.txt"
    else
        : > "$DST/training/label_2/$fid.txt"
    fi
    NT=$((NT+1))
done < "$SRC/ImageSets/train.txt"

while read -r fid; do
    [ -z "$fid" ] && continue
    cp "$SRC/training/label_2/$fid.txt" "$DST/training/label_2/$fid.txt"
    NV=$((NV+1))
done < "$SRC/ImageSets/val.txt"

echo "  train 帧用伪标签: $NT"
echo "  val   帧用真值  : $NV"
echo "  label_2 总文件  : $(ls $DST/training/label_2 | wc -l)"

echo ""
echo "=== 抽查一个 train 帧的伪标签 ==="
FID=$(head -1 "$SRC/ImageSets/train.txt")
echo "--- $FID (伪) ---"; head -3 "$DST/training/label_2/$FID.txt"
echo "--- $FID (真值，仅对照) ---"; grep '^Car' "$SRC/training/label_2/$FID.txt" | head -3

echo ""
echo "=== 目录就绪 ==="
du -sh --exclude=label_2 "$DST" 2>/dev/null
ls -la "$DST/training" | head
