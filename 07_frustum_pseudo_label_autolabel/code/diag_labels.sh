#!/bin/bash
P=~/OpenPCDet/data/kitti_pseudo
L=$P/training/label_2
S=$P/ImageSets

echo "=== 文件数 ==="
echo "label_2      : $(ls $L | wc -l)"
echo "train.txt    : $(wc -l < $S/train.txt)"
echo "val.txt      : $(wc -l < $S/val.txt)"
echo "trainval.txt : $(wc -l < $S/trainval.txt)"
echo "原始 label_2 : $(ls ~/OpenPCDet/data/kitti/training/label_2 | wc -l)"

echo ""
echo "=== 空文件还有吗 ==="
find "$L" -size 0 | wc -l

echo ""
echo "=== train/val 里缺哪些标签文件 ==="
miss=0
for split in train val; do
  while read -r fid; do
    [ -z "$fid" ] && continue
    if [ ! -f "$L/$fid.txt" ]; then
      echo "  缺 [$split] $fid"
      miss=$((miss+1))
      [ $miss -gt 8 ] && break 2
    fi
  done < "$S/$split.txt"
done
echo "缺失合计(最多列 8 个): $miss"

echo ""
echo "=== 003261 附近 ==="
for f in 003258 003260 003261 003262; do
  if [ -f "$L/$f.txt" ]; then
    echo "  $f.txt 存在, $(wc -l < $L/$f.txt) 行"
  else
    echo "  $f.txt 不存在"
  fi
  grep -qx "$f" "$S/train.txt" && echo "     在 train.txt 里"
  grep -qx "$f" "$S/val.txt" && echo "     在 val.txt 里"
done
