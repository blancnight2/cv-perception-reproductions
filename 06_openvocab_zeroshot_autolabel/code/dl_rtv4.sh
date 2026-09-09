#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate det
cd ~/RT-DETRv4/pretrain
rm -f *.part

LOG=~/RT-DETRv4/pretrain/dl.log
: > "$LOG"

# 先下 S 和 M（12GB 卡训练更现实），L 最后
for spec in "S 1jDAVxblqRPEWed7Hxm6GwcEl7zn72U6z" "M 1O-YpP4X-quuOXbi96y2TKkztbjroP5mX" "L 1shO9EzZvXZyKedE2urLsN4dwEv8Jqa_8"; do
  set -- $spec
  k=$1; id=$2
  echo "=== RT-DETRv4-$k ===" >> "$LOG"
  for try in 1 2 3; do
    gdown -c "https://drive.google.com/uc?id=$id" -O "rtv4_$k.pth" >> "$LOG" 2>&1
    if [ -s "rtv4_$k.pth" ] && [ "$(stat -c %s rtv4_$k.pth)" -gt 1000000 ]; then
      echo ">> $k OK $(stat -c %s rtv4_$k.pth) 字节" >> "$LOG"; break
    fi
    echo ">> $k 第 $try 次未完成，重试续传" >> "$LOG"
    sleep 5
  done
done

echo "" >> "$LOG"
echo "=== 最终 ===" >> "$LOG"
ls -lh ~/RT-DETRv4/pretrain/*.pth >> "$LOG" 2>&1
