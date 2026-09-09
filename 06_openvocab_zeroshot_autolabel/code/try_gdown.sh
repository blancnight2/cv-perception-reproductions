#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate det
pip install -q gdown 2>&1 | tail -2

mkdir -p ~/RT-DETRv4/pretrain && cd ~/RT-DETRv4/pretrain

# README 里的 checkpoint 文件 ID
declare -A IDS=(
  [S]=1jDAVxblqRPEWed7Hxm6GwcEl7zn72U6z
  [M]=1O-YpP4X-quuOXbi96y2TKkztbjroP5mX
  [L]=1shO9EzZvXZyKedE2urLsN4dwEv8Jqa_8
)

for k in S M L; do
  echo ""
  echo "=== 尝试下载 RT-DETRv4-$k ==="
  timeout 180 gdown "https://drive.google.com/uc?id=${IDS[$k]}" -O "rtv4_$k.pth" 2>&1 | tail -4
  if [ -s "rtv4_$k.pth" ]; then
    SZ=$(stat -c %s "rtv4_$k.pth")
    echo ">> 得到 $SZ 字节"
    # 小于 1MB 多半是 HTML 错误页
    if [ "$SZ" -lt 1000000 ]; then
      echo ">> 太小，多半是错误页："; head -c 200 "rtv4_$k.pth"; echo; rm -f "rtv4_$k.pth"
    else
      echo ">> ✅ 看起来是真权重"
    fi
  else
    echo ">> ❌ 没下到"
  fi
done

echo ""
echo "=== 结果 ==="
ls -lh ~/RT-DETRv4/pretrain/ 2>/dev/null
