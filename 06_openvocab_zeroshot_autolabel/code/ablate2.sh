#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmpercep
cd ~/openvocab

BEST="car,pedestrian walking,person riding a bicycle"

run () {
  local tag="$1"; shift
  echo ""
  echo "############ $tag ############"
  python eval_zeroshot.py --backend yoloe --tag "$tag" "$@" 2>/dev/null \
    | grep -E "mAP50 |mAP50-95|Car |Pedestrian |Cyclist "
}

echo "基线 C 组 conf=0.25 已知: mAP50=0.1911"

# --- 置信度阈值扫描（文献说开放词汇模型在极低阈值下依然有效）---
run "F conf=0.05"  --prompts "$BEST" --conf 0.05
run "G conf=0.01"  --prompts "$BEST" --conf 0.01
run "H conf=0.001" --prompts "$BEST" --conf 0.001

# --- null 类：吸收 KITTI 未标注的 Van/Truck/Tram，减少误检 ---
run "I null类+conf0.01" --conf 0.01 \
  --prompts "car,pedestrian walking,person riding a bicycle,truck,van,bus,tram,traffic sign,tree,building"

# --- CLIP 模板（YOLOE 用 MobileCLIP 文本编码器）---
run "J CLIP模板+conf0.01" --conf 0.01 \
  --prompts "a photo of a car,a photo of a pedestrian walking,a photo of a person riding a bicycle"

# --- 尺寸描述（KITTI 远处小目标多）---
run "K 尺寸描述+conf0.01" --conf 0.01 \
  --prompts "a car on the road,a small pedestrian walking on the street,a person riding a bicycle"

echo ""
echo "全部完成"
