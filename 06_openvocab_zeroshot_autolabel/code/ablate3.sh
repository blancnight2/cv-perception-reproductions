#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmpercep
cd ~/openvocab

run () {
  local tag="$1"; shift
  echo ""
  echo "############ $tag ############"
  python eval_zeroshot.py --backend yoloe --tag "$tag" "$@" 2>/dev/null \
    | grep -E "mAP50 |mAP50-95|Car |Pedestrian |Cyclist "
}

# 逐类最优组合：Car 用 CLIP 模板(J 最好)，Ped 用朴素词(H 最好)，Cyc 用描述式(K 最好)
MIX="a photo of a car,pedestrian walking,a person riding a bicycle"

run "L 逐类最优 conf=0.01"   --prompts "$MIX" --conf 0.01
run "M 逐类最优 conf=0.001"  --prompts "$MIX" --conf 0.001
run "N CLIP模板全套 conf=0.001" --conf 0.001 \
  --prompts "a photo of a car,a photo of a pedestrian walking,a photo of a person riding a bicycle"
run "O 更低阈值 conf=0.0001" --prompts "$MIX" --conf 0.0001

echo ""
echo "全部完成"
