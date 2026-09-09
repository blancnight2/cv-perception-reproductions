#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mmpercep
cd ~/openvocab

run () {
  echo ""
  echo "############ $1 ############"
  shift
  python eval_zeroshot.py "$@" 2>/dev/null | grep -E "模型|mAP50|Car|Pedestrian|Cyclist|图片数"
}

run "A 基线: car / person / bicycle"        --backend yoloe --prompts "car,person,bicycle"
run "B 数据集原词: car / pedestrian / cyclist" --backend yoloe --prompts "car,pedestrian,cyclist"
run "C 描述式: 强调人车整体"                  --backend yoloe --prompts "car,pedestrian walking,person riding a bicycle"
run "D 泛化词: vehicle"                       --backend yoloe --prompts "vehicle,person,cyclist"
run "E 无提示模式 (内置 4585 类词表)"          --backend yoloe --pf

echo ""
echo "全部完成"
