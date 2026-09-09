#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
export HF_ENDPOINT=https://hf-mirror.com

echo "=== 环境清单 ==="
conda env list | grep -E "^det |^mmpercep "

conda activate det
echo ""
echo "=== det 环境自检 ==="
python - <<'EOF'
import sys
try:
    import torch
    print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
except Exception as e:
    print('torch 缺失:', e); sys.exit(1)
try:
    import rfdetr
    print('rfdetr OK')
except Exception as e:
    print('rfdetr 缺失:', e); sys.exit(2)
try:
    import torchmetrics, faster_coco_eval
    print('评测依赖 OK')
except Exception as e:
    print('评测依赖缺失:', e); sys.exit(3)
EOF
[ $? -ne 0 ] && { echo "!! 环境没装完，先补齐"; exit 1; }

cd ~/openvocab
echo ""
echo "=== 先看 RF-DETR 输出结构（1 张）==="
python - <<'EOF'
from rfdetr import RFDETRMedium
import glob, os
VAL='/mnt/d/GuangFU/PV Detection-LNN/YOLO 检测 + ByteTrack 跟踪/kitti_data/yolo/images/val'
p=sorted(glob.glob(os.path.join(VAL,'*')))[3]
m=RFDETRMedium()
d=m.predict(p, threshold=0.1)
print('图:', os.path.basename(p))
print('类型:', type(d))
print('属性:', [a for a in dir(d) if not a.startswith('_')][:15])
print('框数:', len(d.xyxy) if hasattr(d,'xyxy') else '?')
if hasattr(d,'class_id'):
    print('class_id 前10:', d.class_id[:10])
try:
    from rfdetr.util.coco_classes import COCO_CLASSES
    print('COCO_CLASSES 类型:', type(COCO_CLASSES), '长度', len(COCO_CLASSES))
    print('样例:', list(COCO_CLASSES.items())[:5] if isinstance(COCO_CLASSES,dict) else COCO_CLASSES[:5])
except Exception as e:
    print('COCO_CLASSES 导入失败:', e)
EOF
