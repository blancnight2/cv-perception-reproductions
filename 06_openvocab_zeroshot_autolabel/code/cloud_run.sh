#!/bin/bash
# ============================================================
# 切到 GPU 模式后跑这个
# ============================================================
set -e

echo "########## 1. 架构自检（不过就别往下跑）##########"
python - <<'EOF'
import torch, sys
print('torch      :', torch.__version__)
print('cuda avail :', torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit('!! CUDA 不可用，检查是否在 GPU 模式')
cap = torch.cuda.get_device_capability()
print('GPU        :', torch.cuda.get_device_name(0))
print('capability :', cap)
print('显存       : %.1f GB' % (torch.cuda.get_device_properties(0).total_memory/1e9))
if cap[0] >= 12 and 'cu12' in torch.__version__ and not torch.__version__.endswith('cu128'):
    print('!! 警告：Blackwell(sm_120) 需要 cu128 版 torch')
EOF
echo ""

export VAL_IMG=/root/autodl-tmp/val/images
export VAL_LBL=/root/autodl-tmp/val/labels

echo "########## 2. 先跑 5 张验证链路 ##########"
cd /root/autodl-tmp
python eval_zeroshot.py --backend qwen \
    --model /root/autodl-tmp/qwen36 \
    --limit 5 --tag "Qwen3.6-smoke" 2>&1 | tail -14

echo ""
echo "########## 3. 确认无误后手动跑全量 ##########"
echo "命令："
echo "  cd /root/autodl-tmp && python -u eval_zeroshot.py --backend qwen \\"
echo "      --model /root/autodl-tmp/qwen36 --tag Qwen3.6-35B-full \\"
echo "      2>&1 | tee qwen36_full.log"
echo ""
echo "跑完把这两个文件下载回本地："
echo "  result_Qwen3.6-35B-full.json"
echo "  preds_Qwen3.6-35B-full.jsonl   <-- 预测缓存，能换指标重算"
