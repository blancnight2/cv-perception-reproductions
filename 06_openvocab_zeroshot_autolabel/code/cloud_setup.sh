#!/bin/bash
# ============================================================
# AutoDL 云端跑 Qwen3.6-35B-A3B —— 在【无卡模式】下执行本脚本
# ============================================================
set -e

echo "########## 0. 环境自检 ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "(无卡模式，正常)"
df -h /root/autodl-tmp | tail -1
echo ""

echo "########## 1. 学术加速 + 依赖 ##########"
source /etc/network_turbo 2>/dev/null || echo "(无 network_turbo，跳过)"
export HF_ENDPOINT=https://hf-mirror.com
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc

pip install -q -U transformers accelerate bitsandbytes
pip install -q torchmetrics faster-coco-eval pillow tqdm
python -c "import transformers; print('transformers', transformers.__version__)"
echo ""

echo "########## 2. 解压数据 ##########"
cd /root/autodl-tmp
if [ -f kitti_val.tar.gz ]; then
    tar xzf kitti_val.tar.gz
    echo "图片数: $(ls /root/autodl-tmp/val/images | wc -l)"
    echo "标签数: $(ls /root/autodl-tmp/val/labels | wc -l)"
else
    echo "!! 没找到 kitti_val.tar.gz，请先上传到 /root/autodl-tmp/"
    exit 1
fi
echo ""

echo "########## 3. 下载模型（约 70GB，无卡模式下做最划算）##########"
df -h /root/autodl-tmp | tail -1
AVAIL=$(df -BG --output=avail /root/autodl-tmp | tail -1 | tr -dc '0-9')
if [ "$AVAIL" -lt 80 ]; then
    echo "!! 可用空间只有 ${AVAIL}GB，Qwen3.6-35B 需要约 70GB。"
    echo "!! 请先在 AutoDL 控制台把数据盘扩容到 100GB+，再重跑本脚本。"
    exit 1
fi
hf download Qwen/Qwen3.6-35B-A3B --local-dir /root/autodl-tmp/qwen36

echo ""
echo "########## 完成 ##########"
echo "现在【关机 -> 切 GPU 模式开机】，然后跑 cloud_run.sh"
