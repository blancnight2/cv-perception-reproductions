# -*- coding: utf-8 -*-
"""
KITTI(training) -> YOLO 格式 + 切 train/val(8:2)
生成: kitti_data/yolo/{images,labels}/{train,val}
用真·验证集算 mAP，别用 train 当 val。
运行: python split_kitti.py
"""
import os, glob, random, shutil
from PIL import Image

BASE   = r"D:\GuangFU\PV Detection-LNN\YOLO 检测 + ByteTrack 跟踪\kitti_data"
IMG_SRC = os.path.join(BASE, "training", "image_2")
LBL_SRC = os.path.join(BASE, "training", "label_2")
OUT     = os.path.join(BASE, "yolo")            # 输出数据集根
VAL_RATIO = 0.2
SEED = 0

# KITTI 8 类 -> 合并成 3 类；Tram/Misc/DontCare 忽略
CLS = {"Car":0, "Van":0, "Truck":0,
       "Pedestrian":1, "Person_sitting":1,
       "Cyclist":2}

for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

stems = sorted(os.path.splitext(os.path.basename(f))[0]
               for f in glob.glob(os.path.join(IMG_SRC, "*.png")))
random.seed(SEED); random.shuffle(stems)
n_val = int(len(stems) * VAL_RATIO)
val_set = set(stems[:n_val])

def link_or_copy(src, dst):
    if os.path.exists(dst): return
    try: os.link(src, dst)          # 硬链接：不额外占 12GB
    except Exception: shutil.copy(src, dst)

n_box = 0
for stem in stems:
    split = "val" if stem in val_set else "train"
    img = os.path.join(IMG_SRC, stem + ".png")
    W, H = Image.open(img).size     # 逐图读真实尺寸，最稳
    lines = []
    lbl = os.path.join(LBL_SRC, stem + ".txt")
    if os.path.exists(lbl):
        for ln in open(lbl):
            p = ln.split()
            if not p or p[0] not in CLS: continue
            x1, y1, x2, y2 = map(float, p[4:8])
            cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            w,  h  = (x2 - x1) / W,     (y2 - y1) / H
            lines.append(f"{CLS[p[0]]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            n_box += 1
    link_or_copy(img, os.path.join(OUT, "images", split, stem + ".png"))
    with open(os.path.join(OUT, "labels", split, stem + ".txt"), "w") as f:
        f.write("\n".join(lines))

print(f"完成：总 {len(stems)} 张 -> train {len(stems)-n_val} / val {n_val}；标注框 {n_box} 个")
print("数据集根:", OUT)

# 顺手写好 yaml（覆盖你命令里用的那个 kitti.yaml，在 kitti_data 的上一层）
yaml = os.path.join(os.path.dirname(BASE), "kitti.yaml")
with open(yaml, "w", encoding="utf-8") as f:
    f.write(
        "path: " + OUT.replace("\\", "/") + "\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: Car\n  1: Pedestrian\n  2: Cyclist\n"
    )
print("已写 yaml:", yaml)
