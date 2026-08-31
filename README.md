# CV / 3D 感知算法作品集

作者：陈宗阳（测绘工程硕士 · 中国矿业大学(北京)）
方向：CV/自动驾驶感知 / 点云 / 三维重建 / 遥感

本仓库汇总我在**自建 Linux/CUDA 环境**（RTX 5070，Blackwell sm_120，CUDA 12.8，PyTorch 2.11+cu128）下，对若干经典/前沿感知与重建算法的研究。每个子目录含：上游代码、我的配置/脚本/环境适配与修复、以及**本机实测结果**（非论文照搬）。

| # | 项目 | 任务 | 数据集 | 关键指标（实测） |
|---|---|---|---|---|
| 01 | PointPillars (OpenPCDet) | 点云 3D 目标检测 | KITTI | 3D AP R40/Mod：Car 75.6 / Ped 43.7 / Cyc 62.1 |
| 02 | 3D Gaussian Splatting | 场景重建·新视角合成 | Tanks&Temples (Truck) | PSNR 25.4 / SSIM 0.88 / LPIPS 0.14（≈/>原论文） |
| 03 | YOLO + ByteTrack + TensorRT | 2D 检测·跟踪·部署量化 | KITTI | mAP50 85.0%；TensorRT FP16 近无损、体积 11→6MB |
| 04 | PointNet++ | 点云语义分割 | S3DIS | mIoU 60.3% / OA 87.1% |

> 上游为开源项目（OpenPCDet / graphdeco-inria gaussian-splatting / yanx27 Pointnet_Pointnet2_pytorch），本仓库保留其代码以便复现；我的贡献见各子目录 README。大体积数据集与训练权重（*.pth/*.pt/*.engine/*.ply 等）未纳入版本管理。
