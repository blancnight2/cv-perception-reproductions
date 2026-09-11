# CV / 3D 感知算法作品集

作者：陈宗阳（测绘工程硕士 · 中国矿业大学(北京)）
方向：CV/自动驾驶感知 / 点云 / 三维重建 / 遥感

本仓库汇总我在**自建 Linux/CUDA 环境**（RTX 5070，Blackwell sm_120，CUDA 12.8，PyTorch 2.11+cu128）下，对若干经典/前沿感知与重建算法的研究。每个子目录含：上游代码、我的配置/脚本/环境适配与修复、以及**本机实测结果**（非论文照搬）。

| # | 项目 | 任务 | 数据集 | 关键指标（实测） |
|---|---|---|---|---|
| 01 | PointPillars (OpenPCDet) | 点云 3D 目标检测 | KITTI | 3D AP R40/Mod：Car 75.6 / Ped 43.7 / Cyc 62.1；**pillar 0.08 改进后 76.9 / 49.0 / 63.5** |
| 02 | 3D Gaussian Splatting | 场景重建·新视角合成 | Tanks&Temples (Truck) | PSNR 25.4 / SSIM 0.88 / LPIPS 0.14（≈/>原论文） |
| 03 | YOLO + ByteTrack + TensorRT | 2D 检测·跟踪·部署量化 | KITTI | **mAP50 90.4% / mAP50-95 65.1%**（YOLO11s@960+rect）；跟踪 Car MOTA **68.6%** / IDF1 **81.3%**；TensorRT FP16 端到端 **175.5 FPS** |
| 04 | PointNet++ | 点云语义分割 | S3DIS | mIoU 60.3% / OA 87.1% |
| 05 | Lift-Splat-Shoot 深度监督消融 | BEV 多相机感知·深度估计 | nuScenes | **16 线即达 32 线 95% 的深度精度**；权重校准后分割差异落入种子噪声 |
| 06 | YOLOE / Qwen3-VL 开放词汇 | 零样本检测·自动标注 | KITTI | YOLOE mAP50 **0.4414** vs Qwen3-VL-8B **0.2542**（同协议、同 1496 张验证集） |
| 07 | 视锥几何伪标签 | 点云 3D 自动标注 | KITTI | **零人工标注**下 BEV AP 45.39（全监督 88.00）；误差解耦：几何管线代价是 2D 检测器的 3.9 倍 |

## 方法上的几个习惯
- **先建噪声基线再下结论**：05 用种子重复档量化噪声（0.0028 / 0.0042），小于该值的差异一律不作结论。
- **单变量受控对比**：06 的长宽比实验做**近等像素量**设计，唯一变量是形状；01 的两轮改进各只动一个因素。
- **误差解耦**：07 先用 GT 2D 框测几何管线上限，再换检测器，掉点才能归因。
- **先写解析测试再碰真数据**：05 有 17 项不依赖数据集的解析测试。
- **假设被否掉也写进去**：01 的「有效批量」假设是被自己的实验推翻的，结果照登。

> 上游为开源项目（OpenPCDet / graphdeco-inria gaussian-splatting / yanx27 Pointnet_Pointnet2_pytorch / nv-tlabs lift-splat-shoot），本仓库保留其代码以便复现，**版权与许可证归原作者**（OpenPCDet Apache-2.0；gaussian-splatting 为 Inria/MPII 非商用研究许可；lift-splat-shoot 为 NVIDIA Source Code License）。06/07 的上游体积过大未纳入，clone 地址见各子目录 README。
> 大体积数据集与训练权重（*.pth/*.pt/*.engine/*.ply 等）未纳入版本管理；**不分发任何数据集**（KITTI / nuScenes / Tanks&Temples 请到官网按其许可获取）。
