# 02 · 3D Gaussian Splatting — 场景重建与新视角合成复现

## 我做了什么
- 复现 graphdeco-inria 官方 3DGS；在自建 Linux/CUDA 12.8 环境**编译 diff-gaussian-rasterization / simple-knn 自定义 CUDA 算子**（Blackwell sm_120）。
- 在 **Tanks & Temples · Truck** 场景训练 30k 迭代，渲染并评测 PSNR / SSIM / LPIPS。

## 结果（Truck, 30k）
| 指标 | 实测 | 原论文(≈) |
|---|---|---|
| PSNR | **25.39** | 25.19 |
| SSIM | **0.884** | 0.879 |
| LPIPS | **0.143** | 0.148 |

达到并略优于原论文水平。样例渲染见 `results/`。

## 与测绘 / 高精地图的联系
掌握 多视图图像 / 相机位姿 → 3D 高斯优化 → 新视角实时渲染 全链路；3DGS 实景重建与高精地图静态要素构建 / 可微渲染新视角合成方向高度契合。
