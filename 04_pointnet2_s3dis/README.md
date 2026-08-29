# 04 · PointNet++ — S3DIS 点云语义分割复现

## 我做了什么
- 基于 yanx27 `Pointnet_Pointnet2_pytorch`，在 **S3DIS** 室内点云上复现 PointNet++ 语义分割：房间分块 / 点采样预处理 + SA / FP 层，跑通训练-验证全流程。

## 结果
**mIoU 60.3% / OA 87.1%**。掌握 FPS 最远点采样、球查询分组、多尺度特征聚合等点云网络核心机制。

（与 DGCNN / PCT 合为“点云三范式：采样分组 / 动态图 / 注意力”。）
