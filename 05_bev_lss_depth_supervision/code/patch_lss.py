# -*- coding: utf-8 -*-
"""把 LSS 的深度分布接出来（原版算了但丢掉了），并加一个可选的深度监督返回。

改动极小、且默认行为不变：forward 多返回一个 depth，旧调用方按元组第一项取即可。
每个被改的文件都留 .bak。
"""
import os, shutil

R = os.path.expanduser('~/lift-splat-shoot')
M = os.path.join(R, 'src/models.py')

PATCHES = [
    # ① CamEncode.forward：原版把 depth 算完就扔了
    ("""    def forward(self, x):
        depth, x = self.get_depth_feat(x)

        return x""",
     """    def forward(self, x):
        depth, x = self.get_depth_feat(x)

        # 原版在这里丢掉了 depth。深度监督实验需要它，改为一并返回。
        return depth, x"""),

    # ② get_cam_feats：把 depth 也 reshape 回 (B,N,D,fH,fW)
    ("""        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x""",
     """        x = x.view(B*N, C, imH, imW)
        depth, x = self.camencode(x)
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)
        depth = depth.view(B, N, self.D, imH//self.downsample, imW//self.downsample)

        return x, depth"""),

    # ③ get_voxels：透传
    ("""        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x = self.get_cam_feats(x)

        x = self.voxel_pooling(geom, x)

        return x""",
     """        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x, depth = self.get_cam_feats(x)

        x = self.voxel_pooling(geom, x)

        return x, depth"""),

    # ④ forward：返回 (BEV 输出, 深度分布)
    ("""    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        x = self.bevencode(x)
        return x""",
     """    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        x, depth = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        x = self.bevencode(x)
        return x, depth"""),
]

src = open(M, encoding='utf-8').read()
if not os.path.exists(M + '.bak'):
    shutil.copyfile(M, M + '.bak')
    print('已备份 models.py.bak')

n = 0
for old, new in PATCHES:
    if new.split('\n')[-1].strip() in src and old not in src:
        print('  [跳过] 已打过')
        continue
    if old not in src:
        raise SystemExit('!! 找不到待替换片段，上游代码可能变了：\n%s' % old[:80])
    src = src.replace(old, new, 1)
    n += 1

open(M, 'w', encoding='utf-8').write(src)
print('models.py 打了 %d 处补丁' % n)

# 让 src 包能 import 新模块
INIT = os.path.join(R, 'src/__init__.py')
init = open(INIT, encoding='utf-8').read()
add = 'from .depth_sup import lidar_to_depth, depth_loss, depth_metrics\n' \
      'from .data_depth import DepthSegData, compile_depth_data\n'
if 'depth_sup' not in init:
    open(INIT, 'a', encoding='utf-8').write('\n' + add)
    print('__init__.py 已补 import')

print('完成。原版行为不变，只是 forward 多返回一个 depth。')
