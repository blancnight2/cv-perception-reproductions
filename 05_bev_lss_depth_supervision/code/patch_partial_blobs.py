# -*- coding: utf-8 -*-
"""
让 LSS 支持「只下载了部分 blob」的 nuScenes。

问题：v1.0-trainval_meta 覆盖全部 850 个场景，但每个 blob 只含约 85 个场景的
      实际图像/点云。LSS 的 get_scenes() 用标准 700/150 划分，横跨所有 10 个 blob，
      只下 1 个 blob 就会因大量文件缺失而崩。

做法：在 get_scenes() 之后与「磁盘上真实存在的场景」取交集。
      每个场景只 stat 一个代表文件（CAM_FRONT + LIDAR_TOP），850 次系统调用，很快。
      结果缓存到 dataroot 下，避免每次重扫。

对 v1.0-mini 无影响（文件本来就齐）。原文件备份 data.py.bak。
"""
import os, shutil

F = os.path.expanduser('~/lift-splat-shoot/src/data.py')

OLD = """        scenes = create_splits_scenes()[split]

        return scenes"""

NEW = """        scenes = create_splits_scenes()[split]

        # --- 只保留磁盘上真实存在的场景（支持只下载部分 blob）---
        scenes = self._keep_available(scenes)

        return scenes

    def _keep_available(self, scenes):
        \"\"\"与磁盘上文件齐全的场景取交集。

        只下载了 v1.0-trainval01_blobs 时，元数据仍宣称有 850 个场景，
        但其中只有约 85 个的文件真的在。不过滤会在 __getitem__ 里报文件不存在。
        \"\"\"
        import json as _json
        cache = os.path.join(self.nusc.dataroot, '.available_scenes.json')
        avail = None
        if os.path.exists(cache):
            try:
                avail = set(_json.load(open(cache)))
            except Exception:
                avail = None

        if avail is None:
            avail = set()
            for sc in self.nusc.scene:
                # 取该场景的第一个关键帧，检查前视相机与激光是否都在
                samp = self.nusc.get('sample', sc['first_sample_token'])
                ok = True
                for ch in ('CAM_FRONT', 'LIDAR_TOP'):
                    sd = self.nusc.get('sample_data', samp['data'][ch])
                    if not os.path.exists(os.path.join(self.nusc.dataroot, sd['filename'])):
                        ok = False
                        break
                if ok:
                    avail.add(sc['name'])
            try:
                _json.dump(sorted(avail), open(cache, 'w'))
            except Exception:
                pass

        kept = [s for s in scenes if s in avail]
        if len(kept) < len(scenes):
            print('[部分 blob] %s 划分：元数据 %d 个场景，磁盘上有 %d 个，使用 %d 个'
                  % ('train' if self.is_train else 'val',
                     len(scenes), len(avail), len(kept)))
        if not kept:
            raise RuntimeError(
                '该划分下磁盘上一个场景都没有。检查 blob 是否解压到了 %s'
                % self.nusc.dataroot)
        return kept"""

src = open(F, encoding='utf-8').read()
if '_keep_available' in src:
    print('已经打过补丁，跳过')
else:
    if OLD not in src:
        raise SystemExit('!! 找不到待替换片段')
    if not os.path.exists(F + '.bak'):
        shutil.copyfile(F, F + '.bak')
        print('已备份 data.py.bak')
    open(F, 'w', encoding='utf-8').write(src.replace(OLD, NEW, 1))
    print('补丁已打')

import ast
ast.parse(open(F, encoding='utf-8').read())
print('语法 OK')
print()
print('效果：只下 1 个 blob 时会打印实际可用场景数，并只用这些场景；')
print('      mini 数据集文件本来就齐，行为不变。')
