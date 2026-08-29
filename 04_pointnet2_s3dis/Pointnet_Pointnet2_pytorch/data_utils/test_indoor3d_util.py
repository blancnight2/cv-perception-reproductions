import os
import sys
import tempfile
import unittest


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import indoor3d_util


class ResolveDataPathTest(unittest.TestCase):
    """验证 S3DIS 数据目录解析优先级。"""

    def test_prefers_stanford_indoor3d_layout(self):
        """优先使用当前仓库实际存在的 stanford_indoor3d 目录。"""
        with tempfile.TemporaryDirectory() as root_dir:
            preferred = os.path.join(
                root_dir,
                "data",
                "stanford_indoor3d",
                "Stanford3dDataset_v1.2_Aligned_Version",
            )
            legacy = os.path.join(
                root_dir,
                "data",
                "s3dis",
                "Stanford3dDataset_v1.2_Aligned_Version",
            )
            os.makedirs(preferred)
            os.makedirs(legacy)

            resolved = indoor3d_util.resolve_data_path(root_dir)

            self.assertEqual(preferred, resolved)

    def test_falls_back_to_legacy_layout(self):
        """当新目录不存在时，回退到旧的 s3dis 目录。"""
        with tempfile.TemporaryDirectory() as root_dir:
            legacy = os.path.join(
                root_dir,
                "data",
                "s3dis",
                "Stanford3dDataset_v1.2_Aligned_Version",
            )
            os.makedirs(legacy)

            resolved = indoor3d_util.resolve_data_path(root_dir)

            self.assertEqual(legacy, resolved)


if __name__ == "__main__":
    unittest.main()
