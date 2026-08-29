import shutil
import tempfile
import unittest
from pathlib import Path


class TestS3DISPathResolution(unittest.TestCase):
    """验证 S3DIS 数据根目录的自动解析逻辑。"""

    def setUp(self):
        """创建临时目录结构，模拟仓库中的 S3DIS 数据布局。"""
        self._temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self._temp_dir)

    def tearDown(self):
        """清理测试创建的临时目录。"""
        shutil.rmtree(self._temp_dir)

    def test_resolves_nested_directory_with_area_files(self):
        """当根目录下存在嵌套子目录时，应自动定位到包含 Area 文件的目录。"""
        nested_dir = self.temp_path / "Stanford3dDataset_v1.2_Aligned_Version"
        nested_dir.mkdir()
        (nested_dir / "Area_1_office_1.npy").touch()

        from data_utils.s3dis_path import resolve_s3dis_data_root

        resolved = resolve_s3dis_data_root(str(self.temp_path))

        self.assertEqual(Path(resolved).resolve(), nested_dir.resolve())

    def test_raises_when_no_area_files_exist(self):
        """当目录中不存在处理后的 Area 文件时，应抛出清晰异常。"""
        from data_utils.s3dis_path import resolve_s3dis_data_root

        with self.assertRaisesRegex(ValueError, "Area_\\*\\.npy"):
            resolve_s3dis_data_root(str(self.temp_path))

    def test_lists_only_processed_area_npy_files(self):
        """当目录同时含有原始 Area 文件夹和处理后的样本时，应只返回 .npy 文件。"""
        data_dir = self.temp_path / "Stanford3dDataset_v1.2_Aligned_Version"
        data_dir.mkdir()
        (data_dir / "Area_1").mkdir()
        (data_dir / "Area_1_office_1.npy").touch()
        (data_dir / "Area_2_hallway_1.npy").touch()

        from data_utils.s3dis_path import list_s3dis_room_files

        rooms = list_s3dis_room_files(str(data_dir))

        self.assertEqual(rooms, ["Area_1_office_1.npy", "Area_2_hallway_1.npy"])


if __name__ == "__main__":
    unittest.main()
