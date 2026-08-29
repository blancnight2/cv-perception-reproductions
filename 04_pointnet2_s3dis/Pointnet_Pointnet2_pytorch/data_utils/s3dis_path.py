from pathlib import Path


def resolve_s3dis_data_root(base_dir: str) -> str:
    """解析包含 `Area_*.npy` 语义分割样本的 S3DIS 数据根目录。"""
    base_path = Path(base_dir).resolve()
    if not base_path.exists():
        raise ValueError(f"S3DIS 数据目录不存在: {base_path}")

    if any(base_path.glob("Area_*.npy")):
        return str(base_path)

    nested_matches = [child for child in base_path.iterdir() if child.is_dir() and any(child.glob("Area_*.npy"))]
    if len(nested_matches) == 1:
        return str(nested_matches[0])

    raise ValueError(
        f"在 {base_path} 及其一级子目录中都没有找到处理后的 Area_*.npy 文件。"
        "请先运行 data_utils/collect_indoor3d_data.py，或检查数据放置位置。"
    )


def list_s3dis_room_files(data_root: str) -> list[str]:
    """返回数据目录下所有处理后的 `Area_*.npy` 房间样本文件名。"""
    root_path = Path(data_root).resolve()
    return sorted(path.name for path in root_path.glob("Area_*.npy") if path.is_file())
