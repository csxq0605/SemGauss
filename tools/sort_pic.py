#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def main() -> None:
    parser = argparse.ArgumentParser(
        description="按文件名前缀分类：dep* -> dep 子目录，frame* -> rgb 子目录"
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="待处理的文件夹路径（只处理该目录下一层的文件，不递归）",
    )
    args = parser.parse_args()

    base = args.folder.expanduser().resolve()
    if not base.is_dir():
        raise SystemExit(f"路径不存在或不是目录: {base}")

    dep_dir = base / "depth"
    rgb_dir = base / "rgb"
    dep_dir.mkdir(exist_ok=True)
    rgb_dir.mkdir(exist_ok=True)

    moved = 0
    for entry in base.iterdir():
        if not entry.is_file():
            continue
        suffix = entry.suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            continue

        name_lower = entry.name.lower()
        if name_lower.startswith("dep"):
            target = dep_dir / entry.name
        elif name_lower.startswith("frame"):
            target = rgb_dir / entry.name
        else:
            continue

        if entry.parent == target.parent:
            continue  # 已经在目标目录
        shutil.move(str(entry), target)
        moved += 1

    print(f"完成，移动文件数: {moved}")

if __name__ == "__main__":
    main()