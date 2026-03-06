"""
Batch resize Replica room images to a target resolution.

Defaults:
  - base directory: data0/replica/room
  - target resolution: 640x480 (W x H)
  - resampling: LANCZOS for RGB, NEAREST for semantic/depth to keep labels.

Usage:
  python tools/resize_replica_images.py \
      --base data0/replica/room \
      --width 640 --height 480
"""

import argparse
from pathlib import Path
from typing import Dict

from PIL import Image


def resize_folder(folder: Path, target_size, resample) -> Dict[str, int]:
    stats = {"processed": 0, "skipped": 0, "errors": 0}
    if not folder.exists():
        return stats
    for img_path in sorted(folder.rglob("*.png")):
        try:
            with Image.open(img_path) as img:
                if img.size == target_size:
                    stats["skipped"] += 1
                    continue
                img.resize(target_size, resample=resample).save(img_path)
                stats["processed"] += 1
        except Exception:
            stats["errors"] += 1
    return stats


def main():
    parser = argparse.ArgumentParser(description="Resize Replica room images to a target resolution.")
    parser.add_argument("--base", type=Path, default=Path("data0/replica/room"), help="Base directory containing rgb/semantic_class/depth.")
    parser.add_argument("--width", type=int, default=640, help="Target width.")
    parser.add_argument("--height", type=int, default=480, help="Target height.")
    args = parser.parse_args()

    target_size = (args.width, args.height)
    resample_map = {
        "rgb": Image.LANCZOS,
        "semantic_class": Image.NEAREST,
        "depth": Image.NEAREST,
    }

    print(f"Base dir: {args.base.resolve()}")
    print(f"Target size: {target_size[0]}x{target_size[1]}")

    total = {"processed": 0, "skipped": 0, "errors": 0}
    for subdir, resample in resample_map.items():
        folder = args.base / subdir
        stats = resize_folder(folder, target_size, resample)
        print(f"{folder}: processed={stats['processed']}, skipped={stats['skipped']}, errors={stats['errors']}")
        for k in total:
            total[k] += stats[k]

    print(f"Overall: processed={total['processed']}, skipped={total['skipped']}, errors={total['errors']}")


if __name__ == "__main__":
    main()
