#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import imageio.v2 as imageio


def load_depth(npy_path: Path) -> np.ndarray:
    depth = np.load(npy_path)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth map, got shape {depth.shape} from {npy_path}")
    return depth.astype(np.float32)


def to_uint16(depth: np.ndarray, scale: float) -> np.ndarray:
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth = np.clip(depth, 0.0, None)
    depth_mm = np.round(depth * scale)
    depth_mm = np.clip(depth_mm, 0, np.iinfo(np.uint16).max)
    return depth_mm.astype(np.uint16)


def convert_folder(src_dir: Path, dst_dir: Path, scale: float) -> int:
    count = 0
    for npy_path in src_dir.rglob("*.npy"):
        rel = npy_path.relative_to(src_dir)
        dst_path = (dst_dir / rel).with_suffix(".png")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        depth = load_depth(npy_path)
        depth_u16 = to_uint16(depth, scale)
        imageio.imwrite(dst_path, depth_u16)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert depth .npy files (meters) to 16-bit PNG (scaled)."
    )
    parser.add_argument(
        "src",
        type=Path,
        help="Source directory containing depth .npy files (searched recursively)",
    )
    parser.add_argument(
        "dst",
        type=Path,
        help="Output directory for PNG files (mirrors source structure)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1000.0,
        help="Scale factor from meters to stored units (default: 1000 for mm)",
    )
    args = parser.parse_args()

    src_dir = args.src.expanduser().resolve()
    dst_dir = args.dst.expanduser().resolve()

    if not src_dir.is_dir():
        raise SystemExit(f"Source is not a directory: {src_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    total = convert_folder(src_dir, dst_dir, args.scale)
    print(f"Converted {total} files to {dst_dir}")


if __name__ == "__main__":
    main()
