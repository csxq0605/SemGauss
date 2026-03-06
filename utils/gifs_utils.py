#!/usr/bin/env python3
"""
Generate GIFs from rendered outputs.

Usage:
  python scripts/make_gifs.py --eval-dir experiments/Replica/room_0_2027/eval --fps 5
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def load_frames(folder: Path):
    """Load frames sorted by name."""
    frames = []
    for p in sorted(folder.glob("*.png")):
        arr = imageio.imread(p)
        # GIFs are 8-bit; if depth is 16-bit or float, normalize to 0-255
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            rng = arr.max() - arr.min()
            if rng > 0:
                arr = (255 * (arr - arr.min()) / rng).astype(np.uint8)
            else:
                arr = np.zeros_like(arr, dtype=np.uint8)
        frames.append(arr)
    return frames


def make_gif(src: Path, out: Path, fps: int):
    frames = load_frames(src)
    if not frames:
        print(f"[skip] no frames in {src}")
        return
    imageio.mimsave(out, frames, duration=1 / fps)
    print(f"[ok] {out} ({len(frames)} frames, fps={fps})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True, help="Path to eval folder (contains rendered_* dirs)")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second for GIFs")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    targets = {
        "rendered_rgb": "rendered_rgb.gif",
        "rendered_depth": "rendered_depth.gif",
        # "rendered_semantic": "rendered_semantic.gif",
        "gt_rgb": "gt_rgb.gif",
        "gt_depth": "gt_depth.gif",
        # "gt_semantic": "gt_semantic.gif"
    }

    for sub, name in targets.items():
        src = eval_dir / sub
        out = eval_dir / name
        make_gif(src, out, args.fps)


if __name__ == "__main__":
    main()