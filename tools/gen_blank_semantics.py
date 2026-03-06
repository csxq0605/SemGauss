#!/usr/bin/env python3
"""Generate blank semantic maps (all zeros) as PNGs.

Usage:
  python tools/gen_blank_semantics.py --out data0/Goat-core/4ok/semantic_remap --count 600 --width 640 --height 480
"""
import argparse
from pathlib import Path
import numpy as np
import imageio.v2 as imageio


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate blank semantic PNGs (all zeros)")
    parser.add_argument("--out", required=True, type=Path, help="Output directory (will be created)")
    parser.add_argument("--count", type=int, default=600, help="Number of images to generate")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=480, help="Image height")
    args = parser.parse_args()

    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    blank = np.zeros((args.height, args.width), dtype=np.uint8)
    for i in range(args.count):
        fname = out_dir / f"img{i+1:04d}.png"
        imageio.imwrite(fname, blank)
    print(f"Generated {args.count} blank semantic images in {out_dir}")


if __name__ == "__main__":
    main()
