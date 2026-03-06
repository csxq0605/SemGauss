#!/usr/bin/env python3
import argparse
import math
import os
import sys

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from utils.recon_utils import setup_camera
from utils.slam_helpers import transformed_params2rendervar


def parse_cam_yaml(path):
    keys = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip()
            if k in ["image_width", "image_height", "fx", "fy", "cx", "cy"]:
                keys[k] = float(v.strip().split()[0])
    return keys


def look_at(cam_pos, target, up=np.array([0.0, 1.0, 0.0], dtype=np.float32)):
    forward = target - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    if abs(np.dot(forward, up)) > 0.99:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right = np.cross(up, forward)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(forward, right)
    rot = np.stack([right, up, forward], axis=1)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = rot
    c2w[:3, 3] = cam_pos
    return c2w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True, help="Path to params*.npz")
    parser.add_argument("--camera_yaml", required=True, help="Path to camera yaml for intrinsics")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--orbit_radius", type=float, default=0.2)
    parser.add_argument("--lat_steps", type=int, default=9)
    parser.add_argument("--lon_steps", type=int, default=18)
    parser.add_argument("--make_gif", action="store_true")
    parser.add_argument("--gif_fps", type=int, default=1)
    parser.add_argument("--gif_stride", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cam_cfg = parse_cam_yaml(args.camera_yaml)
    W = int(cam_cfg["image_width"])
    H = int(cam_cfg["image_height"])
    K = np.array(
        [[cam_cfg["fx"], 0.0, cam_cfg["cx"]],
         [0.0, cam_cfg["fy"], cam_cfg["cy"]],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    npz = np.load(args.params, allow_pickle=True)
    means3d = npz["means3D"].astype(np.float32)
    min_pt = means3d.min(axis=0)
    max_pt = means3d.max(axis=0)
    means3d = means3d - (min_pt + max_pt) / 2.0

    params = {
        "means3D": torch.tensor(means3d).cuda(),
        "rgb_colors": torch.tensor(npz["rgb_colors"].astype(np.float32)).cuda(),
        "sem_labels": torch.tensor(npz["sem_labels"].astype(np.float32)).cuda(),
        "unnorm_rotations": torch.tensor(npz["unnorm_rotations"].astype(np.float32)).cuda(),
        "logit_opacities": torch.tensor(npz["logit_opacities"].astype(np.float32)).cuda(),
        "log_scales": torch.tensor(npz["log_scales"].astype(np.float32)).cuda(),
    }

    center = means3d.mean(axis=0)
    radius = float(args.orbit_radius)

    directions = []
    eps = math.radians(5.0)
    lat_steps = max(args.lat_steps, 1)
    lon_steps = max(args.lon_steps, 1)
    for lat_i in range(lat_steps):
        if lat_steps == 1:
            phi = 0.0
        else:
            # Full-sphere latitudes, avoid poles to keep a stable up direction.
            phi = -(math.pi / 2 - eps) + (math.pi - 2 * eps) * (lat_i / (lat_steps - 1))
        for lon_i in range(lon_steps):
            theta = 2.0 * math.pi * (lon_i + 0.5) / lon_steps
            x = math.sin(phi)
            y = math.cos(phi) * math.cos(theta)
            z = math.cos(phi) * math.sin(theta)
            directions.append((x, y, z))

    frame_paths = []
    for i, (x, y, z) in enumerate(directions):
        cam_pos = center + np.array([radius * x, radius * y, radius * z], dtype=np.float32)
        c2w = look_at(cam_pos, center)
        w2c = np.linalg.inv(c2w)

        cam = setup_camera(W, H, K, w2c)
        pts = params["means3D"]
        pts_h = torch.cat([pts, torch.ones((pts.shape[0], 1), device=pts.device)], dim=1)
        w2c_t = torch.tensor(w2c, device=pts.device, dtype=pts.dtype)
        transformed_pts = (w2c_t @ pts_h.T).T[:, :3]
        rendervar = transformed_params2rendervar(params, w2c_t, transformed_pts)
        im_dep, _, _, _ = Renderer(raster_settings=cam)(**rendervar)
        im = im_dep[0:3].permute(1, 2, 0).detach().cpu().numpy()
        im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
        out_path = os.path.join(args.out_dir, f"orbit_{i:04d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        frame_paths.append(out_path)

    if args.make_gif:
        stride = max(args.gif_stride, 1)
        frames = [imageio.imread(p) for p in frame_paths[::stride]]
        gif_path = os.path.join(args.out_dir, "orbit.gif")
        imageio.mimsave(gif_path, frames, duration=1 / max(args.gif_fps, 1))
        print(f"Saved GIF: {gif_path}")

    print(f"Saved {len(directions)} frames to {args.out_dir}")


if __name__ == "__main__":
    main()
