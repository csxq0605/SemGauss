#!/usr/bin/env python3
import argparse
import json
import math
import os
from typing import Dict

import numpy as np

C0 = 0.28209479177387814


def rgb_to_spherical_harmonic(rgb):
    return (rgb-0.5) / C0

# def _sigmoid(x: np.ndarray) -> np.ndarray:
#     return 1.0 / (1.0 + np.exp(-x))


# def _normalize_quat(q: np.ndarray) -> np.ndarray:
#     norm = np.linalg.norm(q, axis=1, keepdims=True)
#     return q / np.clip(norm, 1e-8, None)


# def _parse_camera_yaml(path: str) -> Dict[str, float]:
#     keys = {
#         "image_width": None,
#         "image_height": None,
#         "fx": None,
#         "fy": None,
#         "cx": None,
#         "cy": None,
#     }
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line or line.startswith("#") or ":" not in line:
#                 continue
#             k, v = line.split(":", 1)
#             k = k.strip()
#             if k in keys:
#                 keys[k] = float(v.strip().split()[0])
#     return keys


# def _load_traj(traj_path: str):
#     poses = []
#     with open(traj_path, "r", encoding="utf-8") as f:
#         for line in f:
#             vals = [float(x) for x in line.strip().split()]
#             if len(vals) != 16:
#                 continue
#             c2w = np.array(vals, dtype=np.float32).reshape(4, 4)
#             poses.append(c2w)
#     return poses


def _write_ply(path: str, data: np.ndarray, fields: list):
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {data.shape[0]}",
    ]
    for name, dtype in fields:
        header.append(f"property {dtype} {name}")
    header.append("end_header")
    header_bytes = ("\n".join(header) + "\n").encode("ascii")
    with open(path, "wb") as f:
        f.write(header_bytes)
        data.tofile(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True, help="Path to params*.npz")
    parser.add_argument("--traj", required=False, help="Path to traj.txt (c2w per line)")
    # parser.add_argument("--out_dir", required=True, help="Output directory root for 3DGS viewer")
    parser.add_argument("--iter", type=int, default=0, help="Iteration id for point_cloud folder")
    parser.add_argument("--camera_yaml", default=None, help="Path to camera yaml for intrinsics")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--use_synthetic_cameras", action="store_true")
    parser.add_argument("--num_views", type=int, default=120)
    parser.add_argument("--radius_scale", type=float, default=1.5)
    parser.add_argument(
        "--rotation_mode",
        choices=["c2w", "w2c"],
        default="c2w",
        help="Rotation matrix convention for cameras.json.",
    )
    parser.add_argument(
        "--position_mode",
        choices=["c2w", "w2c"],
        default="c2w",
        help="Position convention for cameras.json.",
    )
    parser.add_argument(
        "--opencv2opengl",
        action="store_true",
        help="Flip Y/Z axes to convert OpenCV camera frame to OpenGL.",
    )
    args = parser.parse_args()

    # cam_cfg = {}
    # if args.camera_yaml:
    #     cam_cfg = _parse_camera_yaml(args.camera_yaml)

    # width = args.width or int(cam_cfg.get("image_width") or 0)
    # height = args.height or int(cam_cfg.get("image_height") or 0)
    # fx = args.fx or float(cam_cfg.get("fx") or 0)
    # fy = args.fy or float(cam_cfg.get("fy") or 0)
    # cx = args.cx or float(cam_cfg.get("cx") or 0)
    # cy = args.cy or float(cam_cfg.get("cy") or 0)

    # if not all([width, height, fx, fy]):
    #     raise ValueError("Missing camera params; provide --camera_yaml or --width/--height/--fx/--fy/--cx/--cy")

    params = np.load(args.params, allow_pickle=True)
    means3d = params["means3D"].astype(np.float32)
    # colors = np.clip(params["rgb_colors"].astype(np.float32), 0.0, 1.0)
    colors = rgb_to_spherical_harmonic(params["rgb_colors"].astype(np.float32))
    # opacities = _sigmoid(params["logit_opacities"].astype(np.float32)).reshape(-1)
    opacities = params["logit_opacities"].astype(np.float32)
    # scales = np.exp(params["log_scales"].astype(np.float32)).reshape(-1, 1)
    scales = params["log_scales"].astype(np.float32)
    if scales.shape[1] == 1:
        scales = np.tile(scales, (1, 3))
    # rotations = _normalize_quat(params["unnorm_rotations"].astype(np.float32))
    rotations = params["unnorm_rotations"].astype(np.float32)

    n = means3d.shape[0]
    # sh_c0 = 0.282095  # SH basis constant for degree 0
    # f_dc = colors / sh_c0

    # PLY fields expected by 3DGS viewer
    nx = np.zeros((n, 3), dtype=np.float32)
    # scales3 = np.repeat(scales, 3, axis=1).astype(np.float32)

    ply_fields = [
        ("x", "float"),
        ("y", "float"),
        ("z", "float"),
        ("nx", "float"),
        ("ny", "float"),
        ("nz", "float"),
        ("f_dc_0", "float"),
        ("f_dc_1", "float"),
        ("f_dc_2", "float"),
        ("opacity", "float"),
        ("scale_0", "float"),
        ("scale_1", "float"),
        ("scale_2", "float"),
        ("rot_0", "float"),
        ("rot_1", "float"),
        ("rot_2", "float"),
        ("rot_3", "float"),
    ]

    ply_data = np.concatenate(
        [
            means3d,
            nx,
            # f_dc,
            colors,
            # opacities.reshape(-1, 1),
            opacities,
            # scales3,
            scales,
            rotations,
        ],
        axis=1,
    ).astype(np.float32)

    # pc_dir = os.path.join(args.out_dir, "point_cloud", f"iteration_{args.iter:06d}")
    # os.makedirs(pc_dir, exist_ok=True)
    # ply_path = os.path.join(pc_dir, "point_cloud.ply")
    ply_path = os.path.join("./experiments/point_cloud.ply")
    _write_ply(ply_path, ply_data, ply_fields)

    # # cameras.json
    # if args.use_synthetic_cameras:
    #     poses = []
    #     center = means3d.mean(axis=0)
    #     radius = np.linalg.norm(means3d - center, axis=1).mean()
    #     radius = float(radius * args.radius_scale)
    #     for i in range(args.num_views):
    #         theta = 2.0 * math.pi * i / args.num_views
    #         cam_pos = center + np.array([radius * math.cos(theta), radius * 0.2, radius * math.sin(theta)], dtype=np.float32)
    #         forward = center - cam_pos
    #         forward = forward / (np.linalg.norm(forward) + 1e-8)
    #         up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    #         right = np.cross(up, forward)
    #         right = right / (np.linalg.norm(right) + 1e-8)
    #         up = np.cross(forward, right)
    #         rot = np.stack([right, up, forward], axis=1)
    #         c2w = np.eye(4, dtype=np.float32)
    #         c2w[:3, :3] = rot
    #         c2w[:3, 3] = cam_pos
    #         poses.append(c2w)
    # else:
    #     if not args.traj:
    #         raise ValueError("Missing --traj (required unless --use_synthetic_cameras is set).")
    #     poses = _load_traj(args.traj)
    # cameras = []
    # cv2gl = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    # for idx, c2w in enumerate(poses):
    #     rot = c2w[:3, :3].copy()
    #     pos = c2w[:3, 3].copy()
    #     if args.opencv2opengl:
    #         rot = rot @ cv2gl
    #         pos = cv2gl @ pos
    #     if args.rotation_mode == "w2c":
    #         rot = rot.T
    #     if args.position_mode == "w2c":
    #         pos = -(rot @ pos)
    #     cam = {
    #         "id": idx,
    #         "img_name": f"{idx:05d}",
    #         "width": int(width),
    #         "height": int(height),
    #         "position": pos.tolist(),
    #         "rotation": rot.tolist(),
    #         "fx": float(fx),
    #         "fy": float(fy),
    #         "cx": float(cx),
    #         "cy": float(cy),
    #     }
    #     cameras.append(cam)

    # with open(os.path.join(args.out_dir, "cameras.json"), "w", encoding="utf-8") as f:
    #     json.dump(cameras, f, indent=2)

    # cfg_args_path = os.path.join(args.out_dir, "cfg_args")
    # cfg_args = (
    #     "Namespace(\n"
    #     f"model_path='{args.out_dir.replace(os.sep, '/')}',\n"
    #     f"source_path='{args.out_dir.replace(os.sep, '/')}',\n"
    #     "images='images',\n"
    #     "resolution=1,\n"
    #     "white_background=False,\n"
    #     "eval=False\n"
    #     ")\n"
    # )
    # with open(cfg_args_path, "w", encoding="utf-8") as f:
    #     f.write(cfg_args)

    print(f"Exported: {ply_path}")
    # print(f"Exported: {os.path.join(args.out_dir, 'cameras.json')}")
    # print(f"Exported: {cfg_args_path}")


if __name__ == "__main__":
    main()
