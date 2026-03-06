import glob
import os
from typing import Optional

import cv2
import imageio
import numpy as np
import torch
from natsort import natsorted

from .basedataset import GradSLAMDataset, as_intrinsics_matrix
from . import datautils


class GoatCoreDataset(GradSLAMDataset):
    """Goat-core dataset loader (similar to Replica) with optional semantics.

    Expects structure:
      basedir/sequence/
        rgb/*.png|jpg
        depth/*.png|jpg
        traj.txt               # one 4x4 c2w per line (row-major)
        [semantic_remap/*.png] # optional labels; if missing, zeros are used
    """

    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "local_pos.txt")
        self.semantic_root = self._find_semantic_root()
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def _find_semantic_root(self) -> Optional[str]:
        for cand in ["semantic_remap", "semantic", "semantics"]:
            cand_dir = os.path.join(self.input_folder, cand)
            if os.path.isdir(cand_dir) and (
                glob.glob(os.path.join(cand_dir, "*.png"))
                or glob.glob(os.path.join(cand_dir, "*.jpg"))
            ):
                return cand_dir
        return None

    def get_filepaths(self):
        color_paths = natsorted(
            glob.glob(f"{self.input_folder}/rgb/*.png")
            + glob.glob(f"{self.input_folder}/rgb/*.jpg")
        )
        depth_paths = natsorted(
            glob.glob(f"{self.input_folder}/depth/*.npy")
        )

        if self.semantic_root is not None:
            semantic_paths = natsorted(
                glob.glob(f"{self.semantic_root}/*.png")
                + glob.glob(f"{self.semantic_root}/*.jpg")
            )
        else:
            semantic_paths = [None] * len(color_paths)

        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, semantic_paths, embedding_paths
    
    def _preprocess_depth(self, depth: np.ndarray):
        r"""Preprocesses the depth image by resizing, adding channel dimension, and scaling values to meters. Optionally
        converts depth from channels last :math:`(H, W, 1)` to channels first :math:`(1, H, W)` representation.

        Args:
            depth (np.ndarray): Raw depth image

        Returns:
            np.ndarray: Preprocessed depth

        Shape:
            - depth: :math:`(H_\text{old}, W_\text{old})`
            - Output: :math:`(H, W, 1)` if `self.channels_first == False`, else :math:`(1, H, W)`.
        """
        depth = np.expand_dims(depth, -1)
        if self.channels_first:
            depth = datautils.channels_first(depth)
        return depth
    
    def quaternion_to_rotation_matrix(self, q):
        norm = np.linalg.norm(q)
        q = q / norm
        x = q[0]
        y = q[1]
        z = q[2]
        w = q[3]
        R = np.array(
                [
                    [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                    [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
                    [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
                ]
            )
        return R

    def load_poses(self):
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        for i in range(self.num_imgs):
            line = lines[i]
            line = line.split()
            index = line[0]
            q = line[1:5]
            position = line[5:]
            q = np.array([float(x) for x in q])

            R_c2w = self.quaternion_to_rotation_matrix(q)
            R_w2c = np.linalg.inv(R_c2w)

            R_RUB_to_RDF = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            R = R_RUB_to_RDF @ R_w2c

            position = np.array([float(x) for x in position])
            t = R_RUB_to_RDF @ position

            c2w = np.eye(4)
            c2w[:3, :3] = R.T
            c2w[:3, 3] = t
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
        return poses

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        semantic_path = self.semantic_paths[index] if self.semantic_paths else None

        color = np.asarray(imageio.imread(color_path), dtype=float)
        color = self._preprocess_color(color)

        depth = np.load(depth_path)

        if semantic_path is not None:
            semantic = np.asarray(imageio.imread(semantic_path), dtype=np.uint8)
            semantic = self._preprocess_semantic(semantic)
        else:
            semantic = np.zeros((self.desired_height, self.desired_width, 1), dtype=np.uint8)

        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        if self.distortion is not None:
            color = cv2.undistort(color, K, self.distortion)
            semantic = cv2.undistort(semantic, K, self.distortion)

        color = torch.from_numpy(color)
        semantic = torch.from_numpy(semantic)

        depth = self._preprocess_depth(depth)
        depth = torch.from_numpy(depth)

        K = torch.from_numpy(K)
        K = datautils.scale_intrinsics(K, self.height_downsample_ratio, self.width_downsample_ratio)
        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K
        pose = self.transformed_poses[index]

        if self.crop_size is not None:
            color = color.permute(2, 0, 1)
            semantic = semantic.squeeze()
            depth = depth.squeeze()
            color = torch.nn.functional.interpolate(color[None], self.crop_size, mode="bilinear", align_corners=True)[0]
            depth = torch.nn.functional.interpolate(depth[None, None], self.crop_size, mode="nearest")[0, 0]
            semantic = torch.nn.functional.interpolate(semantic[None, None], self.crop_size, mode="nearest")[0, 0]
            color = color.permute(1, 2, 0).contiguous()
            semantic = semantic.unsqueeze(0).permute(1, 2, 0)
            depth = depth.unsqueeze(0).permute(1, 2, 0)

        edge = self.crop_edge
        if self.crop_edge != 0:
            color = color[edge:-edge, edge:-edge]
            depth = depth[edge:-edge, edge:-edge]
            semantic = semantic[edge:-edge, edge:-edge]

        return (
            color.to(self.device).type(self.dtype),
            depth.to(self.device).type(self.dtype),
            semantic.to(self.device).type(self.dtype),
            intrinsics.to(self.device).type(self.dtype),
            pose.to(self.device).type(self.dtype),
        )
