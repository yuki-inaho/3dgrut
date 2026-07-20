#!/usr/bin/env python3
"""Render linear or spline camera paths through a 3DGRUT checkpoint."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image

from threedgrut.render import Renderer
from threedgrut.utils.render import apply_background, apply_feature_decoder, apply_post_processing


def _slerp_rotation(start: np.ndarray, end: np.ndarray, t: float) -> np.ndarray:
    """Interpolate camera-to-world rotations with the shortest quaternion arc."""
    def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
        trace = np.trace(matrix)
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            return np.array([(matrix[2, 1] - matrix[1, 2]) / scale,
                             (matrix[0, 2] - matrix[2, 0]) / scale,
                             (matrix[1, 0] - matrix[0, 1]) / scale,
                             0.25 * scale])
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            return np.array([0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale,
                             (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[2, 1] - matrix[1, 2]) / scale])
        if axis == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            return np.array([(matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                             (matrix[1, 2] + matrix[2, 1]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale])
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        return np.array([(matrix[0, 2] + matrix[2, 0]) / scale, (matrix[1, 2] + matrix[2, 1]) / scale,
                         0.25 * scale, (matrix[1, 0] - matrix[0, 1]) / scale])

    def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
        x, y, z, w = quaternion / np.linalg.norm(quaternion)
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    q0, q1 = matrix_to_quaternion(start), matrix_to_quaternion(end)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        return quaternion_to_matrix((1.0 - t) * q0 + t * q1)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    return quaternion_to_matrix((np.sin((1.0 - t) * theta) / sin_theta) * q0 + (np.sin(t * theta) / sin_theta) * q1)


def _catmull_rom(points: np.ndarray, index: int, t: float) -> np.ndarray:
    """Return a clamped Catmull-Rom position through ordered camera centers."""
    p0 = points[max(index - 1, 0)]
    p1 = points[index]
    p2 = points[min(index + 1, len(points) - 1)]
    p3 = points[min(index + 2, len(points) - 1)]
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--path", required=True, help="COLMAP scene adapter")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-frames", type=int, default=120)
    parser.add_argument("--trajectory", choices=("linear", "spline"), default="linear")
    parser.add_argument("--save-depth", action="store_true")
    parser.add_argument("--save-normals", action="store_true")
    args = parser.parse_args()

    if args.num_frames < 2:
        raise ValueError("--num-frames must be at least 2")
    renderer = Renderer.from_checkpoint(
        args.checkpoint,
        path=args.path,
        out_dir=args.out_dir,
        save_gt=False,
        computes_extra_metrics=False,
        enable_normals=args.save_normals,
    )
    source_poses = renderer.dataset.get_poses()
    start, end = source_poses[0], source_poses[-1]
    output_dir = Path(args.out_dir) / f"{args.trajectory}_camera_path"
    output_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = output_dir / "depth" if args.save_depth else None
    normals_dir = output_dir / "normals" if args.save_normals else None
    if depth_dir is not None:
        depth_dir.mkdir(exist_ok=True)
    if normals_dir is not None:
        normals_dir.mkdir(exist_ok=True)
    reference_batch = next(iter(renderer.dataloader))
    output_poses = []

    for frame in range(args.num_frames):
        t = frame / (args.num_frames - 1)
        pose = np.eye(4, dtype=np.float32)
        if args.trajectory == "linear":
            pose[:3, :3] = _slerp_rotation(start[:3, :3], end[:3, :3], t)
            pose[:3, 3] = (1.0 - t) * start[:3, 3] + t * end[:3, 3]
        else:
            curve_coordinate = t * (len(source_poses) - 1)
            index = min(int(curve_coordinate), len(source_poses) - 2)
            local_t = curve_coordinate - index
            pose[:3, :3] = _slerp_rotation(
                source_poses[index, :3, :3], source_poses[index + 1, :3, :3], local_t
            )
            pose[:3, 3] = _catmull_rom(source_poses[:, :3, 3], index, local_t)
        output_poses.append(pose)
        batch = renderer.dataset.get_gpu_batch_with_intrinsics(reference_batch)
        batch = replace(batch, T_to_world=torch.from_numpy(pose).to("cuda").unsqueeze(0), frame_idx=-1)
        outputs = renderer.model(batch)
        if renderer.feature_decoder is not None:
            outputs = apply_feature_decoder(renderer.feature_decoder, outputs, batch, training=False,
                                            center_ray_encoding=bool(getattr(renderer.conf.model.nht_decoder,
                                                                             "center_ray_encoding", False)))
        outputs = apply_background(renderer.model.background, outputs, batch, training=False)
        if renderer.post_processing is not None:
            outputs = apply_post_processing(renderer.post_processing, outputs, batch, training=False)
        torchvision.utils.save_image(outputs["pred_features"].squeeze(0).permute(2, 0, 1),
                                     output_dir / f"{frame:05d}.png")
        if depth_dir is not None:
            depth = outputs["pred_dist"].squeeze().float()
            opacity = outputs["pred_opacity"].squeeze().float()
            valid_depth = depth[opacity > 0.01]
            if valid_depth.numel() == 0:
                depth_image = torch.zeros_like(depth, dtype=torch.uint8)
            else:
                near, far = torch.quantile(valid_depth, torch.tensor([0.02, 0.98], device=depth.device))
                depth_image = ((depth - near) / (far - near).clamp_min(1e-6)).clamp(0, 1).mul(255).to(torch.uint8)
            Image.fromarray(depth_image.cpu().numpy(), mode="L").save(depth_dir / f"{frame:05d}.png")
        if normals_dir is not None:
            normals = outputs["pred_normals"].squeeze(0).float()
            normals_image = ((normals + 1.0) * 0.5).clamp(0, 1)
            torchvision.utils.save_image(normals_image.permute(2, 0, 1), normals_dir / f"{frame:05d}.png")
        print(f"rendered {frame + 1}/{args.num_frames}", flush=True)

    np.savez(
        output_dir / "poses.npz",
        poses=np.stack(output_poses),
        source_poses=source_poses,
        trajectory=args.trajectory,
    )


if __name__ == "__main__":
    main()
