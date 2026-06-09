import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--pkl", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--seq_key", required=True)
args = parser.parse_args()

PKL = args.pkl
OUT = args.out
seq_key = args.seq_key

# TODO: replace with real intrinsics later if needed
fx, fy, cx, cy = 500.0, 500.0, 640.0, 360.0

d = np.load(INP)

T_world_cam = d[f"{SEQ}__aria_traj_T"]  # assumed camera/head to world
T_cam_world = np.linalg.inv(T_world_cam)

def transform_points(T, pts):
    # T: [F,4,4], pts: [F,N,3]
    ones = np.ones((*pts.shape[:2], 1), dtype=pts.dtype)
    pts_h = np.concatenate([pts, ones], axis=-1)
    out = np.einsum("fij,fnj->fni", T, pts_h)
    return out[..., :3]

def project(points_cam):
    x = points_cam[..., 0]
    y = points_cam[..., 1]
    z = points_cam[..., 2]

    u = fx * x / z + cx
    v = fy * y / z + cy

    return np.stack([u, v], axis=-1)

for side in ["left_hand", "right_hand"]:
    pts_world = d[f"{SEQ}__{side}"]
    pts_cam = transform_points(T_cam_world, pts_world)
    pts_2d = project(pts_cam)

    print(side)
    print("  world", pts_world.shape)
    print("  cam z min/max:", pts_cam[..., 2].min(), pts_cam[..., 2].max())
    print("  2d min/max:", np.nanmin(pts_2d), np.nanmax(pts_2d))

np.savez_compressed(OUT)
print("done")