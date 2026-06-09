"""
Solve for the effective Aria RGB camera intrinsics by fitting a least-squares
pinhole model to:
  - GT 3D SMPL wrists in UEM canonical body-world (from ee_val.pt)
  - GT 2D MediaPipe wrist detections (from hand_guidance_2d/*.npz)

Model: u = -fx * x_cam/z_cam + cx
       v = -fy * y_cam/z_cam + cy
where (x_cam, y_cam, z_cam) = R_c2w^T @ (p_world - t_c2w)

Usage:
  cd /work/courses/digital_human/team7/UniEgoMotion
  PYTHONPATH=. python ../my_coord_attempt/calibrate_aria_intrinsics.py
"""

import sys, os
sys.path.insert(0, '.')

import numpy as np
import torch
import utils.rotation_conversions as rc
from dataset.canonicalization import (
    rot_trans_to_matrix, canonicalize_trajectory, saved_sequence_to_full_sequence
)
from dataset.smpl_utils import get_smpl

GUIDANCE_2D_DIR = "/work/courses/digital_human/team7/cooking_vids_uni/hand_guidance_2d"
EE4D_PT = "/work/courses/digital_human/team7/ee4d_motion_uniegomotion/uniegomotion/ee_val.pt"
MIN_CONF = 0.6
LEFT_WRIST_IDX  = 20
RIGHT_WRIST_IDX = 21

# All clips we have 2D guidance for (seq_name, start_local_10fps)
CLIPS = [
    ("indiana_cooking_09_2___10257___11112", 160),
    ("indiana_cooking_09_2___11658___11901",   0),
    ("indiana_cooking_09_2___13536___13797",   0),
    ("indiana_cooking_09_2___15660___16143",  80),
    ("indiana_cooking_09_2___17802___18048",   0),
    ("indiana_cooking_09_2___3888___4272",     0),
    ("indiana_cooking_09_2___9249___9495",     0),
    ("georgiatech_bike_06_10___4314___6195",   0),
    ("georgiatech_bike_06_10___4314___6195", 240),
    ("georgiatech_bike_06_10___4314___6195", 320),
    ("georgiatech_bike_06_10___4314___6195", 400),
]


def get_canonical_data(data, seq_key, start_local, smpl, window=80):
    seq = data[seq_key]
    end_local = start_local + window - 1
    seg_aria = seq['aria_traj'][start_local:end_local+1].clone()
    seg_smpl = {k: v[start_local:end_local+1].clone() for k, v in seq['smpl_params'].items()}
    seg_smpl['betas'] = seq['smpl_params']['betas'][:1].clone()
    seg_kp3d = seq['kp3d'][start_local:end_local+1].clone()

    seg_aria_full, seg_smpl_full = saved_sequence_to_full_sequence(seg_aria, seg_smpl, smpl)
    can_aria, _, can_kp3d = canonicalize_trajectory(seg_aria_full, seg_smpl_full, seg_kp3d)
    return can_aria, can_kp3d  # (T,4,4), (T,76,3)


def collect_correspondences(data, smpl):
    """
    Returns arrays of:
      pts_cam  (N, 3) -- wrist positions in camera frame
      pixels   (N, 2) -- MediaPipe 2D detections (u, v)
    """
    all_pts = []
    all_pix = []

    for seq_key, start_local in CLIPS:
        if seq_key not in data:
            print(f"[SKIP] {seq_key} not in dataset")
            continue

        # 2D guidance filename
        seq_name_clip = f"{seq_key}st{start_local}_uem80"
        f2d_path = os.path.join(GUIDANCE_2D_DIR, f"{seq_name_clip}_2d_wrist_guidance.npz")
        if not os.path.exists(f2d_path):
            print(f"[SKIP] no 2D file: {f2d_path}")
            continue

        try:
            can_aria, can_kp3d = get_canonical_data(data, seq_key, start_local, smpl)
        except Exception as e:
            print(f"[SKIP] {seq_key}:{start_local} error: {e}")
            continue

        f2d = np.load(f2d_path, allow_pickle=True)
        lw2d = f2d['left_wrist_2d'].astype(np.float32)
        rw2d = f2d['right_wrist_2d'].astype(np.float32)
        lv2d = f2d['left_valid_2d'].astype(bool) & (f2d['left_conf_2d'] >= MIN_CONF)
        rv2d = f2d['right_valid_2d'].astype(bool) & (f2d['right_conf_2d'] >= MIN_CONF)

        T = min(len(can_aria), len(lw2d))
        R_c2w = can_aria[:T, :3, :3]  # (T,3,3)
        t_c2w = can_aria[:T, :3,  3]  # (T,3)
        gt_left  = can_kp3d[:T, LEFT_WRIST_IDX]   # (T,3)
        gt_right = can_kp3d[:T, RIGHT_WRIST_IDX]  # (T,3)

        # World → camera: R^T @ (p - t)
        pts_left_cam  = torch.einsum('tji,tj->ti', R_c2w, gt_left  - t_c2w).numpy()
        pts_right_cam = torch.einsum('tji,tj->ti', R_c2w, gt_right - t_c2w).numpy()

        # Keep only frames where z > 0 (in front of camera) and MediaPipe detected
        for t in range(T):
            if lv2d[t] and pts_left_cam[t, 2] > 0.05:
                all_pts.append(pts_left_cam[t])
                all_pix.append(lw2d[t])
            if rv2d[t] and pts_right_cam[t, 2] > 0.05:
                all_pts.append(pts_right_cam[t])
                all_pix.append(rw2d[t])

        n_l = lv2d[:T].sum()
        n_r = rv2d[:T].sum()
        print(f"  {seq_key}:{start_local:4d}  valid2d: L={n_l} R={n_r}")

    return np.array(all_pts, dtype=np.float64), np.array(all_pix, dtype=np.float64)


def solve_intrinsics(pts_cam, pixels):
    """
    Solve: u = -fx*(x/z) + cx,  v = -fy*(y/z) + cy
    via least squares.
    Returns fx, fy, cx, cy.
    """
    x = pts_cam[:, 0]
    y = pts_cam[:, 1]
    z = pts_cam[:, 2]

    Au = np.stack([-x / z, np.ones_like(x)], axis=1)  # (N, 2)
    Av = np.stack([-y / z, np.ones_like(y)], axis=1)  # (N, 2)
    bu = pixels[:, 0]
    bv = pixels[:, 1]

    sol_u, res_u, _, _ = np.linalg.lstsq(Au, bu, rcond=None)
    sol_v, res_v, _, _ = np.linalg.lstsq(Av, bv, rcond=None)

    fx, cx = sol_u
    fy, cy = sol_v

    # Reprojection errors
    pred_u = Au @ sol_u
    pred_v = Av @ sol_v
    err_u = np.abs(pred_u - bu)
    err_v = np.abs(pred_v - bv)
    err_pix = np.sqrt((pred_u - bu)**2 + (pred_v - bv)**2)

    print(f"\nSolved intrinsics:")
    print(f"  fx={fx:.2f}  cx={cx:.2f}")
    print(f"  fy={fy:.2f}  cy={cy:.2f}")
    print(f"  N = {len(pts_cam)} correspondences")
    print(f"  u residual: mean={err_u.mean():.1f}px  median={np.median(err_u):.1f}px")
    print(f"  v residual: mean={err_v.mean():.1f}px  median={np.median(err_v):.1f}px")
    print(f"  2D residual: mean={err_pix.mean():.1f}px  median={np.median(err_pix):.1f}px")

    return float(fx), float(fy), float(cx), float(cy)


def main():
    print("Loading dataset...")
    data = torch.load(EE4D_PT, map_location='cpu', weights_only=False)
    smpl = get_smpl('smplx')

    print("Collecting 3D–2D correspondences...")
    pts_cam, pixels = collect_correspondences(data, smpl)
    print(f"\nTotal correspondences: {len(pts_cam)}")

    if len(pts_cam) < 10:
        print("Not enough correspondences!")
        return

    fx, fy, cx, cy = solve_intrinsics(pts_cam, pixels)

    print(f"\n>>> Use these in project_wrist_world_to_pixels:")
    print(f"ARIA_RGB_INTRINS = np.array([{fx:.1f}, {fy:.1f}, {cx:.1f}, {cy:.1f}])")


if __name__ == "__main__":
    main()
