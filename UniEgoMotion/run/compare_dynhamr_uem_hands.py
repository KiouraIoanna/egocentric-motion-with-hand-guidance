#!/usr/bin/env python3

import os
import sys

PROJECT_ROOT = "/work/courses/digital_human/team7/UniEgoMotion"
sys.path.insert(0, PROJECT_ROOT)

import argparse
import glob
import numpy as np

from dataset.ee4d_motion_dataset import EE4D_Motion_Dataset, careful_collate_fn
from dataset.smpl_utils import get_smpl, evaluate_smpl


ROOT_IDX = 0
LEFT_WRIST_IDX = 20
RIGHT_WRIST_IDX = 21

DATA_DIR = "/work/courses/digital_human/team7/ee4d_motion_uniegomotion"
DEFAULT_HAND_GUIDANCE_DIR = "/work/courses/digital_human/team7/Dyn-HaMR/results"
DEFAULT_OUT_ROOT = "/work/scratch/{user}/hand_comparison"

# Aria RGB camera intrinsics (1408 × 1408) — same as ARIA_RGB_INTRINS in run_merged.py
ARIA_FX, ARIA_FY, ARIA_CX, ARIA_CY = 396.0, 336.3, 717.3, 877.8


def mean_l2(a, b, mask):
    if mask.sum() == 0:
        return np.nan
    return float(np.linalg.norm(a[mask] - b[mask], axis=-1).mean())


def project_world_to_pixels(points_world, aria_traj_T, fx, fy, cx, cy, min_z=0.05):
    """
    Project (T, 3) world-space points through GT Aria c2w matrices.
    Convention matches project_wrist_world_to_pixels in run_merged.py:
      world → camera:  p_cam = R_c2w^T @ (p_world − t_c2w)
      pixel:           u = −fx · p_cam_x / z + cx
                       v = −fy · p_cam_y / z + cy
    Returns (T, 2) pixel coords and (T,) depth_ok mask.
    """
    R_c2w = aria_traj_T[:, :3, :3]
    t_c2w = aria_traj_T[:, :3, 3]
    pts_cam = np.einsum("tji,tj->ti", R_c2w, points_world - t_c2w)
    z = pts_cam[:, 2]
    depth_ok = z > min_z
    z_safe = np.where(depth_ok, z, min_z)
    u = -fx * pts_cam[:, 0] / z_safe + cx
    v = -fy * pts_cam[:, 1] / z_safe + cy
    uv = np.stack([u, v], axis=-1)
    uv[~depth_ok] = np.nan
    return uv, depth_ok


def rigid_align(A, B):
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)

    mu_A = A.mean(axis=0)
    mu_B = B.mean(axis=0)

    AA = A - mu_A
    BB = B - mu_B

    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    scale = S.sum() / ((AA ** 2).sum() + 1e-8)
    t = mu_B - scale * (R @ mu_A)

    return scale, R, t


def apply_similarity(points, scale, R, t):
    return scale * (points @ R.T) + t


def find_dyn_guidance(hand_guidance_dir, seq_name, start_idx):
    seq_key = f"{seq_name}st{start_idx}_uem80"

    patterns = [
        os.path.join(hand_guidance_dir, f"{seq_key}_hand_guidance.npz"),
        os.path.join(hand_guidance_dir, "**", f"{seq_key}_hand_guidance.npz"),
        os.path.join(hand_guidance_dir, "**", f"*{seq_key}*hand_guidance*.npz"),
    ]

    matches = []
    for p in patterns:
        matches.extend(glob.glob(p, recursive=True))

    matches = sorted(set(matches))

    if len(matches) == 0:
        raise FileNotFoundError(
            "Could not find Dyn-HaMR hand guidance file.\nTried:\n"
            + "\n".join(patterns)
        )

    if len(matches) > 1:
        print("[WARN] Multiple Dyn-HaMR files found. Using first:")
        for m in matches[:10]:
            print(" ", m)

    return matches[0], seq_key


def load_uem_from_dataset(seq_name, start_idx):
    print("[INFO] Loading UniEgoMotion dataset sample")
    print("DATA_DIR :", DATA_DIR)
    print("seq_name :", seq_name)
    print("start_idx:", start_idx)

    ds = EE4D_Motion_Dataset(
        data_dir=DATA_DIR,
        split="val",
        repre_type="v4_beta",
        cond_img_feat=True,
        cond_traj=True,
        window=80,
        img_feat_type="dinov2",
        cond_betas=True,
    )

    sample = ds.get_from_seq_and_st(seq_name, start_idx, 0)
    batch = careful_collate_fn([sample])
    mdata = ds.ret_to_full_sequence(batch)

    smpl = get_smpl()
    smpl_params = mdata["smpl_params_full"][0]

    joints, verts, _ = evaluate_smpl(smpl, smpl_params)
    joints = joints.detach().cpu().numpy()

    uem_left = joints[:, LEFT_WRIST_IDX, :]
    uem_right = joints[:, RIGHT_WRIST_IDX, :]
    uem_root = joints[:, ROOT_IDX, :]

    # GT Aria c2w trajectory — used for 2D reprojection
    aria_traj_T = mdata["aria_traj_T"][0].detach().cpu().numpy()  # (T, 4, 4)

    return uem_left, uem_right, uem_root, aria_traj_T


def load_dyn(dyn_path):
    data = np.load(dyn_path, allow_pickle=True)

    dyn_left = data["left_trans"]
    dyn_right = data["right_trans"]

    if "left_valid" in data.files:
        left_valid = data["left_valid"].astype(bool)
    else:
        left_valid = np.ones(len(dyn_left), dtype=bool)

    if "right_valid" in data.files:
        right_valid = data["right_valid"].astype(bool)
    else:
        right_valid = np.ones(len(dyn_right), dtype=bool)

    return dyn_left, dyn_right, left_valid, right_valid

def masked_min_max(err, mask):
    if mask.sum() == 0:
        return np.nan, np.nan
    vals = err[mask]
    return float(vals.min()), float(vals.max())

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--example",
        required=True,
        help="Example in format seq_name:start_idx, e.g. indiana_cooking_09_2___10257___11112:160",
    )

    parser.add_argument(
        "--hand_guidance_dir",
        default=DEFAULT_HAND_GUIDANCE_DIR,
        help="Folder containing Dyn-HaMR *_hand_guidance.npz files.",
    )

    parser.add_argument(
        "--out_root",
        default=DEFAULT_OUT_ROOT.format(user=os.environ.get("USER", "user")),
        help="Output folder.",
    )

    parser.add_argument("--fx", type=float, default=ARIA_FX)
    parser.add_argument("--fy", type=float, default=ARIA_FY)
    parser.add_argument("--cx", type=float, default=ARIA_CX)
    parser.add_argument("--cy", type=float, default=ARIA_CY)

    # accepted only so old commands do not crash
    parser.add_argument("--save_eval_npz", action="store_true")

    args = parser.parse_args()

    if ":" not in args.example:
        raise ValueError("--example must be in format seq_name:start_idx")

    seq_name, start_idx = args.example.rsplit(":", 1)
    start_idx = int(start_idx)

    dyn_npz, seq_key = find_dyn_guidance(
        args.hand_guidance_dir,
        seq_name,
        start_idx,
    )

    os.makedirs(args.out_root, exist_ok=True)

    out_npz = os.path.join(
        args.out_root,
        f"{seq_key}_dynhamr_uem_hand_difference.npz",
    )
    out_csv = os.path.join(
        args.out_root,
        f"{seq_key}_dynhamr_uem_hand_difference.csv",
    )

    print("========== Resolved inputs ==========")
    print("example:", args.example)
    print("seq_key:", seq_key)
    print("Dyn-HaMR npz:", dyn_npz)
    print("out npz:", out_npz)
    print("out csv:", out_csv)
    print()

    uem_left, uem_right, uem_root, aria_traj_T = load_uem_from_dataset(seq_name, start_idx)
    dyn_left, dyn_right, left_valid, right_valid = load_dyn(dyn_npz)

    n = min(
        len(uem_left),
        len(uem_right),
        len(uem_root),
        len(dyn_left),
        len(dyn_right),
        len(left_valid),
        len(right_valid),
    )

    uem_left = uem_left[:n]
    uem_right = uem_right[:n]
    uem_root = uem_root[:n]
    dyn_left = dyn_left[:n]
    dyn_right = dyn_right[:n]
    left_valid = left_valid[:n]
    right_valid = right_valid[:n]

    uem_left_rel = uem_left - uem_root
    uem_right_rel = uem_right - uem_root

    src = []
    tgt = []

    if left_valid.sum() > 0:
        src.append(dyn_left[left_valid])
        tgt.append(uem_left_rel[left_valid])

    if right_valid.sum() > 0:
        src.append(dyn_right[right_valid])
        tgt.append(uem_right_rel[right_valid])

    if len(src) == 0:
        raise RuntimeError("No valid Dyn-HaMR hand frames found.")

    src = np.concatenate(src, axis=0)
    tgt = np.concatenate(tgt, axis=0)

    if len(src) < 4:
        raise RuntimeError(f"Not enough valid alignment points: {len(src)}")

    scale, R, t = rigid_align(src, tgt)

    dyn_left_rel_aligned = apply_similarity(dyn_left, scale, R, t)
    dyn_right_rel_aligned = apply_similarity(dyn_right, scale, R, t)

    dyn_left_aligned = uem_root + dyn_left_rel_aligned
    dyn_right_aligned = uem_root + dyn_right_rel_aligned

    left_err = np.linalg.norm(dyn_left_aligned - uem_left, axis=-1)
    right_err = np.linalg.norm(dyn_right_aligned - uem_right, axis=-1)

    left_min, left_max = masked_min_max(left_err, left_valid)
    right_min, right_max = masked_min_max(right_err, right_valid)

    # 2D reprojection errors through GT Aria camera
    fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy
    traj = aria_traj_T[:n].astype(np.float64)

    dyn_l_uv, dyn_l_depth = project_world_to_pixels(dyn_left_aligned.astype(np.float64),  traj, fx, fy, cx, cy)
    dyn_r_uv, dyn_r_depth = project_world_to_pixels(dyn_right_aligned.astype(np.float64), traj, fx, fy, cx, cy)
    gt_l_uv,  gt_l_depth  = project_world_to_pixels(uem_left.astype(np.float64),          traj, fx, fy, cx, cy)
    gt_r_uv,  gt_r_depth  = project_world_to_pixels(uem_right.astype(np.float64),         traj, fx, fy, cx, cy)

    left_valid_2d  = left_valid  & dyn_l_depth & gt_l_depth
    right_valid_2d = right_valid & dyn_r_depth & gt_r_depth

    left_err_2d  = np.linalg.norm(dyn_l_uv - gt_l_uv,  axis=-1)
    right_err_2d = np.linalg.norm(dyn_r_uv - gt_r_uv,  axis=-1)

    left_min_2d,  left_max_2d  = masked_min_max(left_err_2d,  left_valid_2d)
    right_min_2d, right_max_2d = masked_min_max(right_err_2d, right_valid_2d)

    print("========== Dyn-HaMR vs GT hand difference ==========")
    print(f"frames: {n}")
    print(f"valid left: {int(left_valid.sum())}  valid right: {int(right_valid.sum())}")
    print(f"alignment points: {len(src)}  shared scale: {scale:.6f}")
    print()
    print("--- 3D errors (metres) ---")
    print(f"left  mean={mean_l2(dyn_left_aligned, uem_left, left_valid):.4f}  "
          f"median={np.median(left_err[left_valid]):.4f}  "
          f"min={left_min:.4f}  max={left_max:.4f}")
    print(f"right mean={mean_l2(dyn_right_aligned, uem_right, right_valid):.4f}  "
          f"median={np.median(right_err[right_valid]):.4f}  "
          f"min={right_min:.4f}  max={right_max:.4f}")
    print(f"both  mean={np.nanmean([left_err[left_valid].mean(), right_err[right_valid].mean()]):.4f}")
    print()
    print(f"--- 2D errors (pixels, GT Aria camera, fx={fx} fy={fy} cx={cx} cy={cy}) ---")
    if left_valid_2d.sum() > 0:
        print(f"left  mean={left_err_2d[left_valid_2d].mean():.2f}  "
              f"median={np.median(left_err_2d[left_valid_2d]):.2f}  "
              f"min={left_min_2d:.2f}  max={left_max_2d:.2f}  "
              f"n={int(left_valid_2d.sum())}")
    else:
        print("left : no valid frames")
    if right_valid_2d.sum() > 0:
        print(f"right mean={right_err_2d[right_valid_2d].mean():.2f}  "
              f"median={np.median(right_err_2d[right_valid_2d]):.2f}  "
              f"min={right_min_2d:.2f}  max={right_max_2d:.2f}  "
              f"n={int(right_valid_2d.sum())}")
    else:
        print("right: no valid frames")

    np.savez_compressed(
        out_npz,
        example=args.example,
        seq_key=seq_key,
        dyn_npz=dyn_npz,
        gt_left=uem_left,
        gt_right=uem_right,
        gt_root=uem_root,
        dyn_left_raw=dyn_left,
        dyn_right_raw=dyn_right,
        dyn_left_aligned=dyn_left_aligned,
        dyn_right_aligned=dyn_right_aligned,
        dyn_left_rel_aligned=dyn_left_rel_aligned,
        dyn_right_rel_aligned=dyn_right_rel_aligned,
        left_valid=left_valid,
        right_valid=right_valid,
        left_err_3d_m=left_err,
        right_err_3d_m=right_err,
        left_err_2d_px=left_err_2d,
        right_err_2d_px=right_err_2d,
        left_valid_2d=left_valid_2d,
        right_valid_2d=right_valid_2d,
        gt_left_uv=gt_l_uv,
        gt_right_uv=gt_r_uv,
        dyn_left_uv=dyn_l_uv,
        dyn_right_uv=dyn_r_uv,
        shared_scale=scale,
        shared_R=R,
        shared_t=t,
        intrinsics=np.array([fx, fy, cx, cy], dtype=np.float32),
        num_frames=n,
        num_align_points=len(src),
    )

    rows = ["frame,left_valid,right_valid,left_err_3d_m,right_err_3d_m,left_err_2d_px,right_err_2d_px"]
    for i in range(n):
        rows.append(
            f"{i},{int(left_valid[i])},{int(right_valid[i])},"
            f"{left_err[i]:.6f},{right_err[i]:.6f},"
            f"{left_err_2d[i]:.2f},{right_err_2d[i]:.2f}"
        )

    with open(out_csv, "w") as f:
        f.write("\n".join(rows))

    print()
    print(f"saved npz: {out_npz}")
    print(f"saved csv: {out_csv}")


if __name__ == "__main__":
    main()