#!/usr/bin/env python3
"""
Compute per-frame 3D and 2D projection errors between DynHaMR and UEM wrists.

Input
-----
--eval_npz : eval NPZ produced by run_merged.py --save_eval_npz.
  Required keys:
    dyn_left_aligned, dyn_right_aligned  -- DynHaMR wrists in UEM body-world
                                            (already similarity-transformed via
                                            the SLAM-trajectory alignment in run_merged.py)
    pred_left_wrist, pred_right_wrist    -- UEM-predicted wrists in UEM body-world
    gt_aria_traj_T                       -- GT Aria c2w matrices (T, 4, 4)
    left_valid, right_valid              -- DynHaMR validity masks

Output
------
NPZ with the four keys consumed by load_projection_error_guidance_weights
in run_merged.py:
  left_err3d_m, right_err3d_m    -- per-frame 3D L2 error (metres)
  left_err2d_px, right_err2d_px  -- per-frame 2D pixel error (GT Aria camera)

Projection convention mirrors project_wrist_world_to_pixels in run_merged.py:
  world → camera:  p_cam = R_c2w^T @ (p_world − t_c2w)
  pixel:           u = −fx · p_cam_x / z + cx
                   v = −fy · p_cam_y / z + cy

Typical workflow
----------------
1. run_merged.py --save_eval_npz          (no guidance — captures raw UEM predictions)
2. project_compare_dynhamr_uem_hands.py  --eval_npz <step1_out> --out <err_npz>
3. run_merged.py --use_projection_error_guidance --projection_error_npz <err_npz>
"""

import argparse
import csv
from pathlib import Path

import numpy as np


# Aria RGB camera intrinsics (1408 × 1408) — same as ARIA_RGB_INTRINS in run_merged.py
_ARIA_FX, _ARIA_FY, _ARIA_CX, _ARIA_CY = 396.0, 336.3, 717.3, 877.8


def project_world_to_pixels(points_world, aria_traj_T, fx, fy, cx, cy, min_z=0.05):
    """
    Project (T, 3) world-space points through GT Aria c2w matrices.

    Returns
    -------
    uv       : (T, 2) float64, pixel coordinates (nan where depth invalid)
    depth_ok : (T,) bool
    """
    R_c2w = aria_traj_T[:, :3, :3]   # (T, 3, 3)
    t_c2w = aria_traj_T[:, :3, 3]    # (T, 3)
    # world → camera
    pts_cam = np.einsum("tji,tj->ti", R_c2w, points_world - t_c2w)  # (T, 3)
    z = pts_cam[:, 2]
    depth_ok = z > min_z
    z_safe = np.where(depth_ok, z, min_z)
    # Aria-specific sign convention (both axes negated vs standard OpenCV)
    u = -fx * pts_cam[:, 0] / z_safe + cx
    v = -fy * pts_cam[:, 1] / z_safe + cy
    uv = np.stack([u, v], axis=-1)
    uv[~depth_ok] = np.nan
    return uv, depth_ok


def compute_side_errors(dyn_world, uem_world, aria_traj_T, valid,
                        fx, fy, cx, cy):
    """
    Per-frame 3D (metres) and 2D (pixels) errors between DynHaMR and UEM wrists.

    Both dyn_world and uem_world must already be in UEM body-world space.
    aria_traj_T is the GT Aria c2w trajectory.
    Invalid DynHaMR frames (valid=False) are returned as nan.
    """
    T = len(uem_world)
    err3d = np.full(T, np.nan, dtype=np.float32)
    err2d = np.full(T, np.nan, dtype=np.float32)

    if dyn_world is None or len(dyn_world) == 0:
        return err3d, err2d

    T_avail = len(aria_traj_T) if aria_traj_T is not None else len(uem_world)
    T = min(T, len(dyn_world), T_avail, len(valid))
    dyn_w = dyn_world[:T].astype(np.float64)
    uem_w = uem_world[:T].astype(np.float64)
    v     = valid[:T]

    # 3D error: straightforward L2 in body-world (metres)
    diff3d = np.linalg.norm(dyn_w - uem_w, axis=-1).astype(np.float32)
    err3d[:T] = np.where(v, diff3d, np.nan)

    # 2D error: project both sets through the GT Aria camera, then measure pixel distance.
    # Using GT camera avoids conflating DROID-SLAM camera estimation error with
    # actual wrist-position disagreement.
    # Skipped (left as nan) when gt_aria_traj_T was absent from the eval NPZ;
    # nan 2D errors become weight 0 in load_projection_error_guidance_weights
    # via np.maximum, so only the 3D weights drive guidance in that case.
    if aria_traj_T is not None:
        traj = aria_traj_T[:T].astype(np.float64)
        dyn_uv, dyn_depth_ok = project_world_to_pixels(dyn_w, traj, fx, fy, cx, cy)
        uem_uv, uem_depth_ok = project_world_to_pixels(uem_w, traj, fx, fy, cx, cy)
        valid_2d = v & dyn_depth_ok & uem_depth_ok
        diff2d = np.linalg.norm(dyn_uv - uem_uv, axis=-1).astype(np.float32)
        err2d[:T] = np.where(valid_2d, diff2d, np.nan)

    return err3d, err2d


def print_stats(name, arr):
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        print(f"  {name}: no valid frames")
    else:
        print(f"  {name}: mean={finite.mean():.4f}  median={np.median(finite):.4f}"
              f"  min={finite.min():.4f}  max={finite.max():.4f}  n={len(finite)}")


def main():
    ap = argparse.ArgumentParser(
        description="Compute per-frame DynHaMR vs UEM wrist projection errors."
    )
    ap.add_argument(
        "--eval_npz", required=True,
        help="Eval NPZ from run_merged.py --save_eval_npz.",
    )
    ap.add_argument("--out", required=True, help="Output NPZ path.")
    ap.add_argument("--fx", type=float, default=_ARIA_FX)
    ap.add_argument("--fy", type=float, default=_ARIA_FY)
    ap.add_argument("--cx", type=float, default=_ARIA_CX)
    ap.add_argument("--cy", type=float, default=_ARIA_CY)
    ap.add_argument("--csv", default=None, help="Optional per-frame CSV output.")
    args = ap.parse_args()

    data = np.load(args.eval_npz, allow_pickle=True)
    print(f"[INFO] Loaded: {args.eval_npz}")
    print(f"[INFO] Keys: {sorted(data.keys())}")

    fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy
    print(f"[INFO] Intrinsics: fx={fx} fy={fy} cx={cx} cy={cy}")

    if "gt_aria_traj_T" in data.files:
        aria_traj_T = data["gt_aria_traj_T"]
    else:
        aria_traj_T = None
        print("[WARN] gt_aria_traj_T not found in eval NPZ — "
              "2D pixel errors will be nan (only 3D errors used for guidance weights).")

    left_valid  = data["left_valid"].astype(bool)
    right_valid = data["right_valid"].astype(bool)

    # DynHaMR wrists are stored after the SLAM-trajectory similarity transform
    # that run_merged.py applies to bring them into UEM body-world space.
    # pred_{left,right}_wrist are the raw UEM predictions before any guidance.
    dyn_left  = data["dyn_left_aligned"]
    dyn_right = data["dyn_right_aligned"]
    uem_left  = data["pred_left_wrist"]
    uem_right = data["pred_right_wrist"]

    left_err3d,  left_err2d  = compute_side_errors(
        dyn_left,  uem_left,  aria_traj_T, left_valid,  fx, fy, cx, cy)
    right_err3d, right_err2d = compute_side_errors(
        dyn_right, uem_right, aria_traj_T, right_valid, fx, fy, cx, cy)

    print("\n[3D errors — metres]")
    print_stats("left ", left_err3d)
    print_stats("right", right_err3d)
    print("[2D errors — pixels (GT Aria camera)]")
    print_stats("left ", left_err2d)
    print_stats("right", right_err2d)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        # Keys consumed by load_projection_error_guidance_weights in run_merged.py
        left_err3d_m=left_err3d,
        right_err3d_m=right_err3d,
        left_err2d_px=left_err2d,
        right_err2d_px=right_err2d,
        # Provenance / debugging
        left_valid=left_valid[:len(left_err3d)],
        right_valid=right_valid[:len(right_err3d)],
        intrinsics=np.array([fx, fy, cx, cy], dtype=np.float32),
    )
    print(f"\n[INFO] Saved: {out}")

    if args.csv:
        T = max(len(left_err3d), len(right_err3d))
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "frame",
                "left_err3d_m", "left_err2d_px", "left_valid",
                "right_err3d_m", "right_err2d_px", "right_valid",
            ])
            for t in range(T):
                w.writerow([
                    t,
                    left_err3d[t]  if t < len(left_err3d)  else "",
                    left_err2d[t]  if t < len(left_err2d)  else "",
                    bool(left_valid[t])  if t < len(left_valid)  else "",
                    right_err3d[t] if t < len(right_err3d) else "",
                    right_err2d[t] if t < len(right_err2d) else "",
                    bool(right_valid[t]) if t < len(right_valid) else "",
                ])
        print(f"[INFO] Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
