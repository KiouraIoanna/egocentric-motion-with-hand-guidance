#!/usr/bin/env python3
"""
Recompute Dyn-HaMR vs UEM projection-error npz files with the corrections that
the original generator driver (make_all_projection_errors.sh) never applied.

This is a NEW standalone script. It does not modify the original generator
(UniEgoMotion/run/project_compare_dynhamr_uem_hands.py), its driver, or the
existing projection_errors/ files. It reads the existing npz and writes corrected
copies to a NEW output directory.

Corrections vs the original run:
  1. Constant-offset alignment in camera space. The original generator HAS this
     (best_constant_offset_align) but only under --align_offset, which the driver
     never passed -> a constant ~0.29 m Dyn<->UEM bias was left in, saturating
     every 3D weight to 1.0.
  2. Convention search: pick, per side, the axis-flip of the Dyn points that
     minimises the offset-corrected 3D residual (original had --try_flip_yz,
     also never passed).
  3. Correct Aria RGB intrinsics + projection model. The driver passed
     fx=fy=960, cx=960, cy=540 (a 1920x1080 pinhole) but the real pipeline uses
     ARIA_RGB_INTRINS = [396.0, 336.3, 717.3, 877.8] (1408x1408) with the model
     u = -fx*(x/z)+cx,  v = -fy*(y/z)+cy   (note the negative sign).

The ORIGINAL UEM source npz (/work/scratch/ayopan/...) has been deleted, so the
original generator cannot be re-run. We instead recompute directly from the
EXISTING projection_error npz, which already stores both raw camera-space wrist
tracks (left/right_dyn_xyz_cam, left/right_uem_xyz_cam, *_valid). Only err3d/err2d
(+ derived uv/offset/mode/valid) are changed; the stored UEM points are reused.

Reads:  <in_dir>/*_projection_error.npz   (existing files, unmodified)
Writes: <out_dir>/*_projection_error.npz   (corrected copies, same filenames)
"""

import argparse
import glob
import os
import numpy as np

# Real Aria RGB intrinsics + projection model, copied from run_merged.py.
ARIA_FX, ARIA_FY, ARIA_CX, ARIA_CY = 396.0, 336.3, 717.3, 877.8

_FLIP_MODES = {
    "normal":  np.array([1.0,  1.0,  1.0]),
    "flip_y":  np.array([1.0, -1.0,  1.0]),
    "flip_z":  np.array([1.0,  1.0, -1.0]),
    "flip_yz": np.array([1.0, -1.0, -1.0]),
    "flip_xy": np.array([-1.0, -1.0, 1.0]),
}


def project_aria(pts):
    """u = -fx*(x/z)+cx, v = -fy*(y/z)+cy. Returns uv (T,2), valid (T,)."""
    z = pts[:, 2]
    valid = np.isfinite(pts).all(axis=1) & (np.abs(z) > 1e-6)
    uv = np.full((len(pts), 2), np.nan)
    uv[valid, 0] = -ARIA_FX * pts[valid, 0] / z[valid] + ARIA_CX
    uv[valid, 1] = -ARIA_FY * pts[valid, 1] / z[valid] + ARIA_CY
    return uv, valid


def offset_align(dyn, uem, valid):
    if valid.sum() == 0:
        return dyn.copy(), np.zeros(3)
    off = np.nanmean(uem[valid] - dyn[valid], axis=0)
    return dyn + off, off


def best_correction(dyn, uem, valid):
    """Pick the flip mode whose offset-corrected 3D residual is smallest."""
    best = None
    for mode, sign in _FLIP_MODES.items():
        cand = dyn * sign
        aligned, off = offset_align(cand, uem, valid)
        r = np.linalg.norm(aligned - uem, axis=1)
        score = np.nanmedian(r[valid]) if valid.sum() else np.inf
        if best is None or score < best[0]:
            best = (score, mode, sign, aligned, off)
    return best  # (score, mode, sign, aligned_dyn, offset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",
                    default="/work/courses/digital_human/team7/my_coord_attempt/projection_errors")
    ap.add_argument("--out_dir",
                    default="/work/courses/digital_human/team7/my_coord_attempt/projection_errors_fixed")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.in_dir, "*_projection_error.npz")))
    if not files:
        raise SystemExit(f"No *_projection_error.npz in {args.in_dir}")

    print(f"Recomputing {len(files)} files: {args.in_dir} -> {args.out_dir}\n")
    for f in files:
        x = np.load(f, allow_pickle=True)
        out = {k: x[k] for k in x.files}  # start from existing, overwrite err fields
        name = os.path.basename(f)
        line = [name[:42]]
        for s in ["left", "right"]:
            dk, uk, vk = f"{s}_dyn_xyz_cam", f"{s}_uem_xyz_cam", f"{s}_valid"
            if dk not in x.files:
                continue
            dyn = np.asarray(x[dk], float)
            uem = np.asarray(x[uk], float)
            valid = np.asarray(x[vk], bool)

            score, mode, sign, dyn_corr, off = best_correction(dyn, uem, valid)
            dyn_uv, v1 = project_aria(dyn_corr)
            uem_uv, v2 = project_aria(uem)
            vfin = valid & v1 & v2

            err3d = np.linalg.norm(dyn_corr - uem, axis=1)
            err2d = np.linalg.norm(dyn_uv - uem_uv, axis=1)

            out[f"{s}_dyn_xyz_cam"] = dyn_corr
            out[f"{s}_dyn_uv"] = dyn_uv
            out[f"{s}_uem_uv"] = uem_uv
            out[f"{s}_valid"] = vfin
            out[f"{s}_err3d_m"] = err3d
            out[f"{s}_err2d_px"] = err2d
            out[f"{s}_mode"] = np.array(mode)
            out[f"{s}_offset"] = off

            m3 = np.nanmedian(err3d[vfin]) if vfin.sum() else np.nan
            m2 = np.nanmedian(err2d[vfin]) if vfin.sum() else np.nan
            line.append(f"{s}:{mode}|3d~{m3:.3f}m|2d~{m2:.0f}px|off={np.linalg.norm(off):.2f}")
        print("  ".join(line))
        np.savez_compressed(os.path.join(args.out_dir, name), **out)

    print(f"\nDone. Corrected files in {args.out_dir}")


if __name__ == "__main__":
    main()
