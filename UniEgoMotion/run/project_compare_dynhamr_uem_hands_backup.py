#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def load_npz(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)


def find_key(data, candidates, required=True):
    keys = list(data.keys())
    for c in candidates:
        if c in keys:
            return c
    if required:
        raise KeyError(f"Could not find any of {candidates}. Available keys:\n{keys}")
    return None


def to_T3(x):
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[1] == 1 and x.shape[2] == 3:
        x = x[:, 0, :]
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"Expected shape (T,3), got {x.shape}")
    return x.astype(np.float64)


def project_points(points_cam, K=None, fx=None, fy=None, cx=None, cy=None):
    """
    points_cam: (T,3), camera coordinates.
    Assumes x right, y down/up depending convention, z forward.
    Projection:
        u = fx*x/z + cx
        v = fy*y/z + cy
    """
    pts = np.asarray(points_cam, dtype=np.float64)
    z = pts[:, 2]

    valid = np.isfinite(pts).all(axis=1) & (z > 1e-6)

    if K is not None:
        K = np.asarray(K, dtype=np.float64)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    if None in [fx, fy, cx, cy]:
        raise ValueError("Need either --K or --fx --fy --cx --cy")

    uv = np.full((len(pts), 2), np.nan, dtype=np.float64)
    uv[valid, 0] = fx * pts[valid, 0] / z[valid] + cx
    uv[valid, 1] = fy * pts[valid, 1] / z[valid] + cy
    return uv, valid


def maybe_invert_yz(points):
    """
    Optional convention fix:
    Some pipelines use OpenGL-like camera coordinates:
        x right, y up, z backward
    while image projection often assumes:
        x right, y down, z forward.
    This converts by flipping y and z.
    """
    out = points.copy()
    out[:, 1] *= -1
    out[:, 2] *= -1
    return out


def best_constant_offset_align(pred, gt, valid):
    """
    Optional translation alignment in camera space.
    Useful if both are in camera coordinates but have a constant root/hand offset.
    """
    offset = np.nanmean(gt[valid] - pred[valid], axis=0)
    return pred + offset, offset


def summarize_errors(name, err):
    err = err[np.isfinite(err)]
    if len(err) == 0:
        return {
            f"{name}_count": 0,
            f"{name}_mean": np.nan,
            f"{name}_median": np.nan,
            f"{name}_min": np.nan,
            f"{name}_max": np.nan,
        }
    return {
        f"{name}_count": int(len(err)),
        f"{name}_mean": float(np.mean(err)),
        f"{name}_median": float(np.median(err)),
        f"{name}_min": float(np.min(err)),
        f"{name}_max": float(np.max(err)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dyn_npz", required=True, help="Dyn-HaMR hand_guidance npz")
    ap.add_argument("--uem_npz", required=True, help="UniEgoMotion output npz")
    ap.add_argument("--out", required=True, help="Output npz")
    ap.add_argument("--csv", default=None, help="Optional per-frame CSV")

    ap.add_argument("--fx", type=float)
    ap.add_argument("--fy", type=float)
    ap.add_argument("--cx", type=float)
    ap.add_argument("--cy", type=float)
    ap.add_argument("--K", default=None, help="Optional .npy intrinsics matrix")

    ap.add_argument("--align_offset", action="store_true")
    ap.add_argument("--try_flip_yz", action="store_true")

    args = ap.parse_args()

    dyn = load_npz(args.dyn_npz)
    uem = load_npz(args.uem_npz)

    K = np.load(args.K) if args.K else None

    # Dyn-HaMR guidance keys created by your previous scripts
    dyn_l_key = find_key(dyn, [
        "dyn_left_aligned",
        "left_trans_cam", "left_wrist_cam", "left_hand_cam"
    ], required=False)

    dyn_r_key = find_key(dyn, [
        "dyn_right_aligned",
        "right_trans_cam", "right_wrist_cam", "right_hand_cam"
    ], required=False)

    # UniEgoMotion keys: adjust here if your exported npz uses another name
    uem_l_key = find_key(uem, [
        "uem_left",
        "pred_left_wrist", "gt_left_wrist", "guided_left_wrist",
        "left_wrist_cam", "left_hand_cam", "left_trans_cam", "joints_left_cam"
    ], required=False)

    uem_r_key = find_key(uem, [
        "uem_right",
        "pred_right_wrist", "gt_right_wrist", "guided_right_wrist",
        "right_wrist_cam", "right_hand_cam", "right_trans_cam", "joints_right_cam"
    ], required=False)

    if dyn_l_key is None and dyn_r_key is None:
        raise KeyError(f"No Dyn-HaMR hand camera keys found. Keys: {list(dyn.keys())}")
    if uem_l_key is None and uem_r_key is None:
        raise KeyError(f"No UniEgoMotion hand camera keys found. Keys: {list(uem.keys())}")

    results = {}
    rows = []

    for side, dk, uk in [
        ("left", dyn_l_key, uem_l_key),
        ("right", dyn_r_key, uem_r_key),
    ]:
        if dk is None or uk is None:
            print(f"[WARN] skipping {side}: dyn={dk}, uem={uk}")
            continue

        dyn_xyz = to_T3(dyn[dk])
        uem_xyz = to_T3(uem[uk])

        T = min(len(dyn_xyz), len(uem_xyz))
        dyn_xyz = dyn_xyz[:T]
        uem_xyz = uem_xyz[:T]

        if args.try_flip_yz:
            candidates = {
                "normal": dyn_xyz,
                "flip_yz": maybe_invert_yz(dyn_xyz),
            }
        else:
            candidates = {"normal": dyn_xyz}

        best = None

        for mode, dyn_candidate in candidates.items():
            dyn_use = dyn_candidate.copy()

            base_valid = np.isfinite(dyn_use).all(axis=1) & np.isfinite(uem_xyz).all(axis=1)

            if args.align_offset:
                dyn_use, offset = best_constant_offset_align(dyn_use, uem_xyz, base_valid)
            else:
                offset = np.zeros(3)

            dyn_uv, v1 = project_points(dyn_use, K=K, fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy)
            uem_uv, v2 = project_points(uem_xyz, K=K, fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy)

            valid = base_valid & v1 & v2

            err3d = np.linalg.norm(dyn_use - uem_xyz, axis=1)
            err2d = np.linalg.norm(dyn_uv - uem_uv, axis=1)

            score = np.nanmean(err2d[valid]) if np.any(valid) else np.inf

            if best is None or score < best["score"]:
                best = {
                    "mode": mode,
                    "offset": offset,
                    "dyn_xyz": dyn_use,
                    "uem_xyz": uem_xyz,
                    "dyn_uv": dyn_uv,
                    "uem_uv": uem_uv,
                    "valid": valid,
                    "err3d": err3d,
                    "err2d": err2d,
                    "score": score,
                }

        print(f"\n===== {side.upper()} =====")
        print("Dyn key:", dk)
        print("UEM key:", uk)
        print("mode:", best["mode"])
        print("offset:", best["offset"])

        s3 = summarize_errors(f"{side}_err3d_m", best["err3d"][best["valid"]])
        s2 = summarize_errors(f"{side}_err2d_px", best["err2d"][best["valid"]])
        print(s3)
        print(s2)

        results[f"{side}_dyn_xyz_cam"] = best["dyn_xyz"]
        results[f"{side}_uem_xyz_cam"] = best["uem_xyz"]
        results[f"{side}_dyn_uv"] = best["dyn_uv"]
        results[f"{side}_uem_uv"] = best["uem_uv"]
        results[f"{side}_valid"] = best["valid"]
        results[f"{side}_err3d_m"] = best["err3d"]
        results[f"{side}_err2d_px"] = best["err2d"]
        results[f"{side}_mode"] = np.array(best["mode"])
        results[f"{side}_offset"] = best["offset"]

        for t in range(T):
            rows.append([
                t, side, bool(best["valid"][t]),
                best["err3d"][t], best["err2d"][t],
                *best["dyn_xyz"][t], *best["uem_xyz"][t],
                *best["dyn_uv"][t], *best["uem_uv"][t],
            ])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **results)
    print("\nsaved:", out)

    if args.csv:
        import csv
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "frame", "side", "valid",
                "err3d_m", "err2d_px",
                "dyn_x", "dyn_y", "dyn_z",
                "uem_x", "uem_y", "uem_z",
                "dyn_u", "dyn_v",
                "uem_u", "uem_v",
            ])
            w.writerows(rows)
        print("saved:", csv_path)


if __name__ == "__main__":
    main()