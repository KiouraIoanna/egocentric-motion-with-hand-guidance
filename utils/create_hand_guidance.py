"""
Convert Dyn-HaMR output to a hand_guidance.npz consumed by run_with_guidance.py.

Usage (explicit paths):
    python utils/create_hand_guidance.py \
        --dyn_npz  <path>/smooth_fit/<seq>_000300_world_results.npz \
        --track_info <path>/track_info.json \
        --out cooking_vids_uni/hand_guidance/<seq>_hand_guidance.npz \
        --seq <seq>

Usage (auto-discover latest checkpoint from run dir):
    python utils/create_hand_guidance.py \
  --run_dir Dyn-HaMR/outputs/logs/video-custom/2026-05-22/indiana_cooking_09_2___10257___11112st160_uem80-all-shot-0-0-500 \
  --out cooking_vids_uni/hand_guidance/indiana_cooking_09_2___10257___11112st160_uem80_hand_guidance.npz \
  --seq indiana_cooking_09_2___10257___11112st160_uem80

New fields over the original version:
    left/right_trans_cam        wrist world → camera space  (T, 3)
    left/right_root_orient_rotmat  axis-angle → rotation matrix (T, 3, 3)
    left/right_w2c              4×4 world-to-camera matrix  (T, 4, 4)
    left/right_latent_pose      HMP latent pose before MANO decode (T, 15, 3)
    left/right_init_body_pose   initial pose before optimization (T, 15, 3)
    left/right_wrist_vel_world  finite-diff velocity in world space (T, 3)
    left/right_wrist_vel_cam    finite-diff velocity in camera space (T, 3)
    left/right_handedness_conf  scalar: how consistently left/right the track was
    has_left / has_right        bool: whether each hand was detected
    num_dyn_frames              frames in world_results (may differ from window size)
    num_dyn_tracks              number of tracks Dyn-HaMR found
    dyn_npz_path                path to the source world_results.npz (for traceability)
"""

import argparse
import glob
import json
import os
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Math helpers (pure numpy, no torch dependency)
# ---------------------------------------------------------------------------

def axis_angle_to_rotmat(aa):
    """Rodrigues formula: (..., 3) → (..., 3, 3)."""
    shape = aa.shape[:-1]
    aa = aa.reshape(-1, 3).astype(np.float64)
    angle = np.linalg.norm(aa, axis=-1, keepdims=True).clip(min=1e-8)
    axis = aa / angle
    cos = np.cos(angle)
    sin = np.sin(angle)
    K = np.zeros((len(aa), 3, 3), dtype=np.float64)
    K[:, 0, 1] = -axis[:, 2];  K[:, 0, 2] =  axis[:, 1]
    K[:, 1, 0] =  axis[:, 2];  K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1];  K[:, 2, 1] =  axis[:, 0]
    I = np.eye(3, dtype=np.float64)[None]
    outer = np.einsum("ni,nj->nij", axis, axis)
    R = cos[:, :, None] * I + (1 - cos[:, :, None]) * outer + sin[:, :, None] * K
    return R.reshape(*shape, 3, 3).astype(np.float32)


def build_w2c(cam_R, cam_t):
    """(T,3,3), (T,3) → (T,4,4) homogeneous world-to-camera matrix."""
    T = len(cam_R)
    w2c = np.zeros((T, 4, 4), dtype=np.float32)
    w2c[:, :3, :3] = cam_R
    w2c[:, :3,  3] = cam_t
    w2c[:,  3,  3] = 1.0
    return w2c


def world_to_cam(cam_R, cam_t, pts):
    """(T,3,3), (T,3), (T,3) → (T,3)  pts in camera space."""
    return np.einsum("tij,tj->ti", cam_R, pts) + cam_t


def central_diff_velocity(pts, valid):
    """
    Finite-difference velocity using central differences on valid frames,
    falling back to forward/backward differences at boundaries.
    Invalid frames get a zero velocity.
    """
    T = len(pts)
    vel = np.zeros_like(pts, dtype=np.float32)
    for t in range(T):
        if not valid[t]:
            continue
        has_prev = t > 0 and valid[t - 1]
        has_next = t < T - 1 and valid[t + 1]
        if has_prev and has_next:
            vel[t] = 0.5 * (pts[t + 1] - pts[t - 1])
        elif has_next:
            vel[t] = pts[t + 1] - pts[t]
        elif has_prev:
            vel[t] = pts[t] - pts[t - 1]
    return vel


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def find_latest_world_results(run_dir):
    """Return the highest-iteration world_results.npz from <run_dir>/smooth_fit/."""
    smooth_fit = os.path.join(run_dir, "smooth_fit")
    if not os.path.isdir(smooth_fit):
        raise FileNotFoundError(f"smooth_fit/ not found under {run_dir}")
    candidates = sorted(glob.glob(os.path.join(smooth_fit, "*_world_results.npz")))
    if not candidates:
        raise FileNotFoundError(f"No *_world_results.npz in {smooth_fit}")
    return candidates[-1]  # filenames are zero-padded by iteration → last = latest


def find_track_info(run_dir):
    p = os.path.join(run_dir, "track_info.json")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"track_info.json not found in {run_dir}")
    return p


# ---------------------------------------------------------------------------
# Per-track extraction
# ---------------------------------------------------------------------------

def slice_to(arr, T_out):
    """Truncate or zero-pad the first axis of arr to length T_out."""
    T = len(arr)
    if T >= T_out:
        return arr[:T_out].astype(np.float32)
    pad = np.zeros((T_out - T, *arr.shape[1:]), dtype=np.float32)
    return np.concatenate([arr.astype(np.float32), pad], axis=0)


def extract_track(dyn, track_idx, T_out):
    """Pull all per-frame fields for one track and derive enriched fields."""
    trans          = slice_to(dyn["trans"][track_idx],          T_out)  # (T,3)
    root_orient    = slice_to(dyn["root_orient"][track_idx],    T_out)  # (T,3)
    pose_body      = slice_to(dyn["pose_body"][track_idx],      T_out)  # (T,15,3)
    latent_pose    = slice_to(dyn["latent_pose"][track_idx],    T_out)  # (T,15,3)
    init_body_pose = slice_to(dyn["init_body_pose"][track_idx], T_out)  # (T,15,3)
    is_right_arr   = slice_to(dyn["is_right"][track_idx],       T_out)  # (T,)
    cam_R          = slice_to(dyn["cam_R"][track_idx],          T_out)  # (T,3,3)
    cam_t          = slice_to(dyn["cam_t"][track_idx],          T_out)  # (T,3)

    # betas is (num_tracks, 10) — not time-varying
    betas = dyn["betas"][track_idx].astype(np.float32)  # (10,)

    # Derived
    root_orient_rotmat = axis_angle_to_rotmat(root_orient)   # (T,3,3)
    w2c                = build_w2c(cam_R, cam_t)             # (T,4,4)
    trans_cam          = world_to_cam(cam_R, cam_t, trans)   # (T,3)

    return {
        "trans":             trans,
        "trans_cam":         trans_cam,
        "root_orient":       root_orient,
        "root_orient_rotmat": root_orient_rotmat,
        "pose_body":         pose_body,
        "latent_pose":       latent_pose,
        "init_body_pose":    init_body_pose,
        "betas":             betas,
        "is_right":          is_right_arr,
        "cam_R":             cam_R,
        "cam_t":             cam_t,
        "w2c":               w2c,
    }


def make_empty_track(T_out):
    """All-zero placeholder for a hand that was not detected."""
    return {
        "trans":              np.zeros((T_out, 3),     dtype=np.float32),
        "trans_cam":          np.zeros((T_out, 3),     dtype=np.float32),
        "root_orient":        np.zeros((T_out, 3),     dtype=np.float32),
        "root_orient_rotmat": np.zeros((T_out, 3, 3),  dtype=np.float32),
        "pose_body":          np.zeros((T_out, 15, 3), dtype=np.float32),
        "latent_pose":        np.zeros((T_out, 15, 3), dtype=np.float32),
        "init_body_pose":     np.zeros((T_out, 15, 3), dtype=np.float32),
        "betas":              np.zeros(10,              dtype=np.float32),
        "is_right":           np.zeros(T_out,           dtype=np.float32),
        "cam_R":              np.zeros((T_out, 3, 3),  dtype=np.float32),
        "cam_t":              np.zeros((T_out, 3),     dtype=np.float32),
        "w2c":                np.zeros((T_out, 4, 4),  dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    # Original CLI (explicit paths)
    parser.add_argument("--dyn_npz",    default=None, help="Path to *_world_results.npz")
    parser.add_argument("--track_info", default=None, help="Path to track_info.json")
    # New: auto-discover from run directory
    parser.add_argument("--run_dir",    default=None,
                        help="Dyn-HaMR run directory; auto-picks latest world_results.npz "
                             "and track_info.json")
    # Always required
    parser.add_argument("--out",  required=True, help="Output .npz path")
    parser.add_argument("--seq",  required=True, help="Sequence name tag written into the file")
    parser.add_argument("--fps",  type=float, default=10.0)
    args = parser.parse_args()

    # --- Resolve input paths ---
    if args.run_dir is not None:
        dyn_npz_path    = find_latest_world_results(args.run_dir)
        track_info_path = find_track_info(args.run_dir)
        print(f"[INFO] world_results : {dyn_npz_path}")
        print(f"[INFO] track_info    : {track_info_path}")
    elif args.dyn_npz is not None and args.track_info is not None:
        dyn_npz_path    = args.dyn_npz
        track_info_path = args.track_info
    else:
        parser.error("Provide either --run_dir or both --dyn_npz and --track_info")

    dyn = np.load(dyn_npz_path, allow_pickle=True)
    with open(track_info_path) as f:
        track_info = json.load(f)

    # --- Track identification by majority-vote handedness ---
    is_right        = dyn["is_right"]               # (num_tracks, T_dyn)
    num_tracks, T_dyn = is_right.shape
    track_is_right  = is_right.mean(axis=1)          # (num_tracks,)

    left_candidates  = np.where(track_is_right <  0.5)[0]
    right_candidates = np.where(track_is_right >= 0.5)[0]

    has_left  = len(left_candidates)  > 0
    has_right = len(right_candidates) > 0

    if not has_left and not has_right:
        raise ValueError(f"No tracks at all in {dyn_npz_path}")
    if not has_left:
        print("[WARN] No left-hand track found — left guidance will be all-invalid.")
    if not has_right:
        print("[WARN] No right-hand track found — right guidance will be all-invalid.")

    # Use whichever candidate exists; when a hand is missing the idx is a dummy
    left_idx  = int(left_candidates[0])  if has_left  else int(right_candidates[0])
    right_idx = int(right_candidates[0]) if has_right else int(left_candidates[0])

    # --- Validity masks from track_info ---
    # vis_mask length may differ from T_dyn (e.g. window=80 but Dyn-HaMR ran 76 frames).
    # Use min(vis_mask_length, T_dyn) as the authoritative output length.
    def get_vis_mask(track_idx):
        raw = np.array(track_info["tracks"][str(track_idx)]["vis_mask"], dtype=bool)
        T   = min(len(raw), T_dyn)
        return raw[:T], T

    if has_left:
        left_valid_raw,  T_left  = get_vis_mask(left_idx)
    else:
        T_left = T_dyn
        left_valid_raw  = np.zeros(T_left,  dtype=bool)

    if has_right:
        right_valid_raw, T_right = get_vis_mask(right_idx)
    else:
        T_right = T_dyn
        right_valid_raw = np.zeros(T_right, dtype=bool)

    T_out = min(T_left, T_right)
    left_valid  = left_valid_raw[:T_out]
    right_valid = right_valid_raw[:T_out]

    # --- Extract per-track data ---
    left_data  = extract_track(dyn, left_idx,  T_out) if has_left  else make_empty_track(T_out)
    right_data = extract_track(dyn, right_idx, T_out) if has_right else make_empty_track(T_out)

    # Velocities need the validity mask so they skip invalid frames
    left_vel_world  = central_diff_velocity(left_data["trans"],     left_valid)
    left_vel_cam    = central_diff_velocity(left_data["trans_cam"],  left_valid)
    right_vel_world = central_diff_velocity(right_data["trans"],    right_valid)
    right_vel_cam   = central_diff_velocity(right_data["trans_cam"], right_valid)

    # Scalar handedness confidence: fraction of frames classified consistently
    left_conf  = float(1.0 - left_data["is_right"].mean())  if has_left  else 0.0
    right_conf = float(right_data["is_right"].mean())        if has_right else 0.0

    # --- Assemble output ---
    out = {
        # Metadata
        "seq":                args.seq,
        "fps":                args.fps,
        "num_frames":         T_out,
        "num_dyn_frames":     T_dyn,
        "num_dyn_tracks":     num_tracks,
        "has_left":           has_left,
        "has_right":          has_right,
        "left_track_idx":     left_idx,
        "right_track_idx":    right_idx,
        "left_handedness_conf":  left_conf,
        "right_handedness_conf": right_conf,
        "world_scale":        dyn["world_scale"],
        "intrins":            dyn["intrins"].astype(np.float32),
        "dyn_npz_path":       str(dyn_npz_path),

        # Validity
        "left_valid":         left_valid,
        "right_valid":        right_valid,

        # Left hand — world-space
        "left_trans":               left_data["trans"],
        "left_root_orient":         left_data["root_orient"],
        "left_root_orient_rotmat":  left_data["root_orient_rotmat"],
        "left_pose_body":           left_data["pose_body"],
        "left_latent_pose":         left_data["latent_pose"],
        "left_init_body_pose":      left_data["init_body_pose"],
        "left_betas":               left_data["betas"],
        "left_is_right":            left_data["is_right"],
        # Left hand — camera-space & transforms
        "left_cam_R":               left_data["cam_R"],
        "left_cam_t":               left_data["cam_t"],
        "left_w2c":                 left_data["w2c"],
        "left_trans_cam":           left_data["trans_cam"],
        # Left hand — velocities
        "left_wrist_vel_world":     left_vel_world,
        "left_wrist_vel_cam":       left_vel_cam,

        # Right hand — world-space
        "right_trans":              right_data["trans"],
        "right_root_orient":        right_data["root_orient"],
        "right_root_orient_rotmat": right_data["root_orient_rotmat"],
        "right_pose_body":          right_data["pose_body"],
        "right_latent_pose":        right_data["latent_pose"],
        "right_init_body_pose":     right_data["init_body_pose"],
        "right_betas":              right_data["betas"],
        "right_is_right":           right_data["is_right"],
        # Right hand — camera-space & transforms
        "right_cam_R":              right_data["cam_R"],
        "right_cam_t":              right_data["cam_t"],
        "right_w2c":                right_data["w2c"],
        "right_trans_cam":          right_data["trans_cam"],
        # Right hand — velocities
        "right_wrist_vel_world":    right_vel_world,
        "right_wrist_vel_cam":      right_vel_cam,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)

    print(f"\nSaved : {args.out}")
    print(f"Frames: {T_out}  (Dyn-HaMR had {T_dyn})")
    print(f"Left   track={left_idx}  valid={left_valid.sum()}/{T_out}  "
          f"conf={left_conf:.3f}  detected={has_left}")
    print(f"Right  track={right_idx}  valid={right_valid.sum()}/{T_out}  "
          f"conf={right_conf:.3f}  detected={has_right}")
    print("New fields: trans_cam, root_orient_rotmat, w2c, latent_pose, "
          "init_body_pose, wrist_vel_world, wrist_vel_cam")


if __name__ == "__main__":
    main()
