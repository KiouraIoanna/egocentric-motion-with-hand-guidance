"""
Merged Dyn-HaMR + UniEgoMotion guidance pipeline.

Combines:
  - run_guidance_camspace.py: trajectory-aligned camspace (no chicken-and-egg
    bootstrap), 2D MediaPipe reprojection in smpl_opt with calibrated Aria
    intrinsics, Gaussian-smoothed detections, confidence-weighted reproj loss,
    wrist-position acceleration smoothness, Dyn-HaMR MANO finger-pose
    injection for rendering.
  - run_with_guidance_a.py (partner): projection-error-driven per-frame trust
    weighting for the 3D-wrist guidance loss (read from precomputed npz).

These are orthogonal contributions. Reliability gating from
`compute_framewise_dyn_guidance_reliability` (vel/acc/sep consistency) and
projection-error weighting from `load_projection_error_guidance_weights`
(absolute 3D/2D errors) are complementary: one filters Dyn-HaMR jitter, the
other amplifies Dyn-HaMR where UEM is clearly wrong.

Trajectory-aligned coordinate alignment for Dyn-HaMR + UniEgoMotion guidance.

Key difference from run_with_guidance.py:
  BEFORE: fit a similarity transform from Dyn-HaMR world → UEM body frame
          using the first clean diffusion pass as target (bootstrap / chicken-and-egg).
  NOW:    fit a similarity transform from Dyn-HaMR SLAM world → UEM body-world
          using CAMERA TRAJECTORY POSITIONS as alignment points.
          Dyn-HaMR's DROID-SLAM camera and UEM's Aria camera both track the same
          physical device, so aligning their estimated trajectories (80 points)
          gives a far more stable transform than aligning 2 predicted wrists.
          No model call is needed — guidance runs in a SINGLE diffusion pass.

Falls back to the original similarity-transform bootstrap when w2c is absent.
"""

import copy
import os
import sys
import argparse
import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger

# ---------------------------------------------------------------------------
# Path setup — same as run_with_guidance.py (run from UniEgoMotion/ root)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.defaults import get_cfg
from dataset.ee4d_motion_dataset import EE4D_Motion_DataModule, careful_collate_fn
from dataset.ee4d_motion_dataset import EE4D_Motion_Dataset
from module.uem_module import UEM_Module, UEM_Module_TwoStage
from utils.torch_utils import to_device
from utils import rotation_conversions as rc
from utils.vis_utils import save_video, visualize_sequence_blender, visualize_sequence, pad_filler, pad_filler_traj
from dataset.smpl_utils import evaluate_smpl, get_smpl
from dataset.canonicalization import rot_trans_to_matrix, rotation_to_make_this_forward_batch

# ---------------------------------------------------------------------------
# Aria RGB camera intrinsics (1408×1408 image space)
# Numerically calibrated from GT SMPL wrists + MediaPipe 2D detections.
# Projection model: u = -fx*(x/z) + cx,  v = -fy*(y/z) + cy
# ---------------------------------------------------------------------------
ARIA_RGB_INTRINS = np.array([396.0, 336.3, 717.3, 877.8], dtype=np.float32)  # fx, fy, cx, cy


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_vis_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", action="extend", nargs="+", required=True,
                        help="seq_name:start_idx, repeatable. Multiple values per flag also OK.")
    parser.add_argument("--hand_guidance_dir", default=None)
    parser.add_argument("--save_eval_npz", action="store_true")

    # Guidance mode
    parser.add_argument("--use_traj_align", action="store_true", default=True,
                        help="Align Dyn-HaMR SLAM world → UEM body-world via camera trajectory "
                             "fitting (requires left/right_w2c in hand_guidance.npz).")
    parser.add_argument("--no_traj_align", dest="use_traj_align", action="store_false",
                        help="Fall back to similarity-transform bootstrap (original approach).")

    parser.add_argument("--run_guidance_opt", action="store_true")
    parser.add_argument("--run_diffusion_wrist_guidance", action="store_true")
    parser.add_argument("--diffusion_wrist_guidance_strength", type=float, default=0.05)
    parser.add_argument("--diffusion_wrist_guidance_start_frac", type=float, default=0.7)
    parser.add_argument("--guidance_iters", type=int, default=300)
    parser.add_argument("--guidance_lr", type=float, default=0.03)
    parser.add_argument("--w_prior", type=float, default=1.0)
    parser.add_argument("--w_dyn", type=float, default=1.0)
    parser.add_argument("--w_vel", type=float, default=0.5)
    parser.add_argument("--w_acc", type=float, default=0.0)
    parser.add_argument("--w_sep", type=float, default=0.2)
    parser.add_argument("--w_smooth", type=float, default=0.05)
    parser.add_argument("--dyn_pos_min_scale", type=float, default=0.5)
    parser.add_argument("--dyn_pos_max_scale", type=float, default=2.0)
    parser.add_argument("--bad_align_w_dyn", type=float, default=0.0)
    parser.add_argument("--disable_dyn_reliability_gate", action="store_true")
    parser.add_argument("--dyn_frame_gate_window", type=int, default=7)
    parser.add_argument("--dyn_vel_good_ratio", type=float, default=1.25)
    parser.add_argument("--dyn_vel_bad_ratio", type=float, default=2.0)
    parser.add_argument("--dyn_acc_good_ratio", type=float, default=2.0)
    parser.add_argument("--dyn_acc_bad_ratio", type=float, default=5.0)
    parser.add_argument("--dyn_sep_good_ratio", type=float, default=0.22)
    parser.add_argument("--dyn_sep_bad_ratio", type=float, default=0.35)
    # 2D MediaPipe guidance
    parser.add_argument("--guidance_2d_dir",
                        default="/work/courses/digital_human/team7/cooking_vids_uni/hand_guidance_2d",
                        help="Directory containing *_2d_wrist_guidance.npz files.")
    parser.add_argument("--w_reproj_2d", type=float, default=5.0,
                        help="Weight for 2D reprojection loss in smpl_opt (0 = disabled).")
    parser.add_argument("--reproj_2d_min_conf", type=float, default=0.5,
                        help="Minimum MediaPipe confidence to trust a 2D detection.")
    parser.add_argument("--mp_smooth_sigma", type=float, default=1.0,
                        help="Gaussian sigma (in frames) for smoothing MediaPipe 2D detections "
                             "before reproj loss. 0 = no smoothing.")
    parser.add_argument("--w_smpl_smooth_pos", type=float, default=50.0,
                        help="Weight for wrist-position acceleration smoothness in smpl_opt.")
    parser.add_argument("--w_smpl_wrist_orient", type=float, default=0.0,
                        help="Weight for Dyn-HaMR wrist-orientation loss in smpl_opt. "
                             "Disabled by default: empirically the SMPL-X-wrist vs MANO-root "
                             "canonical-pose offset makes the loss systematically biased and "
                             "regresses wrist MPJPE. Kept available for future experimentation.")
    parser.add_argument("--use_dyn_hand_pose", action="store_true", default=True,
                        help="Replace UEM's PCA-decoded hand pose with Dyn-HaMR's MANO "
                             "finger articulation on frames where Dyn-HaMR is valid.")
    parser.add_argument("--no_dyn_hand_pose", dest="use_dyn_hand_pose", action="store_false")

    # Projection-error-driven per-frame trust weighting (from run_with_guidance_a.py).
    # Reads precomputed per-frame 3D + 2D projection errors from an .npz and
    # amplifies the 3D wrist guidance loss in frames where UEM is far off.
    parser.add_argument("--use_projection_error_guidance", action="store_true",
                        help="Enable per-frame trust weighting of the 3D wrist guidance "
                             "loss using precomputed projection errors.")
    parser.add_argument("--projection_error_dir", default=None)
    parser.add_argument("--projection_error_npz", default=None,
                        help="Path to npz with keys: left_err3d_m, right_err3d_m, "
                             "left_err2d_px, right_err2d_px (per-frame floats).")
    parser.add_argument("--proj_3d_good", type=float, default=0.05,
                        help="3D error (m) below which the frame is trusted (weight 0).")
    parser.add_argument("--proj_3d_bad",  type=float, default=0.20,
                        help="3D error (m) above which Dyn-HaMR gets full weight (1).")
    parser.add_argument("--proj_px_good", type=float, default=30.0,
                        help="2D pixel error below which the frame is trusted (weight 0).")
    parser.add_argument("--proj_px_bad",  type=float, default=150.0,
                        help="2D pixel error above which Dyn-HaMR gets full weight (1).")

    # Ablation master switch: remove EVERY Dyn-HaMR contribution in one flag.
    # Forces w_dyn=0 (no 3D wrist guidance), disables the diffusion-time wrist
    # guidance, and disables Dyn-HaMR finger-pose injection. The 2D MediaPipe
    # reprojection in smpl_opt is left untouched, so the delta vs the full run
    # isolates the causal value of Dyn-HaMR's 3D/hand contribution alone.
    parser.add_argument("--no_dyn_hamr", action="store_true",
                        help="Ablation: strip all Dyn-HaMR contributions "
                             "(w_dyn=0, no diffusion wrist guidance, no finger pose).")
    parser.add_argument("--run_tag", default="",
                        help="Optional tag inserted into every saved npz/video "
                             "filename (e.g. 'ablfull', 'ablnodyn') so outputs are "
                             "self-identifying even outside their directory.")

    parser.add_argument("--run_smpl_opt", action="store_true")
    parser.add_argument("--smpl_opt_iters", type=int, default=120)
    parser.add_argument("--smpl_opt_lr", type=float, default=0.01)
    parser.add_argument("--smpl_w_wrist", type=float, default=10.0)
    parser.add_argument("--run_smpl_reproj_opt", action="store_true")
    parser.add_argument("--smpl_w_reproj", type=float, default=25.0)
    parser.add_argument("--smpl_reproj_huber_delta", type=float, default=0.02)
    parser.add_argument("--smpl_w_pose_prior", type=float, default=1.0)
    parser.add_argument("--smpl_w_smooth", type=float, default=0.1)
    args, remaining = parser.parse_known_args()

    examples = []
    for item in args.example:
        seq_name, start_idx = item.rsplit(":", 1)
        examples.append((seq_name, int(start_idx)))

    return examples, remaining, args


# ---------------------------------------------------------------------------
# Hand guidance loading
# ---------------------------------------------------------------------------

def load_hand_guidance(hand_guidance_dir, seq_name, start_idx):
    if hand_guidance_dir is None:
        return None
    seq_key = f"{seq_name}st{start_idx}_uem80"
    path = os.path.join(hand_guidance_dir, f"{seq_key}_hand_guidance.npz")
    if not os.path.exists(path):
        print(f"[WARN] Hand guidance not found: {path}")
        return None
    print(f"[INFO] Loading hand guidance: {path}")
    return np.load(path, allow_pickle=True)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def rigid_align(A, B):
    mu_A = A.mean(0);  mu_B = B.mean(0)
    AA = A - mu_A;     BB = B - mu_B
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    var_A = (AA ** 2).sum()
    scale = np.sum(S) / var_A
    t = mu_B - scale * (R @ mu_A)
    return scale * (A @ R.T) + t, scale, R, t


def apply_similarity(points, scale, R, t):
    return scale * (points @ R.T) + t


def aria_repre_to_world_T(aria_traj_repre):
    aria_local_T = rot_trans_to_matrix(
        rc.rotation_6d_to_matrix(aria_traj_repre[:, :6]),
        aria_traj_repre[:, 6:9],
    )
    delta_aria_invtsfm_T = rot_trans_to_matrix(
        rc.rotation_6d_to_matrix(aria_traj_repre[:, 9:15]),
        aria_traj_repre[:, 15:18],
    )
    aria_invtsfm_T = [delta_aria_invtsfm_T[0]]
    for i in range(1, aria_traj_repre.shape[0]):
        aria_invtsfm_T.append(aria_invtsfm_T[-1] @ delta_aria_invtsfm_T[i])
    aria_invtsfm_T = torch.stack(aria_invtsfm_T, dim=0)
    return aria_invtsfm_T @ aria_local_T


def rotation_to_make_forward_x_device(rotmat):
    forward = rotmat[:, :, 0].clone()
    forward[:, 2] = 0.0
    forward = forward / (torch.norm(forward, dim=-1, keepdim=True) + 1e-8)
    theta = torch.atan2(forward[:, 1], forward[:, 0])
    cos_t = torch.cos(theta);  sin_t = torch.sin(theta)
    rot = torch.zeros((rotmat.shape[0], 3, 3), dtype=rotmat.dtype, device=rotmat.device)
    rot[:, 0, 0] = cos_t;  rot[:, 0, 1] = sin_t
    rot[:, 1, 0] = -sin_t; rot[:, 1, 1] = cos_t
    rot[:, 2, 2] = 1.0
    return rot


def wrist_world_to_v4_local(wrist_world, aria_traj_T):
    tsfm_rot = rotation_to_make_forward_x_device(aria_traj_T[:, :3, :3])
    tsfm_trans = aria_traj_T[:, :3, 3] * torch.tensor(
        [-1.0, -1.0, 0.0], dtype=aria_traj_T.dtype, device=aria_traj_T.device)
    tsfm_trans = (tsfm_rot @ tsfm_trans[..., None])[..., 0]
    tsfm_T = rot_trans_to_matrix(tsfm_rot, tsfm_trans)
    wrist_h = torch.cat([wrist_world, torch.ones_like(wrist_world[:, :1])], dim=1)
    return (tsfm_T @ wrist_h[..., None])[:, :3, 0]


# ---------------------------------------------------------------------------
# CAMSPACE: new alignment function
# ---------------------------------------------------------------------------

def traj_align_dyn_to_uem(w2c, aria_traj_T):
    """
    Fit a similarity transform from Dyn-HaMR SLAM world → UEM body-world by
    aligning camera trajectory positions — no model call needed.

    w2c         : (T, 4, 4) numpy — SLAM world-to-camera (from hand_guidance *_w2c)
    aria_traj_T : (T, 4, 4) torch — UEM camera-to-world (c2w) from conditioning

    SLAM camera centre in world: C = -R_w2c^T @ t_w2c
    Aria camera centre in world: aria_traj_T[:, :3, 3]
    Returns (scale, R, t, n_pts) or None if too few frames.
    """
    R_w2c = w2c[:, :3, :3].astype(np.float64)
    t_w2c = w2c[:, :3, 3].astype(np.float64)
    slam_cam_pos = np.einsum("tji,tj->ti", R_w2c, -t_w2c)  # camera centre in SLAM world

    aria_cam_pos = aria_traj_T[:, :3, 3].detach().cpu().numpy().astype(np.float64)

    n = min(len(slam_cam_pos), len(aria_cam_pos))
    if n < 4:
        return None

    _, scale, R, t = rigid_align(slam_cam_pos[:n], aria_cam_pos[:n])
    return scale, R, t, n


def aria_traj_T_from_conditioning(traj_norm, ds, device):
    """Compute (T, 4, 4) c2w tensor from the conditioning trajectory — no model pass needed."""
    traj = ds.denormalize(traj_norm.detach().cpu(), "traj").to(device)
    return aria_repre_to_world_T(traj)


# ---------------------------------------------------------------------------
# Shared: similarity-transform fallback (kept for old hand_guidance files)
# ---------------------------------------------------------------------------

def fit_shared_hand_alignment(dyn_left, dyn_right, target_left, target_right,
                               left_valid, right_valid):
    dyn_pts, tgt_pts = [], []
    if left_valid.sum() > 0:
        dyn_pts.append(dyn_left[left_valid]);  tgt_pts.append(target_left[left_valid])
    if right_valid.sum() > 0:
        dyn_pts.append(dyn_right[right_valid]); tgt_pts.append(target_right[right_valid])
    if not dyn_pts:
        return None
    dyn_pts = np.concatenate(dyn_pts, axis=0)
    tgt_pts = np.concatenate(tgt_pts, axis=0)
    if len(dyn_pts) <= 3:
        return None
    _, scale, R, t = rigid_align(dyn_pts, tgt_pts)
    return scale, R, t, len(dyn_pts)


# ---------------------------------------------------------------------------
# Diffusion-time wrist guidance (unchanged from run_with_guidance.py)
# ---------------------------------------------------------------------------

def make_diffusion_wrist_denoised_fn(traj_norm, ds, left_wrist_world, right_wrist_world,
                                      left_valid, right_valid, strength=0.05,
                                      start_frac=0.7, num_diffusion_steps=1000, device="cuda"):
    if ds.repre_type != "v4_beta":
        raise ValueError(f"Diffusion wrist guidance requires v4_beta, got {ds.repre_type}")

    strength = float(np.clip(strength, 0.0, 1.0))
    start_frac = float(np.clip(start_frac, 0.0, 1.0))
    start_step = int(round(start_frac * num_diffusion_steps))
    active_steps = max(1, num_diffusion_steps - start_step)
    call_state = {"idx": 0}

    traj = ds.denormalize(traj_norm.detach().cpu(), "traj").to(device)
    n = min(len(traj), len(left_wrist_world), len(right_wrist_world),
            len(left_valid), len(right_valid))
    traj = traj[:n]
    aria_traj_T = aria_repre_to_world_T(traj)

    left_world  = torch.tensor(left_wrist_world[:n],  dtype=torch.float32, device=device)
    right_world = torch.tensor(right_wrist_world[:n], dtype=torch.float32, device=device)
    left_local  = wrist_world_to_v4_local(left_world,  aria_traj_T)
    right_local = wrist_world_to_v4_local(right_world, aria_traj_T)

    motion_mean = ds.stats["motion_mean"].to(device)
    motion_std  = ds.stats["motion_std"].to(device) + 1e-6

    left_channels  = torch.tensor([20 * 9 + 6, 20 * 9 + 7, 20 * 9 + 8], device=device)
    right_channels = torch.tensor([21 * 9 + 6, 21 * 9 + 7, 21 * 9 + 8], device=device)
    left_target  = (left_local  - motion_mean[left_channels])  / motion_std[left_channels]
    right_target = (right_local - motion_mean[right_channels]) / motion_std[right_channels]
    left_mask  = torch.tensor(left_valid[:n],  dtype=torch.bool, device=device)
    right_mask = torch.tensor(right_valid[:n], dtype=torch.bool, device=device)

    def denoised_fn(x_start):
        step_idx = call_state["idx"];  call_state["idx"] += 1
        if step_idx < start_step:
            return x_start
        ramp = min(1.0, float(step_idx - start_step + 1) / active_steps)
        step_strength = strength * ramp
        guided = x_start.clone()
        if left_mask.any():
            cur = guided[:, :n, left_channels]
            cur[:, left_mask] = ((1.0 - step_strength) * cur[:, left_mask]
                                 + step_strength * left_target[left_mask][None])
            guided[:, :n, left_channels] = cur
        if right_mask.any():
            cur = guided[:, :n, right_channels]
            cur[:, right_mask] = ((1.0 - step_strength) * cur[:, right_mask]
                                  + step_strength * right_target[right_mask][None])
            guided[:, :n, right_channels] = cur
        return guided

    return denoised_fn


# ---------------------------------------------------------------------------
# Post-inference optimisers (unchanged from run_with_guidance.py)
# ---------------------------------------------------------------------------

def dyn_position_weight_from_scale(scale, w_dyn, min_scale=0.5, max_scale=2.0,
                                    bad_align_w_dyn=0.0):
    if min_scale <= scale <= max_scale:
        return w_dyn, True
    return bad_align_w_dyn, False


def linear_reliability_score(ratio, good_ratio, bad_ratio):
    if not np.isfinite(ratio):
        return 0.0
    if bad_ratio <= good_ratio:
        return float(ratio <= good_ratio)
    return float(np.clip((bad_ratio - ratio) / (bad_ratio - good_ratio), 0.0, 1.0))


def derivative_consistency_ratio(dyn_left, dyn_right, pred_left, pred_right,
                                  left_valid, right_valid, order=1):
    def diff_order(x):
        return x[1:] - x[:-1] if order == 1 else x[2:] - 2 * x[1:-1] + x[:-2]
    def diff_mask(m):
        return m[1:] & m[:-1] if order == 1 else m[2:] & m[1:-1] & m[:-2]
    ratios = []
    for dyn, pred, mask in [(dyn_left, pred_left, left_valid),
                             (dyn_right, pred_right, right_valid)]:
        if len(dyn) <= order:
            continue
        valid = diff_mask(mask)
        if valid.sum() == 0:
            continue
        dyn_d  = diff_order(dyn)[valid]
        pred_d = diff_order(pred)[valid]
        mismatch = np.linalg.norm(dyn_d - pred_d, axis=-1).mean()
        pred_mag = np.linalg.norm(pred_d, axis=-1).mean()
        ratios.append(mismatch / (pred_mag + 1e-6))
    return float(np.mean(ratios)) if ratios else np.inf


def separation_consistency_ratio(dyn_left, dyn_right, pred_left, pred_right,
                                  left_valid, right_valid):
    both = left_valid & right_valid
    if both.sum() == 0:
        return np.inf
    dyn_sep  = np.linalg.norm(dyn_left[both]  - dyn_right[both],  axis=-1)
    pred_sep = np.linalg.norm(pred_left[both] - pred_right[both], axis=-1)
    return float(np.mean(np.abs(dyn_sep - pred_sep)) / (np.mean(pred_sep) + 1e-6))


def valid_smooth(values, valid, window=7):
    values = np.asarray(values, dtype=np.float32)
    valid  = np.asarray(valid,  dtype=bool) & np.isfinite(values)
    if not len(values):
        return values
    window = max(1, int(window))
    if window == 1:
        out = values.copy();  out[~valid] = np.inf;  return out
    kernel = np.ones(window, dtype=np.float32)
    num = np.convolve(np.where(valid, values, 0.0), kernel, mode="same")
    den = np.convolve(valid.astype(np.float32),     kernel, mode="same")
    out = np.full_like(values, np.inf)
    keep = den > 0;  out[keep] = num[keep] / den[keep]
    return out


def framewise_derivative_consistency(dyn_left, dyn_right, pred_left, pred_right,
                                      left_valid, right_valid, order=1, window=7):
    def diff_order(x):
        return x[1:] - x[:-1] if order == 1 else x[2:] - 2 * x[1:-1] + x[:-2]
    def diff_mask(m):
        return m[1:] & m[:-1] if order == 1 else m[2:] & m[1:-1] & m[:-2]
    out_len = max(0, len(pred_left) - order)
    ratios, valids = [], []
    for dyn, pred, mask in [(dyn_left, pred_left, left_valid),
                             (dyn_right, pred_right, right_valid)]:
        if len(dyn) <= order:
            continue
        valid = diff_mask(mask)
        dyn_d  = diff_order(dyn);  pred_d = diff_order(pred)
        ratio  = np.linalg.norm(dyn_d - pred_d, axis=-1) / (np.linalg.norm(pred_d, axis=-1) + 1e-6)
        ratios.append(ratio);  valids.append(valid)
    if not ratios:
        return np.full(out_len, np.inf, dtype=np.float32), np.zeros(out_len, dtype=bool)
    ratio_stack = np.stack(ratios);  valid_stack = np.stack(valids)
    ratio_sum   = np.where(valid_stack, ratio_stack, 0.0).sum(axis=0)
    valid_count = valid_stack.sum(axis=0)
    combined    = np.full(out_len, np.inf, dtype=np.float32)
    ok          = valid_count > 0;  combined[ok] = ratio_sum[ok] / valid_count[ok]
    return valid_smooth(combined, ok, window), ok


def framewise_separation_consistency(dyn_left, dyn_right, pred_left, pred_right,
                                      left_valid, right_valid, window=7):
    both     = left_valid & right_valid
    dyn_sep  = np.linalg.norm(dyn_left  - dyn_right,  axis=-1)
    pred_sep = np.linalg.norm(pred_left - pred_right, axis=-1)
    ratio    = np.abs(dyn_sep - pred_sep) / (pred_sep + 1e-6)
    return valid_smooth(ratio, both, window), both


def compute_framewise_dyn_guidance_reliability(dyn_left, dyn_right, pred_left, pred_right,
                                                left_valid, right_valid,
                                                vel_good_ratio=1.25, vel_bad_ratio=2.0,
                                                acc_good_ratio=2.0, acc_bad_ratio=5.0,
                                                sep_good_ratio=0.22, sep_bad_ratio=0.35,
                                                window=7):
    vel_ratio, vel_valid = framewise_derivative_consistency(
        dyn_left, dyn_right, pred_left, pred_right, left_valid, right_valid, order=1, window=window)
    acc_ratio, acc_valid = framewise_derivative_consistency(
        dyn_left, dyn_right, pred_left, pred_right, left_valid, right_valid, order=2, window=window)
    sep_ratio, sep_valid = framewise_separation_consistency(
        dyn_left, dyn_right, pred_left, pred_right, left_valid, right_valid, window=window)

    def score_arr(arr, good, bad):
        return np.array([linear_reliability_score(v, good, bad) for v in arr], dtype=np.float32)

    vel_w = score_arr(vel_ratio, vel_good_ratio, vel_bad_ratio);  vel_w[~vel_valid] = 0.0
    acc_w = score_arr(acc_ratio, acc_good_ratio, acc_bad_ratio);  acc_w[~acc_valid] = 0.0
    sep_w = score_arr(sep_ratio, sep_good_ratio, sep_bad_ratio);  sep_w[~sep_valid] = 0.0

    def vm(vals, valid):
        return float(np.asarray(vals)[valid].mean()) if valid.sum() > 0 else np.inf

    return {
        "dyn_vel_frame_ratios": vel_ratio, "dyn_acc_frame_ratios": acc_ratio,
        "dyn_sep_frame_ratios": sep_ratio,
        "dyn_vel_frame_weights": vel_w, "dyn_acc_frame_weights": acc_w,
        "dyn_sep_frame_weights": sep_w,
        "dyn_vel_consistency_ratio": vm(vel_ratio, vel_valid),
        "dyn_acc_consistency_ratio": vm(acc_ratio, acc_valid),
        "dyn_sep_consistency_ratio": vm(sep_ratio, sep_valid),
        "dyn_vel_reliability": float(vel_w[vel_valid].mean()) if vel_valid.sum() > 0 else 0.0,
        "dyn_acc_reliability": float(acc_w[acc_valid].mean()) if acc_valid.sum() > 0 else 0.0,
        "dyn_sep_reliability": float(sep_w[sep_valid].mean()) if sep_valid.sum() > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def masked_mse(a, b, mask):
    if mask.sum() == 0:
        return torch.tensor(0.0, device=a.device)
    return ((a[mask] - b[mask]) ** 2).mean()


def weighted_masked_mse(a, b, mask, weights=None):
    if weights is None:
        return masked_mse(a, b, mask)
    weights = weights.to(device=a.device, dtype=a.dtype)
    valid   = mask & (weights > 0)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=a.device)
    per_item = ((a - b) ** 2).mean(dim=-1)
    w = weights[valid]
    return (per_item[valid] * w).sum() / w.sum().clamp_min(1e-6)


def mean_l2(a, b, mask=None):
    if mask is None:
        return float(np.linalg.norm(a - b, axis=-1).mean())
    if mask.sum() == 0:
        return np.nan
    return float(np.linalg.norm(a[mask] - b[mask], axis=-1).mean())


def masked_sequence_l2(a, b, mask):
    if not len(a) or mask.sum() == 0:
        return np.nan
    return mean_l2(a, b, mask)


def velocity_error(a, b, mask):
    if len(a) < 2:
        return np.nan
    vm = mask[1:] & mask[:-1]
    return masked_sequence_l2(a[1:] - a[:-1], b[1:] - b[:-1], vm)


def acceleration_error(a, b, mask):
    if len(a) < 3:
        return np.nan
    am  = mask[2:] & mask[1:-1] & mask[:-2]
    acc_a = a[2:] - 2 * a[1:-1] + a[:-2]
    acc_b = b[2:] - 2 * b[1:-1] + b[:-2]
    return masked_sequence_l2(acc_a, acc_b, am)


def rotation_geodesic_deg(rot_a, rot_b):
    rel   = np.matmul(rot_a, np.swapaxes(rot_b, -1, -2))
    trace = np.trace(rel, axis1=-2, axis2=-1)
    cos   = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos)).mean())


def compute_wrist_metrics(name, left_wrist, right_wrist, gt_left, gt_right, lv, rv):
    both = lv & rv
    return {
        f"{name}_left_wrist_mpjpe":   mean_l2(left_wrist,  gt_left,  lv),
        f"{name}_right_wrist_mpjpe":  mean_l2(right_wrist, gt_right, rv),
        f"{name}_left_wrist_vel_err":  velocity_error(left_wrist,  gt_left,  lv),
        f"{name}_right_wrist_vel_err": velocity_error(right_wrist, gt_right, rv),
        f"{name}_left_wrist_acc_err":  acceleration_error(left_wrist,  gt_left,  lv),
        f"{name}_right_wrist_acc_err": acceleration_error(right_wrist, gt_right, rv),
        f"{name}_hand_sep_err": mean_l2(
            np.linalg.norm(left_wrist  - right_wrist,  axis=-1)[:, None],
            np.linalg.norm(gt_left - gt_right, axis=-1)[:, None], both),
    }


def _procrustes_align_traj(pred, gt, mask):
    """Per-sequence rigid (scale+R+t) Procrustes of `pred` onto `gt` over valid
    frames, applied to the whole sequence. Removes any global similarity offset
    so the metric reflects shape/articulation of the trajectory, not placement.
    Falls back to identity when too few valid frames."""
    if mask.sum() < 3:
        return pred.copy()
    A = pred[mask]
    B = gt[mask]
    mu_A = A.mean(0); mu_B = B.mean(0)
    AA = A - mu_A;    BB = B - mu_B
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    var_A = (AA ** 2).sum()
    scale = float(np.sum(S) / var_A) if var_A > 1e-12 else 1.0
    t = mu_B - scale * (R @ mu_A)
    return scale * (pred @ R.T) + t


def compute_wrist_metrics_decomposed(name, left_wrist, right_wrist,
                                     gt_left, gt_right, lv, rv,
                                     pred_root=None, gt_root=None):
    """Wrist error decomposed to isolate articulation from body placement.

    Reports three variants per side:
      *_wrist_mpjpe         : world-frame (same as compute_wrist_metrics) — includes
                              whole-body root/trajectory error.
      *_wrist_mpjpe_rootrel : after subtracting each body's own pelvis root
                              (pred_root from prediction, gt_root from GT). Removes
                              translational placement; keeps global orientation error.
                              Only emitted when roots are provided.
      *_wrist_mpjpe_proc    : after a per-sequence similarity (scale+R+t) Procrustes
                              of the predicted wrist track onto GT. Removes placement
                              AND global rotation/scale — pure trajectory-shape error.

    This is the metric that matches what the eye does when matching the rendered
    ball to the GT mesh's hand: a local comparison, not a world-frame one.
    """
    out = {}
    # World-frame (baseline, identical to the legacy metric).
    out[f"{name}_left_wrist_mpjpe"]  = mean_l2(left_wrist,  gt_left,  lv)
    out[f"{name}_right_wrist_mpjpe"] = mean_l2(right_wrist, gt_right, rv)

    # Root-relative (subtract each body's own pelvis).
    if pred_root is not None and gt_root is not None:
        out[f"{name}_left_wrist_mpjpe_rootrel"]  = mean_l2(
            left_wrist - pred_root,  gt_left  - gt_root, lv)
        out[f"{name}_right_wrist_mpjpe_rootrel"] = mean_l2(
            right_wrist - pred_root, gt_right - gt_root, rv)

    # Procrustes-aligned (strip placement + global rotation/scale).
    lw_p = _procrustes_align_traj(left_wrist,  gt_left,  lv)
    rw_p = _procrustes_align_traj(right_wrist, gt_right, rv)
    out[f"{name}_left_wrist_mpjpe_proc"]  = mean_l2(lw_p, gt_left,  lv)
    out[f"{name}_right_wrist_mpjpe_proc"] = mean_l2(rw_p, gt_right, rv)
    return out


def compute_motion_metrics(name, joints, gt_joints, lv, rv):
    body_idx       = np.arange(22)
    left_arm_idx   = np.array([16, 18, 20])
    right_arm_idx  = np.array([17, 19, 21])
    metrics = {
        f"{name}_body_mpjpe":       mean_l2(joints[:, body_idx],      gt_joints[:, body_idx]),
        f"{name}_left_arm_mpjpe":   mean_l2(joints[:, left_arm_idx],  gt_joints[:, left_arm_idx]),
        f"{name}_right_arm_mpjpe":  mean_l2(joints[:, right_arm_idx], gt_joints[:, right_arm_idx]),
    }
    metrics.update(compute_wrist_metrics(name,
        joints[:, 20], joints[:, 21], gt_joints[:, 20], gt_joints[:, 21], lv, rv))
    return metrics


def print_eval_table(title, metrics, names, rows):
    print(title)
    for key, label in rows:
        row = [f"{label:20s}"]
        for name in names:
            value = metrics.get(f"{name}_{key}", np.nan)
            row.append(f"{name}: {value:.4f}")
        print("  " + " | ".join(row))


def print_comprehensive_eval(metrics, full_body_names, wrist_names):
    print_eval_table("[EVAL] Full-body / arm metrics (meters)", metrics, full_body_names, [
        ("body_mpjpe", "Body MPJPE"),
        ("left_arm_mpjpe", "Left arm MPJPE"),
        ("right_arm_mpjpe", "Right arm MPJPE"),
    ])
    print_eval_table("[EVAL] Wrist guidance metrics (meters)", metrics, wrist_names, [
        ("left_wrist_mpjpe",   "Left wrist MPJPE"),
        ("right_wrist_mpjpe",  "Right wrist MPJPE"),
        ("left_wrist_vel_err", "Left wrist velocity"),
        ("right_wrist_vel_err","Right wrist velocity"),
        ("left_wrist_acc_err", "Left wrist accel"),
        ("right_wrist_acc_err","Right wrist accel"),
        ("hand_sep_err",       "Hand separation"),
    ])
    # Placement-decomposed wrist error: world vs root-relative vs Procrustes.
    # rootrel strips body translation; proc additionally strips global rot+scale.
    print_eval_table("[EVAL] Wrist MPJPE decomposed (meters)", metrics, wrist_names, [
        ("left_wrist_mpjpe",         "L world"),
        ("left_wrist_mpjpe_rootrel", "L root-rel"),
        ("left_wrist_mpjpe_proc",    "L procrustes"),
        ("right_wrist_mpjpe",        "R world"),
        ("right_wrist_mpjpe_rootrel","R root-rel"),
        ("right_wrist_mpjpe_proc",   "R procrustes"),
    ])


def project_points_numpy(points_world, cam_R, cam_t, intrins, eps=1e-4):
    pts_cam = np.einsum("tij,tj->ti", cam_R, points_world) + cam_t
    z = np.maximum(pts_cam[:, 2], eps)
    fx, fy, cx, cy = intrins.reshape(-1)[:4]
    pts_2d = np.stack([fx * pts_cam[:, 0] / z + cx, fy * pts_cam[:, 1] / z + cy], axis=-1)
    return pts_2d, pts_cam[:, 2]


# SMPL-X parent-chain body_pose indices to each wrist (verified via smpl.parents).
# body_pose[i] is the local rotation of joint (i+1) wrt its parent. Multiply in
# order: global_orient (pelvis) → spine1 → spine2 → spine3 → collar → shoulder
# → elbow → wrist.  Result = global rotation of the wrist joint.
_LWRIST_CHAIN = [2, 5, 8, 12, 15, 17, 19]   # L_collar=13→[12], L_shoulder=16→[15], etc.
_RWRIST_CHAIN = [2, 5, 8, 13, 16, 18, 20]


def smplx_global_wrist_rot(global_orient, body_pose, chain):
    R = global_orient
    for i in chain:
        R = R @ body_pose[:, i]
    return R   # (T, 3, 3)


def project_points_torch(points_world, cam_R, cam_t, intrins, eps=1e-4):
    pts_cam = torch.einsum("tij,tj->ti", cam_R, points_world) + cam_t
    z = pts_cam[:, 2].clamp_min(eps)
    fx, fy, cx, cy = intrins
    pts_2d = torch.stack([fx * pts_cam[:, 0] / z + cx, fy * pts_cam[:, 1] / z + cy], dim=-1)
    return pts_2d, pts_cam[:, 2]


def normalized_reprojection_loss(pred_2d, target_2d, mask, focal, huber_delta=0.02, weights=None):
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_2d.device)
    err      = (pred_2d[mask] - target_2d[mask]) / focal.clamp_min(1e-6)
    err_norm = torch.sqrt((err ** 2).sum(dim=-1) + 1e-8)
    delta    = torch.tensor(huber_delta, dtype=err_norm.dtype, device=err_norm.device)
    huber    = torch.where(err_norm < delta, 0.5 * err_norm ** 2 / delta, err_norm - 0.5 * delta)
    if weights is None:
        return huber.mean()
    w = weights[mask].to(huber.dtype)
    return (huber * w).sum() / w.sum().clamp_min(1e-6)


def compute_reprojection_metrics(name, left_rel, right_rel, dyn_left_raw, dyn_right_raw,
                                  left_valid, right_valid, shared_scale, shared_R, shared_t,
                                  left_cam_R, right_cam_R, left_cam_t, right_cam_t, intrins):
    if (left_cam_R is None or right_cam_R is None or left_cam_t is None
            or right_cam_t is None or intrins is None or abs(float(shared_scale)) <= 1e-8):
        return {f"{name}_left_reproj_px": np.nan, f"{name}_right_reproj_px": np.nan}
    pred_left_dyn  = ((left_rel  - shared_t) @ shared_R) / shared_scale
    pred_right_dyn = ((right_rel - shared_t) @ shared_R) / shared_scale
    pl2, pld = project_points_numpy(pred_left_dyn,  left_cam_R,  left_cam_t,  intrins)
    pr2, prd = project_points_numpy(pred_right_dyn, right_cam_R, right_cam_t, intrins)
    tl2, tld = project_points_numpy(dyn_left_raw,  left_cam_R,  left_cam_t,  intrins)
    tr2, trd = project_points_numpy(dyn_right_raw, right_cam_R, right_cam_t, intrins)
    lm = left_valid  & (pld > 1e-4) & (tld > 1e-4)
    rm = right_valid & (prd > 1e-4) & (trd > 1e-4)
    return {f"{name}_left_reproj_px": mean_l2(pl2, tl2, lm),
            f"{name}_right_reproj_px": mean_l2(pr2, tr2, rm)}


def evaluate_smpl_grad(smpl, smpl_params):
    out = smpl(**smpl_params, return_full_pose=True)
    return out.joints, out.vertices, out.full_pose


def clone_smpl_params(smpl_params, device):
    return {k: v.detach().clone().to(device) for k, v in smpl_params.items()}


def optimize_wrist_guidance(pred_left, pred_right, dyn_left, dyn_right,
                              left_valid, right_valid, num_iters=300, lr=0.03,
                              w_prior=1.0, w_dyn=1.0, w_vel=0.5, w_acc=0.0,
                              w_sep=0.2, w_smooth=0.05,
                              pos_weights_l=None, pos_weights_r=None,
                              vel_weights=None, acc_weights=None, sep_weights=None,
                              device="cuda"):
    pred_left  = torch.tensor(pred_left,  dtype=torch.float32, device=device)
    pred_right = torch.tensor(pred_right, dtype=torch.float32, device=device)
    dyn_left   = torch.tensor(dyn_left,   dtype=torch.float32, device=device)
    dyn_right  = torch.tensor(dyn_right,  dtype=torch.float32, device=device)
    lv = torch.tensor(left_valid,  dtype=torch.bool, device=device)
    rv = torch.tensor(right_valid, dtype=torch.bool, device=device)
    both = lv & rv
    if vel_weights is not None: vel_weights = torch.tensor(vel_weights, dtype=torch.float32, device=device)
    if acc_weights is not None: acc_weights = torch.tensor(acc_weights, dtype=torch.float32, device=device)
    if sep_weights is not None: sep_weights = torch.tensor(sep_weights, dtype=torch.float32, device=device)
    if pos_weights_l is not None: pos_weights_l = torch.tensor(pos_weights_l, dtype=torch.float32, device=device)
    if pos_weights_r is not None: pos_weights_r = torch.tensor(pos_weights_r, dtype=torch.float32, device=device)

    opt_l = pred_left.clone().detach().requires_grad_(True)
    opt_r = pred_right.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([opt_l, opt_r], lr=lr)

    for _ in range(num_iters):
        optimizer.zero_grad()
        loss_prior = ((opt_l - pred_left)**2).mean() + ((opt_r - pred_right)**2).mean()
        # Projection-error-weighted dyn loss: per-frame trust scales the L2.
        loss_dyn   = (weighted_masked_mse(opt_l, dyn_left,  lv, pos_weights_l)
                    + weighted_masked_mse(opt_r, dyn_right, rv, pos_weights_r))

        ol_vel = opt_l[1:] - opt_l[:-1];  or_vel = opt_r[1:] - opt_r[:-1]
        dl_vel = dyn_left[1:] - dyn_left[:-1]; dr_vel = dyn_right[1:] - dyn_right[:-1]
        lv_vel = lv[1:] & lv[:-1];  rv_vel = rv[1:] & rv[:-1]
        loss_vel = (weighted_masked_mse(ol_vel, dl_vel, lv_vel, vel_weights)
                  + weighted_masked_mse(or_vel, dr_vel, rv_vel, vel_weights))

        ol_acc = opt_l[2:] - 2*opt_l[1:-1] + opt_l[:-2]
        or_acc = opt_r[2:] - 2*opt_r[1:-1] + opt_r[:-2]
        dl_acc = dyn_left[2:] - 2*dyn_left[1:-1] + dyn_left[:-2]
        dr_acc = dyn_right[2:] - 2*dyn_right[1:-1] + dyn_right[:-2]
        lv_acc = lv[2:] & lv[1:-1] & lv[:-2];  rv_acc = rv[2:] & rv[1:-1] & rv[:-2]
        loss_acc = (weighted_masked_mse(ol_acc, dl_acc, lv_acc, acc_weights)
                  + weighted_masked_mse(or_acc, dr_acc, rv_acc, acc_weights))

        opt_sep = torch.norm(opt_l - opt_r, dim=-1)
        dyn_sep = torch.norm(dyn_left - dyn_right, dim=-1)
        if sep_weights is None:
            loss_sep = ((opt_sep[both] - dyn_sep[both])**2).mean() if both.sum() > 0 else torch.tensor(0.0, device=device)
        else:
            sv = both & (sep_weights > 0)
            loss_sep = (((opt_sep[sv] - dyn_sep[sv])**2) * sep_weights[sv]).sum() / sep_weights[sv].sum().clamp_min(1e-6) if sv.sum() > 0 else torch.tensor(0.0, device=device)

        loss_smooth = (ol_acc**2).mean() + (or_acc**2).mean()
        loss = (w_prior*loss_prior + w_dyn*loss_dyn + w_vel*loss_vel
               + w_acc*loss_acc + w_sep*loss_sep + w_smooth*loss_smooth)
        loss.backward();  optimizer.step()

    return opt_l.detach().cpu().numpy(), opt_r.detach().cpu().numpy()


def optimize_smpl_arm_guidance(smpl, pred_smpl_params, guided_left_wrist, guided_right_wrist,
                                left_valid, right_valid, num_iters=120, lr=0.01,
                                w_wrist=10.0, w_reproj=0.0, reproj_huber_delta=0.02,
                                dyn_left_raw=None, dyn_right_raw=None,
                                left_cam_R=None, right_cam_R=None,
                                left_cam_t=None, right_cam_t=None, intrins=None,
                                shared_scale=1.0, shared_R=None, shared_t=None,
                                w_pose_prior=1.0, w_smooth=0.1, device="cuda"):
    ROOT_IDX = 0; LEFT_WRIST_IDX = 20; RIGHT_WRIST_IDX = 21
    arm_body_pose_indices = torch.tensor([12,13,15,16,17,18,19,20], device=device)

    opt_params = clone_smpl_params(pred_smpl_params, device)
    smpl_orig_device = next(smpl.buffers()).device
    smpl = smpl.to(device)
    body_pose_base = opt_params["body_pose"].detach()
    body_pose_6d_base = rc.matrix_to_rotation_6d(body_pose_base)
    arm_pose_6d = body_pose_6d_base[:, arm_body_pose_indices].clone().detach().requires_grad_(True)

    guided_left_wrist  = torch.tensor(guided_left_wrist,  dtype=torch.float32, device=device)
    guided_right_wrist = torch.tensor(guided_right_wrist, dtype=torch.float32, device=device)
    lv = torch.tensor(left_valid,  dtype=torch.bool, device=device)
    rv = torch.tensor(right_valid, dtype=torch.bool, device=device)

    reproj_enabled = (
        w_reproj > 0.0 and dyn_left_raw is not None and dyn_right_raw is not None
        and left_cam_R is not None and right_cam_R is not None
        and left_cam_t is not None and right_cam_t is not None and intrins is not None
        and shared_R is not None and shared_t is not None
        and np.isfinite(shared_scale) and abs(float(shared_scale)) > 1e-8
    )
    if reproj_enabled:
        dyn_left_raw   = torch.tensor(dyn_left_raw,   dtype=torch.float32, device=device)
        dyn_right_raw  = torch.tensor(dyn_right_raw,  dtype=torch.float32, device=device)
        left_cam_R     = torch.tensor(left_cam_R,     dtype=torch.float32, device=device)
        right_cam_R    = torch.tensor(right_cam_R,    dtype=torch.float32, device=device)
        left_cam_t     = torch.tensor(left_cam_t,     dtype=torch.float32, device=device)
        right_cam_t    = torch.tensor(right_cam_t,    dtype=torch.float32, device=device)
        intrins        = torch.tensor(intrins,        dtype=torch.float32, device=device).reshape(-1)[:4]
        shared_R       = torch.tensor(shared_R,       dtype=torch.float32, device=device)
        shared_t       = torch.tensor(shared_t,       dtype=torch.float32, device=device)
        shared_scale_t = torch.tensor(float(shared_scale), dtype=torch.float32, device=device)
        focal          = torch.mean(intrins[:2])
        tl2, tld = project_points_torch(dyn_left_raw,  left_cam_R,  left_cam_t,  intrins)
        tr2, trd = project_points_torch(dyn_right_raw, right_cam_R, right_cam_t, intrins)

    optimizer = torch.optim.Adam([arm_pose_6d], lr=lr)
    for _ in range(num_iters):
        optimizer.zero_grad()
        bp6d = body_pose_6d_base.clone()
        bp6d[:, arm_body_pose_indices] = arm_pose_6d
        opt_params["body_pose"] = rc.rotation_6d_to_matrix(bp6d)
        joints, _, _ = evaluate_smpl_grad(smpl, opt_params)
        n_guided = len(guided_left_wrist)
        loss_wrist  = masked_mse(joints[:n_guided, LEFT_WRIST_IDX],  guided_left_wrist,  lv)
        loss_wrist += masked_mse(joints[:n_guided, RIGHT_WRIST_IDX], guided_right_wrist, rv)

        loss_reproj = torch.tensor(0.0, device=device)
        if reproj_enabled:
            root    = joints[:len(lv), ROOT_IDX]
            lr_rel  = joints[:len(lv), LEFT_WRIST_IDX]  - root
            rr_rel  = joints[:len(rv), RIGHT_WRIST_IDX] - root
            pld_dyn = ((lr_rel - shared_t) @ shared_R) / shared_scale_t
            prd_dyn = ((rr_rel - shared_t) @ shared_R) / shared_scale_t
            pl2, pld = project_points_torch(pld_dyn, left_cam_R,  left_cam_t,  intrins)
            pr2, prd = project_points_torch(prd_dyn, right_cam_R, right_cam_t, intrins)
            lrv = lv & (tld > 1e-4) & (pld > 1e-4)
            rrv = rv & (trd > 1e-4) & (prd > 1e-4)
            loss_reproj  = normalized_reprojection_loss(pl2, tl2, lrv, focal, reproj_huber_delta)
            loss_reproj += normalized_reprojection_loss(pr2, tr2, rrv, focal, reproj_huber_delta)

        loss_prior  = ((arm_pose_6d - body_pose_6d_base[:, arm_body_pose_indices])**2).mean()
        arm_acc     = arm_pose_6d[2:] - 2*arm_pose_6d[1:-1] + arm_pose_6d[:-2]
        loss_smooth = (arm_acc**2).mean()
        loss = w_wrist*loss_wrist + w_reproj*loss_reproj + w_pose_prior*loss_prior + w_smooth*loss_smooth
        loss.backward();  optimizer.step()

    with torch.no_grad():
        bp6d = body_pose_6d_base.clone()
        bp6d[:, arm_body_pose_indices] = arm_pose_6d
        opt_params["body_pose"] = rc.rotation_6d_to_matrix(bp6d)
        joints, verts, _ = evaluate_smpl_grad(smpl, opt_params)
    smpl.to(smpl_orig_device)
    return {k: v.detach().cpu() for k, v in opt_params.items()}, joints.detach().cpu(), verts.detach().cpu()


# ---------------------------------------------------------------------------
# 2D MediaPipe guidance helpers
# ---------------------------------------------------------------------------

def load_2d_guidance(guidance_2d_dir, seq_name, start_idx):
    if guidance_2d_dir is None:
        return None
    seq_key  = f"{seq_name}st{start_idx}_uem80"
    path = os.path.join(guidance_2d_dir, f"{seq_key}_2d_wrist_guidance.npz")
    if not os.path.exists(path):
        print(f"[2D] No 2D guidance file: {path}")
        return None
    print(f"[2D] Loading 2D wrist guidance: {path}")
    return np.load(path, allow_pickle=True)


# ---------------------------------------------------------------------------
# Projection-error-driven per-frame trust weighting (from run_with_guidance_a.py)
# ---------------------------------------------------------------------------

def error_to_guidance_weight(err, good, bad):
    """Linear ramp:  err <= good -> 0,  err >= bad -> 1, linear in between."""
    err = np.asarray(err, dtype=np.float32)
    w = (err - good) / max(bad - good, 1e-6)
    return np.clip(w, 0.0, 1.0)


def load_projection_error_guidance_weights(path, n, cli_args):
    """
    Read precomputed per-frame projection errors and convert to per-frame trust
    weights for the 3D wrist guidance loss. Frames where UEM is far off get
    upweighted so the optimizer pulls them harder toward Dyn-HaMR.

    Expects npz with keys:
      left_err3d_m  / right_err3d_m   (T,) — 3D error in meters
      left_err2d_px / right_err2d_px  (T,) — 2D pixel error
    """
    if path is None or not os.path.exists(path):
        return None
    x = np.load(path, allow_pickle=True)
    left_3d  = x["left_err3d_m"][:n]
    right_3d = x["right_err3d_m"][:n]
    left_px  = x["left_err2d_px"][:n]
    right_px = x["right_err2d_px"][:n]
    left_w3d  = error_to_guidance_weight(left_3d,  cli_args.proj_3d_good, cli_args.proj_3d_bad)
    right_w3d = error_to_guidance_weight(right_3d, cli_args.proj_3d_good, cli_args.proj_3d_bad)
    left_wpx  = error_to_guidance_weight(left_px,  cli_args.proj_px_good, cli_args.proj_px_bad)
    right_wpx = error_to_guidance_weight(right_px, cli_args.proj_px_good, cli_args.proj_px_bad)
    # Per-frame max of 3D-derived and 2D-derived weight.
    left_w  = np.maximum(left_w3d,  left_wpx ).astype(np.float32)
    right_w = np.maximum(right_w3d, right_wpx).astype(np.float32)
    return left_w, right_w


def override_hand_pose_with_dyn_hamr(pred_smpl_params, hand_guidance):
    """
    Replace SMPL-X's PCA-decoded hand pose with Dyn-HaMR's per-frame MANO finger
    articulation on frames where Dyn-HaMR is valid. UEM-X does not actually estimate
    fingers (PCA hand pose is essentially a smooth average), but Dyn-HaMR / HaMeR is
    a hand pose estimator -- its `pose_body` carries genuine finger information that
    UEM cannot synthesise.

    Both formats follow MANO's 15-finger-joint axis-angle convention, so the values
    transfer directly. Wrist position (joint 20/21) is unchanged -- only the 15
    finger joints downstream of the wrist are affected.
    """
    if hand_guidance is None or "left_pose_body" not in hand_guidance.files:
        return pred_smpl_params, 0, 0

    lp = hand_guidance["left_pose_body"].astype(np.float32)   # (T, 15, 3)
    rp = hand_guidance["right_pose_body"].astype(np.float32)  # (T, 15, 3)
    lv = hand_guidance["left_valid"].astype(bool)
    rv = hand_guidance["right_valid"].astype(bool)

    lhp = pred_smpl_params["left_hand_pose"]   # (T, 15, 3, 3) torch
    rhp = pred_smpl_params["right_hand_pose"]
    dev, dtype = lhp.device, lhp.dtype
    T = min(lhp.shape[0], len(lp))

    lp_t  = torch.tensor(lp[:T], dtype=dtype, device=dev)
    rp_t  = torch.tensor(rp[:T], dtype=dtype, device=dev)
    lv_t  = torch.tensor(lv[:T], device=dev)
    rv_t  = torch.tensor(rv[:T], device=dev)
    lp_mat = rc.axis_angle_to_matrix(lp_t)   # (T, 15, 3, 3)
    rp_mat = rc.axis_angle_to_matrix(rp_t)

    new_lhp = lhp.clone()
    new_rhp = rhp.clone()
    new_lhp[:T][lv_t] = lp_mat[lv_t]
    new_rhp[:T][rv_t] = rp_mat[rv_t]

    pred_smpl_params = {k: v for k, v in pred_smpl_params.items()}
    pred_smpl_params["left_hand_pose"]  = new_lhp
    pred_smpl_params["right_hand_pose"] = new_rhp
    return pred_smpl_params, int(lv_t.sum()), int(rv_t.sum())


def smooth_2d_detections(values, valid, sigma):
    """
    Gaussian-smooth (T, 2) MediaPipe detections along time, treating invalid
    frames as missing (linear-interpolated from valid neighbours so they don't
    inject zero/garbage into the kernel). Only frames originally marked valid
    are used downstream; this just denoises their target positions.
    """
    if sigma <= 0 or not valid.any() or valid.sum() < 2:
        return values.astype(np.float32, copy=True)
    from scipy.ndimage import gaussian_filter1d
    out = values.astype(np.float32, copy=True)
    valid_idx = np.where(valid)[0]
    all_idx = np.arange(len(values))
    for c in range(values.shape[1]):
        out[:, c] = np.interp(all_idx, valid_idx, values[valid_idx, c])
        out[:, c] = gaussian_filter1d(out[:, c], sigma=sigma)
    return out


def project_wrist_world_to_pixels(wrist_world, aria_traj_T, intrins):
    """
    Project 3D wrists in UEM body-world → pixel space using GT Aria c2w.

    wrist_world  : (T, 3) torch
    aria_traj_T  : (T, 4, 4) torch — camera-to-world
    intrins      : (4,) torch — [fx, fy, cx, cy]
    Returns (T, 2) pixel coords and (T,) positive-depth mask.
    """
    R_c2w = aria_traj_T[:, :3, :3]   # (T,3,3)
    t_c2w = aria_traj_T[:, :3, 3]    # (T,3)
    # world → camera: p_cam = R^T (p_world - t)
    pts_cam = torch.einsum("tji,tj->ti", R_c2w, wrist_world - t_c2w)
    z = pts_cam[:, 2]
    depth_ok = z > 0.05
    z_safe = z.clamp_min(0.05)
    fx, fy, cx, cy = intrins[0], intrins[1], intrins[2], intrins[3]
    # Aria canonical frame: camera y-axis points UP in world.
    # Both axes are negated relative to standard OpenCV pinhole.
    u = -fx * pts_cam[:, 0] / z_safe + cx
    v = -fy * pts_cam[:, 1] / z_safe + cy
    return torch.stack([u, v], dim=-1), depth_ok


def optimize_smpl_arm_2d(smpl, pred_smpl_params, guided_left_wrist, guided_right_wrist,
                          left_valid, right_valid,
                          mp_left_2d, mp_right_2d, left_valid_2d, right_valid_2d,
                          gt_aria_traj_T, intrins,
                          left_conf_2d=None, right_conf_2d=None,
                          dyn_left_rot_uem=None, dyn_right_rot_uem=None,
                          num_iters=120, lr=0.01,
                          w_wrist=10.0, w_reproj_2d=5.0,
                          w_pose_prior=1.0, w_smooth=0.1, w_smooth_pos=50.0,
                          w_wrist_orient=5.0,
                          reproj_huber_delta=8.0,
                          device="cuda"):
    """
    SMPL arm optimisation with 3D wrist loss + 2D MediaPipe reprojection loss.
    Replaces the Dyn-HaMR reprojection with clean Aria-camera reprojection.
    """
    ROOT_IDX = 0; LEFT_WRIST_IDX = 20; RIGHT_WRIST_IDX = 21
    arm_body_pose_indices = torch.tensor([12,13,15,16,17,18,19,20], device=device)

    # Move SMPL to device, remember original so we can restore (this module is
    # shared across clips — leaving it on CUDA breaks the next clip's CPU eval).
    smpl_orig_device = next(smpl.buffers()).device
    smpl = smpl.to(device)
    opt_params = clone_smpl_params(pred_smpl_params, device)
    body_pose_6d_base = rc.matrix_to_rotation_6d(opt_params["body_pose"].clone())
    arm_pose_6d = body_pose_6d_base[:, arm_body_pose_indices].clone().requires_grad_(True)

    lv = torch.tensor(left_valid,  dtype=torch.bool, device=device)
    rv = torch.tensor(right_valid, dtype=torch.bool, device=device)
    lv2 = torch.tensor(left_valid_2d,  dtype=torch.bool, device=device)
    rv2 = torch.tensor(right_valid_2d, dtype=torch.bool, device=device)

    guided_left_wrist_t  = torch.tensor(guided_left_wrist,  dtype=torch.float32, device=device)
    guided_right_wrist_t = torch.tensor(guided_right_wrist, dtype=torch.float32, device=device)
    mp_left_t  = torch.tensor(mp_left_2d,  dtype=torch.float32, device=device)
    mp_right_t = torch.tensor(mp_right_2d, dtype=torch.float32, device=device)
    lconf_t = (torch.tensor(left_conf_2d,  dtype=torch.float32, device=device)
               if left_conf_2d  is not None else None)
    rconf_t = (torch.tensor(right_conf_2d, dtype=torch.float32, device=device)
               if right_conf_2d is not None else None)
    lrot_t  = (torch.tensor(dyn_left_rot_uem,  dtype=torch.float32, device=device)
               if dyn_left_rot_uem  is not None else None)
    rrot_t  = (torch.tensor(dyn_right_rot_uem, dtype=torch.float32, device=device)
               if dyn_right_rot_uem is not None else None)
    use_wrist_orient = w_wrist_orient > 0 and lrot_t is not None and rrot_t is not None

    n_guided = len(guided_left_wrist)
    n_2d = len(left_valid_2d)
    n_traj = min(len(gt_aria_traj_T), n_2d)

    intrins_t = torch.tensor(intrins, dtype=torch.float32, device=device).reshape(-1)[:4]
    focal = intrins_t[:2].mean()
    aria_t = gt_aria_traj_T[:n_traj].to(device)

    optimizer = torch.optim.Adam([arm_pose_6d], lr=lr)
    for _ in range(num_iters):
        optimizer.zero_grad()
        bp6d = body_pose_6d_base.clone()
        bp6d[:, arm_body_pose_indices] = arm_pose_6d
        opt_params["body_pose"] = rc.rotation_6d_to_matrix(bp6d)

        joints, _, _ = evaluate_smpl_grad(smpl, opt_params)

        # 3D wrist loss
        loss_wrist  = masked_mse(joints[:n_guided, LEFT_WRIST_IDX],  guided_left_wrist_t,  lv)
        loss_wrist += masked_mse(joints[:n_guided, RIGHT_WRIST_IDX], guided_right_wrist_t, rv)

        # 2D reprojection loss using GT Aria camera
        loss_reproj_2d = torch.tensor(0.0, device=device)
        if w_reproj_2d > 0 and n_traj > 0:
            left_px,  ldepth_ok = project_wrist_world_to_pixels(
                joints[:n_traj, LEFT_WRIST_IDX],  aria_t, intrins_t)
            right_px, rdepth_ok = project_wrist_world_to_pixels(
                joints[:n_traj, RIGHT_WRIST_IDX], aria_t, intrins_t)

            left_mask_2d  = lv2[:n_traj] & ldepth_ok
            right_mask_2d = rv2[:n_traj] & rdepth_ok
            lw_w = lconf_t[:n_traj] if lconf_t is not None else None
            rw_w = rconf_t[:n_traj] if rconf_t is not None else None
            loss_reproj_2d  = normalized_reprojection_loss(
                left_px,  mp_left_t[:n_traj],  left_mask_2d,  focal,
                reproj_huber_delta / focal, weights=lw_w)
            loss_reproj_2d += normalized_reprojection_loss(
                right_px, mp_right_t[:n_traj], right_mask_2d, focal,
                reproj_huber_delta / focal, weights=rw_w)

        loss_prior  = ((arm_pose_6d - body_pose_6d_base[:, arm_body_pose_indices])**2).mean()
        arm_acc     = arm_pose_6d[2:] - 2*arm_pose_6d[1:-1] + arm_pose_6d[:-2]
        loss_smooth = (arm_acc**2).mean()

        # Wrist-position acceleration penalty (directly regularises the metric
        # we care about — joint-position jitter, which arm-rotation smoothness
        # only indirectly bounds).
        loss_smooth_pos = torch.tensor(0.0, device=device)
        if w_smooth_pos > 0 and joints.shape[0] >= 3:
            lw_pos = joints[:, LEFT_WRIST_IDX]
            rw_pos = joints[:, RIGHT_WRIST_IDX]
            lw_acc = lw_pos[2:] - 2*lw_pos[1:-1] + lw_pos[:-2]
            rw_acc = rw_pos[2:] - 2*rw_pos[1:-1] + rw_pos[:-2]
            loss_smooth_pos = (lw_acc**2).mean() + (rw_acc**2).mean()

        # Wrist-orientation loss: SMPL-X global wrist rotation (computed via
        # kinematic chain forward kinematics) vs Dyn-HaMR's wrist orientation
        # transformed into UEM body-world frame. Constrains the arm chain
        # rotation, which the wrist-position loss alone leaves under-determined
        # (multiple arm poses can place the wrist at the same point).
        loss_wrist_orient = torch.tensor(0.0, device=device)
        if use_wrist_orient:
            go = opt_params["global_orient"]
            bp = opt_params["body_pose"]
            T_ = min(go.shape[0], lrot_t.shape[0])
            R_l_smpl = smplx_global_wrist_rot(go[:T_], bp[:T_], _LWRIST_CHAIN)
            R_r_smpl = smplx_global_wrist_rot(go[:T_], bp[:T_], _RWRIST_CHAIN)
            diff_l = (R_l_smpl - lrot_t[:T_]).reshape(T_, -1)
            diff_r = (R_r_smpl - rrot_t[:T_]).reshape(T_, -1)
            lf_l = lv[:T_].float()
            lf_r = rv[:T_].float()
            if lf_l.sum() > 0:
                loss_wrist_orient = loss_wrist_orient + (
                    (diff_l ** 2).sum(dim=-1) * lf_l).sum() / lf_l.sum().clamp_min(1)
            if lf_r.sum() > 0:
                loss_wrist_orient = loss_wrist_orient + (
                    (diff_r ** 2).sum(dim=-1) * lf_r).sum() / lf_r.sum().clamp_min(1)

        loss = (w_wrist * loss_wrist
                + w_reproj_2d * loss_reproj_2d
                + w_pose_prior * loss_prior
                + w_smooth * loss_smooth
                + w_smooth_pos * loss_smooth_pos
                + w_wrist_orient * loss_wrist_orient)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        bp6d = body_pose_6d_base.clone()
        bp6d[:, arm_body_pose_indices] = arm_pose_6d
        opt_params["body_pose"] = rc.rotation_6d_to_matrix(bp6d)
        joints, verts, _ = evaluate_smpl_grad(smpl, opt_params)
    smpl.to(smpl_orig_device)
    return {k: v.detach().cpu() for k, v in opt_params.items()}, joints.detach().cpu(), verts.detach().cpu()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    input_examples, remaining_argv, cli_args = parse_vis_args()
    sys.argv = [sys.argv[0]] + remaining_argv
    device = torch.device("cuda")

    # Ablation master switch — override all Dyn-HaMR-driven knobs at once.
    if cli_args.no_dyn_hamr:
        cli_args.w_dyn = 0.0
        cli_args.bad_align_w_dyn = 0.0
        cli_args.run_diffusion_wrist_guidance = False
        cli_args.use_dyn_hand_pose = False
        print("[ABLATION] no_dyn_hamr=True -> w_dyn=0, diffusion wrist guidance "
              "OFF, Dyn-HaMR finger pose OFF. 2D MediaPipe reproj unchanged.")

    # Filename tag: "_<tag>" when --run_tag given, else "" (legacy filenames).
    run_tag_suffix = f"_{cli_args.run_tag}" if cli_args.run_tag else ""

    pl.seed_everything(62, workers=True)
    sys.argv = sys.argv + ["TRAIN.ONLY_VALIDATE", "True"]
    cfg = get_cfg()
    assert cfg.TRAIN.EXP_PATH is not None and os.path.exists(cfg.TRAIN.EXP_PATH)

    ds_name = "ee4d"
    out_dir = f"{cfg.TRAIN.EXP_PATH}/{ds_name}_vis{cfg.TRAIN.EVAL_SUFFIX}"
    os.makedirs(out_dir, exist_ok=True)

    is_twostage = cfg.MODEL.TRAJ_CKPT_PATH is not None
    if not is_twostage:
        ckpt_path = cfg.MODEL.CKPT_PATH
        if cfg.MODEL.CKPT_PATH == "last_ckpt":
            ckpt_path = os.path.join(cfg.TRAIN.EXP_PATH, "last.ckpt")
        assert os.path.exists(ckpt_path)
        model = UEM_Module.load_from_checkpoint(ckpt_path, cfg=cfg, map_location="cpu").to(device).eval()
    else:
        model = UEM_Module_TwoStage(cfg=cfg).to(device).eval()

    smpl = get_smpl()
    use_blender = False

    for split in ["val"]:
        ds = EE4D_Motion_Dataset(
            data_dir=cfg.DATA.DATA_DIR, split=split, repre_type=cfg.DATA.REPRE_TYPE,
            cond_img_feat=cfg.DATA.COND_IMG_FEAT, cond_traj=cfg.DATA.COND_TRAJ,
            window=cfg.DATA.WINDOW, img_feat_type=cfg.DATA.IMG_FEAT_TYPE,
            cond_betas=cfg.DATA.COND_BETAS,
        )

        task = "recon"
        for seq_name, start_idx in input_examples:
            for jdx in range(1):
                if is_twostage and task == "recon":
                    continue
                with torch.inference_mode():
                    idx    = 0
                    sample = ds.get_from_seq_and_st(seq_name, start_idx, idx)
                    pred_sample = ds.process_sample_for_task(sample, task)
                    batch       = careful_collate_fn([sample])
                    pred_batch  = careful_collate_fn([pred_sample])
                    y = to_device(pred_batch["y"], device)

                    # ----------------------------------------------------------
                    # CAMSPACE: compute aria_traj_T from conditioning trajectory
                    # BEFORE any model pass — this is what enables single-pass guidance.
                    # ----------------------------------------------------------
                    traj_norm  = pred_batch["y"]["traj"][0]
                    aria_traj_T = aria_traj_T_from_conditioning(traj_norm, ds, device)

                    # Load hand guidance
                    hand_guidance = load_hand_guidance(
                        cli_args.hand_guidance_dir, seq_name, start_idx)

                    # Load 2D MediaPipe guidance
                    guidance_2d = load_2d_guidance(
                        cli_args.guidance_2d_dir, seq_name, start_idx)

                    # ----------------------------------------------------------
                    # CAMSPACE: resolve aligned wrist targets before diffusion
                    # ----------------------------------------------------------
                    cam_dyn_left  = None
                    cam_dyn_right = None
                    cam_space_used = False
                    traj_scale = None
                    traj_R     = None
                    traj_t     = None

                    if hand_guidance is not None and cli_args.use_traj_align:
                        has_left_hg  = bool(hand_guidance["has_left"])  if "has_left"  in hand_guidance.files else True
                        has_right_hg = bool(hand_guidance["has_right"]) if "has_right" in hand_guidance.files else True
                        w2c_hg = None
                        if "left_w2c" in hand_guidance.files and has_left_hg:
                            w2c_hg = hand_guidance["left_w2c"].astype(np.float32)
                        elif "right_w2c" in hand_guidance.files and has_right_hg:
                            w2c_hg = hand_guidance["right_w2c"].astype(np.float32)

                        if w2c_hg is not None:
                            traj_aln = traj_align_dyn_to_uem(w2c_hg, aria_traj_T)
                            if traj_aln is not None:
                                traj_scale, traj_R, traj_t, tr_n = traj_aln
                                dyn_l_hg = hand_guidance["left_trans"].astype(np.float32)
                                dyn_r_hg = hand_guidance["right_trans"].astype(np.float32)
                                n_hg = min(len(dyn_l_hg), len(dyn_r_hg))
                                cam_dyn_left  = apply_similarity(dyn_l_hg[:n_hg], traj_scale, traj_R, traj_t)
                                cam_dyn_right = apply_similarity(dyn_r_hg[:n_hg], traj_scale, traj_R, traj_t)
                                cam_space_used = True
                                left_valid_hg  = hand_guidance["left_valid"].astype(bool)[:n_hg]
                                right_valid_hg = hand_guidance["right_valid"].astype(bool)[:n_hg]
                                print(
                                    f"[TRAJAL] Trajectory alignment: scale={traj_scale:.4f}, "
                                    f"n_traj_pts={tr_n}, n_hg={n_hg}, "
                                    f"left_valid={left_valid_hg.sum()}, right_valid={right_valid_hg.sum()}"
                                )
                            else:
                                print("[TRAJAL] Too few trajectory frames for alignment (<4). "
                                      "Falling back to similarity-transform bootstrap.")
                        else:
                            print("[TRAJAL] No w2c found in hand_guidance — "
                                  "falling back to similarity-transform bootstrap.")

                    # ----------------------------------------------------------
                    # Diffusion sampling
                    # ----------------------------------------------------------
                    if (cam_space_used
                            and cli_args.run_diffusion_wrist_guidance
                            and not is_twostage
                            and not cfg.MODEL.LEARN_TRAJ):
                        # CAMSPACE: single pass — guidance available before sampling
                        left_valid_hg  = hand_guidance["left_valid"].astype(bool)[:n_hg]
                        right_valid_hg = hand_guidance["right_valid"].astype(bool)[:n_hg]
                        denoised_fn = make_diffusion_wrist_denoised_fn(
                            traj_norm=traj_norm,
                            ds=ds,
                            left_wrist_world=cam_dyn_left,
                            right_wrist_world=cam_dyn_right,
                            left_valid=left_valid_hg,
                            right_valid=right_valid_hg,
                            strength=cli_args.diffusion_wrist_guidance_strength,
                            start_frac=cli_args.diffusion_wrist_guidance_start_frac,
                            num_diffusion_steps=model.diffusion.num_timesteps,
                            device=device,
                        )
                        x = model.sample(y, 1, cond_scale=cfg.TRAIN.COND_SCALE,
                                         denoised_fn=denoised_fn)
                        print("[TRAJAL] Single guided diffusion pass (no bootstrap needed).")
                    else:
                        # Original path: clean first pass
                        x = model.sample(y, 1, cond_scale=cfg.TRAIN.COND_SCALE)

                        # Original similarity-transform bootstrap for diffusion guidance
                        if (hand_guidance is not None
                                and cli_args.run_diffusion_wrist_guidance
                                and not cam_space_used
                                and not is_twostage
                                and not cfg.MODEL.LEARN_TRAJ):
                            pred_batch["pred"]["motion"] = to_device(x, "cpu")
                            pred_mdata_tmp = ds.ret_to_full_sequence(pred_batch)
                            tmp_joints, _, _ = evaluate_smpl(
                                smpl, pred_mdata_tmp["smpl_params_full"][0])
                            tmp_joints = tmp_joints.detach().cpu().numpy()
                            init_left  = tmp_joints[:, 20, :]
                            init_right = tmp_joints[:, 21, :]
                            init_root  = tmp_joints[:,  0, :]
                            dyn_l = hand_guidance["left_trans"];  dyn_r = hand_guidance["right_trans"]
                            lv_   = hand_guidance["left_valid"].astype(bool)
                            rv_   = hand_guidance["right_valid"].astype(bool)
                            n_b   = min(len(init_left), len(dyn_l), len(lv_))
                            aln   = fit_shared_hand_alignment(
                                dyn_l[:n_b], dyn_r[:n_b],
                                init_left[:n_b] - init_root[:n_b],
                                init_right[:n_b] - init_root[:n_b],
                                lv_[:n_b], rv_[:n_b])
                            if aln is not None:
                                sc, R_, t_, pts_ = aln
                                ddl = init_root[:n_b] + apply_similarity(dyn_l[:n_b], sc, R_, t_)
                                ddr = init_root[:n_b] + apply_similarity(dyn_r[:n_b], sc, R_, t_)
                                print(f"[FALLBACK] Similarity-transform bootstrap: "
                                      f"scale={sc:.4f}, points={pts_}")
                                dfn = make_diffusion_wrist_denoised_fn(
                                    traj_norm=traj_norm, ds=ds,
                                    left_wrist_world=ddl, right_wrist_world=ddr,
                                    left_valid=lv_[:n_b], right_valid=rv_[:n_b],
                                    strength=cli_args.diffusion_wrist_guidance_strength,
                                    start_frac=cli_args.diffusion_wrist_guidance_start_frac,
                                    num_diffusion_steps=model.diffusion.num_timesteps,
                                    device=device)
                                x = model.sample(y, 1, cond_scale=cfg.TRAIN.COND_SCALE,
                                                 denoised_fn=dfn)

                    if not is_twostage:
                        if cfg.MODEL.LEARN_TRAJ:
                            pred_batch["pred"]["traj"]   = to_device(x, "cpu")
                        else:
                            pred_batch["pred"]["motion"] = to_device(x, "cpu")
                    else:
                        pred_batch["pred"]["traj"]   = to_device(x[0], "cpu")
                        pred_batch["pred"]["motion"] = to_device(x[1], "cpu")

                # ------------------------------------------------------------------
                # Post-diffusion: extract joints & run optimisation
                # ------------------------------------------------------------------
                gt_mdata   = ds.ret_to_full_sequence(batch)
                pred_mdata = ds.ret_to_full_sequence(pred_batch)
                gt_aria_traj_T   = gt_mdata["aria_traj_T"][0]
                gt_smpl_params   = gt_mdata["smpl_params_full"][0]
                pred_smpl_params = pred_mdata["smpl_params_full"][0]
                nf_pred = len(pred_smpl_params["global_orient"])

                pred_joints, pred_verts, _ = evaluate_smpl(smpl, pred_smpl_params)

                # Save original UniEgoMotion result BEFORE any guidance
                uem_joints = pred_joints.detach().cpu().clone()
                uem_verts  = pred_verts.detach().cpu().clone()

                gt_joints, gt_verts, _ = evaluate_smpl(smpl, gt_smpl_params)

                guided_wrist_jpos = None

                if hand_guidance is not None:
                    ROOT_IDX = 0; LEFT_WRIST_IDX = 20; RIGHT_WRIST_IDX = 21

                    pred_joints_np = pred_joints.detach().cpu().numpy()
                    gt_joints_np   = gt_joints.detach().cpu().numpy()
                    uem_body_pose_np = pred_smpl_params["body_pose"].detach().cpu().numpy()

                    pred_left_wrist  = pred_joints_np[:, LEFT_WRIST_IDX, :]
                    pred_right_wrist = pred_joints_np[:, RIGHT_WRIST_IDX, :]
                    pred_root        = pred_joints_np[:, ROOT_IDX, :]
                    gt_left_wrist    = gt_joints_np[:, LEFT_WRIST_IDX, :]
                    gt_right_wrist   = gt_joints_np[:, RIGHT_WRIST_IDX, :]
                    gt_root          = gt_joints_np[:, ROOT_IDX, :]

                    dyn_left_raw  = hand_guidance["left_trans"]
                    dyn_right_raw = hand_guidance["right_trans"]
                    left_valid    = hand_guidance["left_valid"].astype(bool)
                    right_valid   = hand_guidance["right_valid"].astype(bool)
                    left_cam_R    = hand_guidance["left_cam_R"]  if "left_cam_R"  in hand_guidance.files else None
                    right_cam_R   = hand_guidance["right_cam_R"] if "right_cam_R" in hand_guidance.files else None
                    left_cam_t    = hand_guidance["left_cam_t"]  if "left_cam_t"  in hand_guidance.files else None
                    right_cam_t   = hand_guidance["right_cam_t"] if "right_cam_t" in hand_guidance.files else None
                    intrins       = hand_guidance["intrins"]      if "intrins"     in hand_guidance.files else None

                    n = min(len(pred_joints_np), len(gt_joints_np),
                            len(dyn_left_raw), len(dyn_right_raw),
                            len(left_valid),   len(right_valid))
                    if left_cam_R is not None:  n = min(n, len(left_cam_R))
                    if right_cam_R is not None: n = min(n, len(right_cam_R))

                    pred_left_wrist  = pred_left_wrist[:n];   pred_right_wrist = pred_right_wrist[:n]
                    gt_left_wrist    = gt_left_wrist[:n];     gt_right_wrist   = gt_right_wrist[:n]
                    pred_root        = pred_root[:n];          gt_root          = gt_root[:n]
                    dyn_left_raw     = dyn_left_raw[:n];       dyn_right_raw    = dyn_right_raw[:n]
                    left_valid       = left_valid[:n];         right_valid      = right_valid[:n]
                    if left_cam_R  is not None: left_cam_R  = left_cam_R[:n]
                    if right_cam_R is not None: right_cam_R = right_cam_R[:n]
                    if left_cam_t  is not None: left_cam_t  = left_cam_t[:n]
                    if right_cam_t is not None: right_cam_t = right_cam_t[:n]

                    pred_left_rel  = pred_left_wrist  - pred_root
                    pred_right_rel = pred_right_wrist - pred_root

                    # ----------------------------------------------------------
                    # CAMSPACE: aligned wrists for optimisation
                    # ----------------------------------------------------------
                    if cam_space_used and cam_dyn_left is not None:
                        dyn_left_aligned  = cam_dyn_left[:n]
                        dyn_right_aligned = cam_dyn_right[:n]
                        shared_scale = float(traj_scale) if traj_scale is not None else 1.0
                        shared_R     = np.asarray(traj_R) if traj_R is not None else np.eye(3)
                        shared_t     = np.asarray(traj_t) if traj_t is not None else np.zeros(3)
                        shared_align_points = n
                        print(f"[TRAJAL] Using trajectory-aligned wrists for optimisation "
                              f"(n={n}, scale={shared_scale:.4f}).")
                    else:
                        # Fallback: similarity transform from model predictions
                        shared_alignment = fit_shared_hand_alignment(
                            dyn_left_raw, dyn_right_raw, pred_left_rel, pred_right_rel,
                            left_valid, right_valid)
                        if shared_alignment is not None:
                            shared_scale, shared_R, shared_t, shared_align_points = shared_alignment
                            dyn_left_aligned  = pred_root + apply_similarity(dyn_left_raw, shared_scale, shared_R, shared_t)
                            dyn_right_aligned = pred_root + apply_similarity(dyn_right_raw, shared_scale, shared_R, shared_t)
                            print(f"[FALLBACK] Similarity-transform alignment: "
                                  f"scale={shared_scale:.4f}, points={shared_align_points}")
                        else:
                            shared_scale = 1.0; shared_R = np.eye(3); shared_t = np.zeros(3)
                            shared_align_points = 0
                            dyn_left_aligned  = pred_left_wrist.copy()
                            dyn_right_aligned = pred_right_wrist.copy()
                            print("[WARN] Alignment failed; using UEM wrists.")

                    # ----------------------------------------------------------
                    # Reliability scoring (same as original)
                    # ----------------------------------------------------------
                    dyn_reliability = compute_framewise_dyn_guidance_reliability(
                        dyn_left_aligned, dyn_right_aligned,
                        pred_left_wrist, pred_right_wrist,
                        left_valid, right_valid,
                        vel_good_ratio=cli_args.dyn_vel_good_ratio,
                        vel_bad_ratio=cli_args.dyn_vel_bad_ratio,
                        acc_good_ratio=cli_args.dyn_acc_good_ratio,
                        acc_bad_ratio=cli_args.dyn_acc_bad_ratio,
                        sep_good_ratio=cli_args.dyn_sep_good_ratio,
                        sep_bad_ratio=cli_args.dyn_sep_bad_ratio,
                        window=cli_args.dyn_frame_gate_window,
                    )
                    if cli_args.disable_dyn_reliability_gate:
                        for k in ["dyn_vel_frame_weights", "dyn_acc_frame_weights", "dyn_sep_frame_weights"]:
                            dyn_reliability[k] = np.ones_like(dyn_reliability[k])
                        dyn_reliability["dyn_vel_reliability"] = 1.0
                        dyn_reliability["dyn_acc_reliability"] = 1.0
                        dyn_reliability["dyn_sep_reliability"] = 1.0

                    # CAMSPACE: no scale gate (metric alignment)
                    if cam_space_used:
                        effective_w_dyn  = cli_args.w_dyn
                        dyn_pos_trusted  = True
                    else:
                        effective_w_dyn, dyn_pos_trusted = dyn_position_weight_from_scale(
                            shared_scale, cli_args.w_dyn,
                            min_scale=cli_args.dyn_pos_min_scale,
                            max_scale=cli_args.dyn_pos_max_scale,
                            bad_align_w_dyn=cli_args.bad_align_w_dyn)

                    effective_w_vel = cli_args.w_vel
                    effective_w_acc = cli_args.w_acc
                    effective_w_sep = cli_args.w_sep

                    print(f"[RELIABILITY] vel_ratio={dyn_reliability['dyn_vel_consistency_ratio']:.4f} "
                          f"score={dyn_reliability['dyn_vel_reliability']:.4f}, "
                          f"acc_ratio={dyn_reliability['dyn_acc_consistency_ratio']:.4f} "
                          f"score={dyn_reliability['dyn_acc_reliability']:.4f}, "
                          f"sep_ratio={dyn_reliability['dyn_sep_consistency_ratio']:.4f} "
                          f"score={dyn_reliability['dyn_sep_reliability']:.4f}")
                    print(f"[GUIDANCE] traj_align={cam_space_used}, w_dyn={effective_w_dyn:.4f}, "
                          f"trusted_pos={dyn_pos_trusted}, scale={shared_scale:.4f}")

                    guided_left_wrist  = pred_left_wrist.copy()
                    guided_right_wrist = pred_right_wrist.copy()

                    # Optional projection-error-driven per-frame trust weights for
                    # the 3D wrist guidance loss (orthogonal to dyn_reliability gating).
                    proj_left_weights = proj_right_weights = None
                    if cli_args.use_projection_error_guidance:
                        projection_error_npz = cli_args.projection_error_npz

                        if cli_args.projection_error_dir is not None:
                            seq_key = f"{seq_name}st{start_idx}_uem80"
                            projection_error_npz = os.path.join(
                                cli_args.projection_error_dir,
                                f"{seq_key}_projection_error.npz"
                            )

                        print(f"[PROJ_GUIDANCE] Looking for projection-error npz: {projection_error_npz}")

                        proj_weights = load_projection_error_guidance_weights(
                            projection_error_npz, n, cli_args)
                        if proj_weights is not None:
                            proj_left_weights, proj_right_weights = proj_weights
                            print(f"[PROJ_GUIDANCE] Using projection-error weights: "
                                  f"L_mean={proj_left_weights.mean():.4f}, "
                                  f"R_mean={proj_right_weights.mean():.4f}")
                        else:
                            print("[PROJ_GUIDANCE] No projection-error npz found; "
                                  "falling back to uniform 3D guidance weight.")

                    if cli_args.run_guidance_opt:
                        guided_left_wrist, guided_right_wrist = optimize_wrist_guidance(
                            pred_left=pred_left_wrist, pred_right=pred_right_wrist,
                            dyn_left=dyn_left_aligned, dyn_right=dyn_right_aligned,
                            left_valid=left_valid, right_valid=right_valid,
                            num_iters=cli_args.guidance_iters, lr=cli_args.guidance_lr,
                            w_prior=cli_args.w_prior, w_dyn=effective_w_dyn,
                            w_vel=effective_w_vel, w_acc=effective_w_acc,
                            w_sep=effective_w_sep, w_smooth=cli_args.w_smooth,
                            pos_weights_l=proj_left_weights,
                            pos_weights_r=proj_right_weights,
                            vel_weights=dyn_reliability["dyn_vel_frame_weights"],
                            acc_weights=dyn_reliability["dyn_acc_frame_weights"],
                            sep_weights=dyn_reliability["dyn_sep_frame_weights"],
                            device=device)

                    smpl_opt_joints_np = smpl_opt_left_wrist = smpl_opt_right_wrist = None
                    smpl_opt_arm_pose_dev_deg = np.nan
                    effective_smpl_w_reproj   = 0.0

                    if cli_args.run_smpl_opt:
                        # Prefer 2D MediaPipe reprojection over Dyn-HaMR reprojection
                        use_2d_reproj = (
                            guidance_2d is not None
                            and cli_args.w_reproj_2d > 0
                        )

                        # Dyn-HaMR wrist orientation transformed Dyn-HaMR-world → UEM
                        # body-world: R_uem = shared_R @ R_dyn @ shared_R.T (change of basis).
                        dyn_l_rot_uem = dyn_r_rot_uem = None
                        if (cli_args.w_smpl_wrist_orient > 0 and hand_guidance is not None
                                and "left_root_orient_rotmat"  in hand_guidance.files
                                and "right_root_orient_rotmat" in hand_guidance.files):
                            l_rot_dyn = hand_guidance["left_root_orient_rotmat" ].astype(np.float32)
                            r_rot_dyn = hand_guidance["right_root_orient_rotmat"].astype(np.float32)
                            sR = np.asarray(shared_R, dtype=np.float32)
                            dyn_l_rot_uem = sR @ l_rot_dyn @ sR.T
                            dyn_r_rot_uem = sR @ r_rot_dyn @ sR.T

                        if use_2d_reproj:
                            mp_lw2d_raw = guidance_2d["left_wrist_2d"].astype(np.float32)
                            mp_rw2d_raw = guidance_2d["right_wrist_2d"].astype(np.float32)
                            lconf = guidance_2d["left_conf_2d"].astype(np.float32)
                            rconf = guidance_2d["right_conf_2d"].astype(np.float32)
                            lv2d = (guidance_2d["left_valid_2d"].astype(bool)
                                    & (lconf >= cli_args.reproj_2d_min_conf))
                            rv2d = (guidance_2d["right_valid_2d"].astype(bool)
                                    & (rconf >= cli_args.reproj_2d_min_conf))
                            # Temporally smooth detections to reduce per-frame jitter
                            # that the optimizer would otherwise track.
                            mp_lw2d = smooth_2d_detections(mp_lw2d_raw, lv2d, cli_args.mp_smooth_sigma)
                            mp_rw2d = smooth_2d_detections(mp_rw2d_raw, rv2d, cli_args.mp_smooth_sigma)
                            n2d  = min(n, len(mp_lw2d), len(lv2d))
                            print(f"[SMPL_OPT_2D] iters={cli_args.smpl_opt_iters}, "
                                  f"w_wrist={cli_args.smpl_w_wrist}, "
                                  f"w_reproj_2d={cli_args.w_reproj_2d}, "
                                  f"w_smooth_pos={cli_args.w_smpl_smooth_pos}, "
                                  f"w_wrist_orient={cli_args.w_smpl_wrist_orient}, "
                                  f"mp_sigma={cli_args.mp_smooth_sigma}, "
                                  f"2d_valid_left={lv2d[:n2d].sum()}, "
                                  f"2d_valid_right={rv2d[:n2d].sum()}")
                            pred_smpl_params, pred_joints, pred_verts = optimize_smpl_arm_2d(
                                smpl=smpl, pred_smpl_params=pred_smpl_params,
                                guided_left_wrist=guided_left_wrist,
                                guided_right_wrist=guided_right_wrist,
                                left_valid=left_valid, right_valid=right_valid,
                                mp_left_2d=mp_lw2d[:n2d], mp_right_2d=mp_rw2d[:n2d],
                                left_valid_2d=lv2d[:n2d], right_valid_2d=rv2d[:n2d],
                                left_conf_2d=lconf[:n2d], right_conf_2d=rconf[:n2d],
                                dyn_left_rot_uem=dyn_l_rot_uem,
                                dyn_right_rot_uem=dyn_r_rot_uem,
                                gt_aria_traj_T=gt_aria_traj_T,
                                intrins=ARIA_RGB_INTRINS,
                                num_iters=cli_args.smpl_opt_iters, lr=cli_args.smpl_opt_lr,
                                w_wrist=cli_args.smpl_w_wrist,
                                w_reproj_2d=cli_args.w_reproj_2d,
                                w_pose_prior=cli_args.smpl_w_pose_prior,
                                w_smooth=cli_args.smpl_w_smooth,
                                w_smooth_pos=cli_args.w_smpl_smooth_pos,
                                w_wrist_orient=cli_args.w_smpl_wrist_orient,
                                reproj_huber_delta=8.0,
                                device=device)
                        else:
                            smpl_reproj_available = (
                                left_cam_R is not None and right_cam_R is not None
                                and left_cam_t is not None and right_cam_t is not None
                                and intrins is not None and shared_align_points > 0
                            )
                            effective_smpl_w_reproj = (cli_args.smpl_w_reproj
                                if cli_args.run_smpl_reproj_opt and smpl_reproj_available else 0.0)
                            print(f"[SMPL_OPT] iters={cli_args.smpl_opt_iters}, "
                                  f"w_wrist={cli_args.smpl_w_wrist}, "
                                  f"w_reproj={effective_smpl_w_reproj}")
                            pred_smpl_params, pred_joints, pred_verts = optimize_smpl_arm_guidance(
                                smpl=smpl, pred_smpl_params=pred_smpl_params,
                                guided_left_wrist=guided_left_wrist,
                                guided_right_wrist=guided_right_wrist,
                                left_valid=left_valid, right_valid=right_valid,
                                num_iters=cli_args.smpl_opt_iters, lr=cli_args.smpl_opt_lr,
                                w_wrist=cli_args.smpl_w_wrist,
                                w_reproj=effective_smpl_w_reproj,
                                reproj_huber_delta=cli_args.smpl_reproj_huber_delta,
                                dyn_left_raw=dyn_left_raw, dyn_right_raw=dyn_right_raw,
                                left_cam_R=left_cam_R, right_cam_R=right_cam_R,
                                left_cam_t=left_cam_t, right_cam_t=right_cam_t,
                                intrins=intrins,
                                shared_scale=shared_scale, shared_R=shared_R, shared_t=shared_t,
                                w_pose_prior=cli_args.smpl_w_pose_prior,
                                w_smooth=cli_args.smpl_w_smooth, device=device)
                        smpl_opt_joints_np    = pred_joints.detach().cpu().numpy()
                        smpl_opt_left_wrist   = smpl_opt_joints_np[:n, LEFT_WRIST_IDX, :]
                        smpl_opt_right_wrist  = smpl_opt_joints_np[:n, RIGHT_WRIST_IDX, :]
                        arm_idxs              = np.array([12,13,15,16,17,18,19,20])
                        smpl_opt_arm_pose_dev_deg = rotation_geodesic_deg(
                            pred_smpl_params["body_pose"][:n, arm_idxs].detach().cpu().numpy(),
                            uem_body_pose_np[:n, arm_idxs])

                    # Replace UEM's PCA hand pose with Dyn-HaMR's actual finger
                    # articulation. Wrist position is unchanged -- only the 15 finger
                    # joints downstream of the wrist. This makes renders qualitatively
                    # better (visible fingers grasping objects) without affecting the
                    # body/wrist MPJPE metrics.
                    dyn_hand_used = False
                    if cli_args.use_dyn_hand_pose and hand_guidance is not None:
                        pred_smpl_params, n_l_hand, n_r_hand = override_hand_pose_with_dyn_hamr(
                            pred_smpl_params, hand_guidance)
                        if n_l_hand or n_r_hand:
                            dyn_hand_used = True
                            print(f"[HAND] Injected Dyn-HaMR finger pose: "
                                  f"L={n_l_hand} frames, R={n_r_hand} frames")
                            # Re-evaluate so pred_verts reflects new finger pose for render.
                            pred_joints, pred_verts, _ = evaluate_smpl(smpl, pred_smpl_params)

                    guided_wrist_jpos = np.ones((nf_pred, 2, 3), dtype=np.float32) * -100.0
                    guided_wrist_jpos[:n, 0] = guided_left_wrist
                    guided_wrist_jpos[:n, 1] = guided_right_wrist

                    # ----------------------------------------------------------
                    # Eval metrics
                    # ----------------------------------------------------------
                    eval_metrics = {}
                    eval_metrics.update(compute_motion_metrics(
                        "uem", pred_joints_np[:n], gt_joints_np[:n], left_valid, right_valid))
                    eval_metrics.update(compute_wrist_metrics(
                        "dyn_hamr", dyn_left_aligned, dyn_right_aligned,
                        gt_left_wrist, gt_right_wrist, left_valid, right_valid))
                    eval_metrics.update(compute_wrist_metrics(
                        "guided", guided_left_wrist, guided_right_wrist,
                        gt_left_wrist, gt_right_wrist, left_valid, right_valid))

                    # Root-relative + Procrustes wrist error: isolates arm
                    # articulation from whole-body placement. pred_root / gt_root
                    # are the pelvis joints already sliced to [:n] above.
                    eval_metrics.update(compute_wrist_metrics_decomposed(
                        "uem", pred_left_wrist, pred_right_wrist,
                        gt_left_wrist, gt_right_wrist, left_valid, right_valid,
                        pred_root=pred_root, gt_root=gt_root))
                    eval_metrics.update(compute_wrist_metrics_decomposed(
                        "dyn_hamr", dyn_left_aligned, dyn_right_aligned,
                        gt_left_wrist, gt_right_wrist, left_valid, right_valid,
                        pred_root=pred_root, gt_root=gt_root))
                    eval_metrics.update(compute_wrist_metrics_decomposed(
                        "guided", guided_left_wrist, guided_right_wrist,
                        gt_left_wrist, gt_right_wrist, left_valid, right_valid,
                        pred_root=pred_root, gt_root=gt_root))

                    full_body_names = ["uem"]
                    wrist_names     = ["uem", "dyn_hamr", "guided"]

                    if smpl_opt_joints_np is not None:
                        eval_metrics.update(compute_motion_metrics(
                            "smpl_opt", smpl_opt_joints_np[:n], gt_joints_np[:n],
                            left_valid, right_valid))
                        smpl_opt_root = smpl_opt_joints_np[:n, ROOT_IDX, :]
                        eval_metrics.update(compute_wrist_metrics_decomposed(
                            "smpl_opt", smpl_opt_left_wrist, smpl_opt_right_wrist,
                            gt_left_wrist, gt_right_wrist, left_valid, right_valid,
                            pred_root=smpl_opt_root, gt_root=gt_root))
                        eval_metrics["smpl_opt_arm_pose_dev_deg"] = smpl_opt_arm_pose_dev_deg
                        full_body_names.append("smpl_opt")
                        wrist_names.append("smpl_opt")

                    print(f"[EVAL] Frames={n}, valid_left={int(left_valid.sum())}, "
                          f"valid_right={int(right_valid.sum())}, "
                          f"traj_align={cam_space_used}, scale={shared_scale:.4f}")
                    print_comprehensive_eval(eval_metrics, full_body_names, wrist_names)

                    if cli_args.save_eval_npz:
                        save_path = os.path.join(
                            out_dir,
                            f"{seq_name}_st{start_idx}{run_tag_suffix}_traj_align_eval.npz")
                        extra_save = {}
                        if smpl_opt_left_wrist is not None:
                            extra_save["smpl_opt_left_wrist"]  = smpl_opt_left_wrist
                            extra_save["smpl_opt_right_wrist"] = smpl_opt_right_wrist
                            extra_save["smpl_opt_root"] = smpl_opt_joints_np[:n, ROOT_IDX, :]
                        np.savez_compressed(save_path,
                            seq_name=seq_name, start_idx=start_idx,
                            cam_space_used=cam_space_used,
                            shared_scale=shared_scale,
                            pred_left_wrist=pred_left_wrist, pred_right_wrist=pred_right_wrist,
                            gt_left_wrist=gt_left_wrist,     gt_right_wrist=gt_right_wrist,
                            pred_root=pred_root, gt_root=gt_root,
                            dyn_left_aligned=dyn_left_aligned, dyn_right_aligned=dyn_right_aligned,
                            guided_left_wrist=guided_left_wrist,
                            guided_right_wrist=guided_right_wrist,
                            left_valid=left_valid, right_valid=right_valid,
                            **extra_save,
                            **eval_metrics)
                        print(f"[INFO] Saved: {save_path}")

                # Try and visualise GT, UEM and guided on same
                vis_fn = visualize_sequence_blender if use_blender else visualize_sequence
                
                # GT panel
                imgs_gt = vis_fn(
                    aria_traj=gt_aria_traj_T,
                    verts=None,
                    faces=smpl.faces,
                    pred_verts=gt_verts,
                    pred_global_jpos=None,
                )

                # Original UniEgoMotion panel
                imgs_uem = vis_fn(
                    aria_traj=gt_aria_traj_T,
                    verts=None,
                    faces=smpl.faces,
                    pred_verts=uem_verts,
                    pred_global_jpos=None,
                )

                # Our guided / optimized result panel
                imgs_guided = vis_fn(
                    aria_traj=gt_aria_traj_T,
                    verts=None,
                    faces=smpl.faces,
                    pred_verts=pred_verts,
                    pred_global_jpos=None,
                )

                # concatenate horizontally: GT | UEM | Guided
                imgs = np.concatenate([imgs_gt, imgs_uem, imgs_guided], axis=2)

                seq_name_b = batch["misc"]["seq_name"][0]
                st, en = batch["misc"]["start_idx"][0], batch["misc"]["end_idx"][0]
                hand_tag = "_dynhand" if dyn_hand_used else ""

                save_video(
                    imgs[..., ::-1],
                    f"{split}_idx{idx}_{seq_name_b}_{st}_{en}_{task}_{jdx}_gt_uem_guided{run_tag_suffix}{hand_tag}",
                    out_dir,
                    fps=10,
                )


if __name__ == "__main__":
    main()
