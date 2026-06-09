import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ARIA_RGB_INTRINS = np.array([396.0, 336.3, 717.3, 877.8], dtype=np.float32)


def project_uem_world_to_pixels(points_world, aria_traj_T, intrins):
    R_c2w = aria_traj_T[:, :3, :3]
    t_c2w = aria_traj_T[:, :3, 3]

    pts_cam = np.einsum("tji,tj->ti", R_c2w, points_world - t_c2w)
    z = pts_cam[:, 2]

    depth_ok = z > 0.05
    z_safe = np.maximum(z, 0.05)

    fx, fy, cx, cy = intrins
    u = -fx * pts_cam[:, 0] / z_safe + cx
    v = -fy * pts_cam[:, 1] / z_safe + cy

    return np.stack([u, v], axis=-1), depth_ok


def mean_px_err(a, b, valid):
    if valid.sum() == 0:
        return np.nan
    return np.linalg.norm(a[valid] - b[valid], axis=-1).mean()


def plot_one(ax, gt_2d, uem_2d, dyn_2d, guided_2d, valid, title):
    gt = gt_2d[valid]
    uem = uem_2d[valid]
    dyn = dyn_2d[valid]
    guided = guided_2d[valid]

    if len(gt) == 0:
        ax.set_title(title + " - no valid points")
        return

    ax.plot(gt[:, 0], gt[:, 1], label="GT", linewidth=3)
    ax.plot(uem[:, 0], uem[:, 1], label="UEM", linewidth=2)
    ax.plot(dyn[:, 0], dyn[:, 1], label="Dyn-HaMR aligned", linewidth=2)
    ax.plot(guided[:, 0], guided[:, 1], label="Guided", linewidth=2)

    ax.scatter(gt[0, 0], gt[0, 1], marker="o", s=80)
    ax.scatter(dyn[0, 0], dyn[0, 1], marker="x", s=80)

    all_pts = np.concatenate([gt, uem, dyn, guided], axis=0)

    u_min, u_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    v_min, v_max = all_pts[:, 1].min(), all_pts[:, 1].max()

    center_u = (u_min + u_max) / 2
    center_v = (v_min + v_max) / 2

    size = max(u_max - u_min, v_max - v_min)
    size = max(size * 1.3, 100)

    ax.set_xlim(center_u - size / 2, center_u + size / 2)
    ax.set_ylim(center_v + size / 2, center_v - size / 2)

    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel("u pixel")
    ax.set_ylabel("v pixel")
    ax.legend()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", help="Path to *_traj_align_eval.npz")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    x = np.load(args.npz)

    if "gt_aria_traj_T" not in x.files:
        raise KeyError("Missing gt_aria_traj_T in npz.")

    required = [
        "gt_left_wrist", "gt_right_wrist",
        "pred_left_wrist", "pred_right_wrist",
        "dyn_left_aligned", "dyn_right_aligned",
        "guided_left_wrist", "guided_right_wrist",
        "left_valid", "right_valid",
    ]

    for k in required:
        if k not in x.files:
            raise KeyError(f"Missing key in npz: {k}")

    aria_traj_T = x["gt_aria_traj_T"]

    n = min(
        len(x["gt_left_wrist"]),
        len(x["pred_left_wrist"]),
        len(x["dyn_left_aligned"]),
        len(x["guided_left_wrist"]),
        len(aria_traj_T),
    )

    gt_left = x["gt_left_wrist"][:n]
    gt_right = x["gt_right_wrist"][:n]
    uem_left = x["pred_left_wrist"][:n]
    uem_right = x["pred_right_wrist"][:n]
    dyn_left = x["dyn_left_aligned"][:n]
    dyn_right = x["dyn_right_aligned"][:n]
    guided_left = x["guided_left_wrist"][:n]
    guided_right = x["guided_right_wrist"][:n]

    left_valid = x["left_valid"][:n].astype(bool)
    right_valid = x["right_valid"][:n].astype(bool)
    aria_traj_T = aria_traj_T[:n]

    gt_left_2d, gt_left_depth = project_uem_world_to_pixels(gt_left, aria_traj_T, ARIA_RGB_INTRINS)
    uem_left_2d, uem_left_depth = project_uem_world_to_pixels(uem_left, aria_traj_T, ARIA_RGB_INTRINS)
    dyn_left_2d, dyn_left_depth = project_uem_world_to_pixels(dyn_left, aria_traj_T, ARIA_RGB_INTRINS)
    guided_left_2d, guided_left_depth = project_uem_world_to_pixels(guided_left, aria_traj_T, ARIA_RGB_INTRINS)

    gt_right_2d, gt_right_depth = project_uem_world_to_pixels(gt_right, aria_traj_T, ARIA_RGB_INTRINS)
    uem_right_2d, uem_right_depth = project_uem_world_to_pixels(uem_right, aria_traj_T, ARIA_RGB_INTRINS)
    dyn_right_2d, dyn_right_depth = project_uem_world_to_pixels(dyn_right, aria_traj_T, ARIA_RGB_INTRINS)
    guided_right_2d, guided_right_depth = project_uem_world_to_pixels(guided_right, aria_traj_T, ARIA_RGB_INTRINS)

    left_valid = left_valid & gt_left_depth & uem_left_depth & dyn_left_depth & guided_left_depth
    right_valid = right_valid & gt_right_depth & uem_right_depth & dyn_right_depth & guided_right_depth

    print("2D reprojection errors against GT, pixels")
    print(f"Left  UEM    : {mean_px_err(uem_left_2d, gt_left_2d, left_valid):.2f}")
    print(f"Left  DynHaMR: {mean_px_err(dyn_left_2d, gt_left_2d, left_valid):.2f}")
    print(f"Left  Guided : {mean_px_err(guided_left_2d, gt_left_2d, left_valid):.2f}")
    print(f"Right UEM    : {mean_px_err(uem_right_2d, gt_right_2d, right_valid):.2f}")
    print(f"Right DynHaMR: {mean_px_err(dyn_right_2d, gt_right_2d, right_valid):.2f}")
    print(f"Right Guided : {mean_px_err(guided_right_2d, gt_right_2d, right_valid):.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    plot_one(
        axes[0],
        gt_left_2d,
        uem_left_2d,
        dyn_left_2d,
        guided_left_2d,
        left_valid,
        "Left wrist in Aria image plane",
    )

    plot_one(
        axes[1],
        gt_right_2d,
        uem_right_2d,
        dyn_right_2d,
        guided_right_2d,
        right_valid,
        "Right wrist in Aria image plane",
    )

    plt.tight_layout()

    if args.out is None:
        args.out = args.npz.replace(".npz", "_wrist_2d_camera_centered.png")

    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print("Saved:", args.out)


if __name__ == "__main__":
    main()