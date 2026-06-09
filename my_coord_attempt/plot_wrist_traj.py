import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot_one(ax, gt, uem, dyn, title):
    ax.plot(gt[:,0], gt[:,1], gt[:,2], label="GT", linewidth=3)
    ax.plot(uem[:,0], uem[:,1], uem[:,2], label="UEM", linewidth=2)
    ax.plot(dyn[:,0], dyn[:,1], dyn[:,2], label="Dyn-HaMR aligned", linewidth=2)

    ax.scatter(gt[0,0], gt[0,1], gt[0,2], marker="o", s=50)
    ax.scatter(dyn[0,0], dyn[0,1], dyn[0,2], marker="x", s=50)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", help="Path to *_traj_align_eval.npz")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    x = np.load(args.npz)

    gt_left = x["gt_left_wrist"]
    gt_right = x["gt_right_wrist"]
    uem_left = x["pred_left_wrist"]
    uem_right = x["pred_right_wrist"]
    dyn_left = x["dyn_left_aligned"]
    dyn_right = x["dyn_right_aligned"]

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection="3d")
    plot_one(ax1, gt_left, uem_left, dyn_left, "Left wrist trajectory")

    ax2 = fig.add_subplot(122, projection="3d")
    plot_one(ax2, gt_right, uem_right, dyn_right, "Right wrist trajectory")

    plt.tight_layout()

    if args.out is None:
        args.out = args.npz.replace(".npz", "_wrist_traj.png")

    plt.savefig(args.out, dpi=200)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()