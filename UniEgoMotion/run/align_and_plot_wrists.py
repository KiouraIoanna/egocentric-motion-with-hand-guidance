import numpy as np
import matplotlib.pyplot as plt
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

uem = np.load(UEM)
dyn = np.load(DYN)

uem_left = uem[f"{SEQ}__left_wrist"]
uem_right = uem[f"{SEQ}__right_wrist"]
dyn_hand = dyn["trans"][0]

T = min(len(uem_left), len(dyn_hand))
uem_left = uem_left[:T]
uem_right = uem_right[:T]
dyn_hand = dyn_hand[:T]

def align_rigid(A, B):
    """
    Find R,t such that A_aligned = A @ R.T + t approximates B.
    A, B: [T, 3]
    """
    A_mean = A.mean(axis=0)
    B_mean = B.mean(axis=0)

    A0 = A - A_mean
    B0 = B - B_mean

    H = A0.T @ B0
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # avoid reflection
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    t = B_mean - A_mean @ R.T
    A_aligned = A @ R.T + t
    return A_aligned, R, t

def errors(A, B):
    e = np.linalg.norm(A - B, axis=1)
    return e

# Align Dyn-HaMR to UEM left and right separately
dyn_to_left, R_left, t_left = align_rigid(dyn_hand, uem_left)
dyn_to_right, R_right, t_right = align_rigid(dyn_hand, uem_right)

err_left_before = errors(dyn_hand, uem_left)
err_right_before = errors(dyn_hand, uem_right)

err_left_after = errors(dyn_to_left, uem_left)
err_right_after = errors(dyn_to_right, uem_right)

print("Frames:", T)
print("Dyn is_right mean:", dyn["is_right"].mean())

print("\nBEFORE alignment")
print("left  mean/median/min/max:", err_left_before.mean(), np.median(err_left_before), err_left_before.min(), err_left_before.max())
print("right mean/median/min/max:", err_right_before.mean(), np.median(err_right_before), err_right_before.min(), err_right_before.max())

print("\nAFTER rigid alignment")
print("left  mean/median/min/max:", err_left_after.mean(), np.median(err_left_after), err_left_after.min(), err_left_after.max())
print("right mean/median/min/max:", err_right_after.mean(), np.median(err_right_after), err_right_after.min(), err_right_after.max())

# choose better aligned side
if err_left_after.mean() < err_right_after.mean():
    best_side = "left"
    dyn_aligned = dyn_to_left
    uem_best = uem_left
    best_err = err_left_after
else:
    best_side = "right"
    dyn_aligned = dyn_to_right
    uem_best = uem_right
    best_err = err_right_after

print("\nBest side after alignment:", best_side)
print("best aligned mean meters:", best_err.mean())
print("best aligned median meters:", np.median(best_err))

# Save results
np.savez_compressed(
    "wrist_alignment_result.npz",
    dyn_original=dyn_hand,
    dyn_aligned=dyn_aligned,
    uem_left=uem_left,
    uem_right=uem_right,
    uem_best=uem_best,
    best_err=best_err,
    best_side=np.array(best_side),
)

# ---------- 3D plot ----------
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot(uem_left[:, 0], uem_left[:, 1], uem_left[:, 2], label="UEM left wrist")
ax.plot(uem_right[:, 0], uem_right[:, 1], uem_right[:, 2], label="UEM right wrist")
ax.plot(dyn_hand[:, 0], dyn_hand[:, 1], dyn_hand[:, 2], label="Dyn-HaMR original")
ax.plot(dyn_aligned[:, 0], dyn_aligned[:, 1], dyn_aligned[:, 2], label=f"Dyn-HaMR aligned to UEM {best_side}")

ax.scatter(uem_best[0, 0], uem_best[0, 1], uem_best[0, 2], marker="o", label="UEM start")
ax.scatter(dyn_aligned[0, 0], dyn_aligned[0, 1], dyn_aligned[0, 2], marker="x", label="Dyn aligned start")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

plt.tight_layout()
plt.savefig("wrist_alignment_3d.png", dpi=200)
print("\nsaved wrist_alignment_3d.png")
print("saved wrist_alignment_result.npz")