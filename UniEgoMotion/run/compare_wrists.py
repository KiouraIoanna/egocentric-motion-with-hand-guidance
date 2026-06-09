import numpy as np

parser.add_argument("--uem_hands", required=True)
parser.add_argument("--dynhamr", required=True)
parser.add_argument("--out", required=True)

uem = np.load(UEM)
dyn = np.load(DYN)

uem_left = uem[f"{SEQ}__left_wrist"]      # [80, 3]
uem_right = uem[f"{SEQ}__right_wrist"]    # [80, 3]
dyn_hand = dyn["trans"][0]                # [62, 3]

T = min(len(uem_left), len(dyn_hand))

uem_left = uem_left[:T]
uem_right = uem_right[:T]
dyn_hand = dyn_hand[:T]

left_err = np.linalg.norm(uem_left - dyn_hand, axis=1)
right_err = np.linalg.norm(uem_right - dyn_hand, axis=1)

print("Approx comparison: Dyn-HaMR skipped invisible frames, so comparing first", T, "UEM frames.")
print("Dyn-HaMR is_left because is_right mean =", dyn["is_right"].mean())

print("\nLeft wrist vs Dyn-HaMR hand/root:")
print("mean meters:  ", left_err.mean())
print("median meters:", np.median(left_err))
print("min meters:   ", left_err.min())
print("max meters:   ", left_err.max())

print("\nRight wrist vs Dyn-HaMR hand/root:")
print("mean meters:  ", right_err.mean())
print("median meters:", np.median(right_err))
print("min meters:   ", right_err.min())
print("max meters:   ", right_err.max())

np.savez_compressed(
    "wrist_difference_approx.npz",
    left_err=left_err,
    right_err=right_err,
    uem_left=uem_left,
    uem_right=uem_right,
    dyn_hand=dyn_hand,
)

print("\nsaved wrist_difference_approx.npz")