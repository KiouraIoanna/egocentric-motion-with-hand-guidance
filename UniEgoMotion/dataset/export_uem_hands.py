import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--pkl", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--seq_key", required=True)
args = parser.parse_args()

PKL = args.pkl
OUT = args.out
seq_key = args.seq_keys

d = np.load(INP)
export = {}

seqs = sorted(set(k.rsplit("__", 1)[0] for k in d.files))

LEFT_WRIST = 20
RIGHT_WRIST = 21

LEFT_HAND_START = 25
LEFT_HAND_END = 40

RIGHT_HAND_START = 40
RIGHT_HAND_END = 55

for seq in seqs:
    joints = d[f"{seq}__joints"]

    export[f"{seq}__left_wrist"] = joints[:, LEFT_WRIST, :]
    export[f"{seq}__right_wrist"] = joints[:, RIGHT_WRIST, :]

    export[f"{seq}__left_hand"] = joints[:, LEFT_HAND_START:LEFT_HAND_END, :]
    export[f"{seq}__right_hand"] = joints[:, RIGHT_HAND_START:RIGHT_HAND_END, :]

    export[f"{seq}__aria_traj_T"] = d[f"{seq}__aria_traj_T"]

    print(seq)
    print("  left_wrist ", export[f"{seq}__left_wrist"].shape)
    print("  right_wrist", export[f"{seq}__right_wrist"].shape)
    print("  left_hand  ", export[f"{seq}__left_hand"].shape)
    print("  right_hand ", export[f"{seq}__right_hand"].shape)

np.savez_compressed(OUT, **export)
print("saved:", OUT)