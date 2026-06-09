import numpy as np
import torch
from smpl_utils import get_smpl, evaluate_smpl
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

d = np.load(INP)
smpl = get_smpl()

export = {}

seqs = sorted(set(k.rsplit("__", 1)[0] for k in d.files))

for seq in seqs:
    smpl_params = {
        "global_orient": torch.tensor(d[f"{seq}__global_orient"]).float(),
        "body_pose": torch.tensor(d[f"{seq}__body_pose"]).float(),
        "betas": torch.tensor(d[f"{seq}__betas"]).float(),
        "left_hand_pose": torch.tensor(d[f"{seq}__left_hand_pose"]).float(),
        "right_hand_pose": torch.tensor(d[f"{seq}__right_hand_pose"]).float(),
        "transl": torch.tensor(d[f"{seq}__transl"]).float(),
    }

    joints, verts, rots = evaluate_smpl(smpl, smpl_params)

    export[f"{seq}__joints"] = joints.detach().cpu().numpy()
    export[f"{seq}__verts"] = verts.detach().cpu().numpy()
    export[f"{seq}__rots"] = rots.detach().cpu().numpy()
    export[f"{seq}__aria_traj_T"] = d[f"{seq}__aria_traj_T"]

    print(seq, joints.shape, verts.shape, rots.shape)

np.savez_compressed(OUT, **export)
print("saved:", OUT)