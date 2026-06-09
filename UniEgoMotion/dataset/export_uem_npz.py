import joblib
import numpy as np
import torch
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

item = joblib.load(PKL)

def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

export = {}

safe_key = seq_key.replace("/", "_")

export[f"{safe_key}__aria_traj_T"] = to_np(item["aria_traj_T"])

for k, v in item["smpl_params"].items():
    export[f"{safe_key}__{k}"] = to_np(v)

np.savez_compressed(OUT, **export)

print("saved:", OUT)
print("keys:")
for k, v in export.items():
    print(k, v.shape)