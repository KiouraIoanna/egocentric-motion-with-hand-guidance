#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    x = np.load(args.npz, allow_pickle=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for side in ["left", "right"]:
        dyn_uv = x[f"{side}_dyn_uv"]
        uem_uv = x[f"{side}_uem_uv"]
        valid = x[f"{side}_valid"].astype(bool)

        plt.figure(figsize=(10, 6))

        plt.plot(dyn_uv[valid, 0], dyn_uv[valid, 1], "o-", label=f"Dyn-HaMR {side}")
        plt.plot(uem_uv[valid, 0], uem_uv[valid, 1], "o-", label=f"UniEgoMotion {side}")

        # draw frame-to-frame correspondence lines
        for i in np.where(valid)[0]:
            plt.plot(
                [dyn_uv[i, 0], uem_uv[i, 0]],
                [dyn_uv[i, 1], uem_uv[i, 1]],
                "-",
                alpha=0.25,
            )

        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.xlabel("u / image x pixel")
        plt.ylabel("v / image y pixel")
        plt.title(f"Projected {side} wrist trajectory")
        plt.legend()
        plt.grid(True)

        save_path = out.with_name(out.stem + f"_{side}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

        print("saved:", save_path)


if __name__ == "__main__":
    main()