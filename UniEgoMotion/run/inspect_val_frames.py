"""
save as: inspect_val_frames.py
run : python run/inspect_val_frames.py \
       --ee_val /work/courses/digital_human/team7/ee4d_motion_uniegomotion/uniegomotion/ee_val.pt \
       --take indiana_cooking_09_2
"""

import argparse
import torch


def parse_seq_name(seq_name):
    take_name, start, end = seq_name.split("___")
    return take_name, int(start), int(end)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ee_val", required=True)
    parser.add_argument("--take", required=True)
    args = parser.parse_args()

    data = torch.load(args.ee_val, map_location="cpu", weights_only=False)

    matches = []
    for seq_name, seq_data in data.items():
        if not seq_name.startswith(args.take + "___"):
            continue

        take_name, start_30, end_30 = parse_seq_name(seq_name)

        num_frames = seq_data.get("num_frames", None)
        if num_frames is None and "aria_traj" in seq_data:
            num_frames = len(seq_data["aria_traj"])

        matches.append((seq_name, start_30, end_30, num_frames))

    if not matches:
        print(f"No validation sequences found for take: {args.take}")
        return

    print(f"Available validation sequences for: {args.take}")
    print()

    for seq_name, start_30, end_30, num_frames in sorted(matches):
        print(seq_name)
        print(f"  original 30fps frames: {start_30} -> {end_30}")
        print(f"  available 10fps frames: {start_30}, {start_30 + 3}, {start_30 + 6}, ... up to {end_30}")
        print(f"  processed 10fps length: {num_frames}")
        print()


if __name__ == "__main__":
    main()