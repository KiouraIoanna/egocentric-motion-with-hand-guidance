# run: python run/cut_and_run_dynhamr.py \
#       --ee_val /work/courses/digital_human/team7/ee4d_motion_uniegomotion/uniegomotion/ee_val.pt \
#       --take indiana_cooking_09_2 \
#       --input_video /work/courses/digital_human/team7/cooking_vids_uni/videos/indiana_cooking_09_2.mp4 \
#       --root /work/courses/digital_human/team7/cooking_vids_uni
#
# NOTE: Run this inside the Dyn-HaMR environment:
#   module load cuda/11.8 && source /work/courses/digital_human/team7/Dyn-HaMR/.dynhamr/bin/activate
#   export LD_LIBRARY_PATH=...

import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2
import torch


DYNHAMR_DIR = Path("/work/courses/digital_human/team7/Dyn-HaMR/dyn-hamr")
WINDOW = 80


def parse_seq_name(seq_name):
    take_name, seq_start, seq_end_part = seq_name.split("___")
    seq_end = seq_end_part.split("st")[0]
    return take_name, int(seq_start), int(seq_end)


def make_seq_name(take_name, seq_start_30, seq_end_30, start_idx):
    return f"{take_name}___{seq_start_30}___{seq_end_30}st{start_idx}_uem{WINDOW}"


def cut_video(input_video, seq_name, start_idx, out_path, out_fps=10.0):
    take_name, seq_start_30, seq_end_30 = parse_seq_name(seq_name)

    frame_indices = [seq_start_30 + 3 * i for i in range(start_idx, start_idx + WINDOW)]

    if frame_indices[-1] > seq_end_30:
        raise ValueError(
            f"Window exceeds seq end: last={frame_indices[-1]}, seq_end={seq_end_30}"
        )

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    written = 0
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Could not read frame {frame_idx}")
        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (w, h))
        writer.write(frame)
        written += 1

    cap.release()
    if writer is not None:
        writer.release()

    print(f"  Cut: {out_path.name} ({written} frames, first={frame_indices[0]}, last={frame_indices[-1]})")


def run_dynhamr(seq_name, root):
    cmd = [
        sys.executable, "run_opt.py",
        "data=video_driod",
        "run_opt=True",
        f"data.seq={seq_name}",
        "is_static=False",
        f"data.root={root}",
    ]
    print(f"  Running Dyn-HaMR: {seq_name}")
    result = subprocess.run(cmd, cwd=str(DYNHAMR_DIR))
    if result.returncode != 0:
        print(f"  WARNING: Dyn-HaMR exited with code {result.returncode} for {seq_name}")
        return False
    return True


def check_all_frames_tracked(root, seq_name):
    import json
    track_preds_dir = Path(root) / "dynhamr" / "track_preds" / seq_name
    if not track_preds_dir.exists():
        return False

    left_frames = set()
    right_frames = set()
    for track_dir in track_preds_dir.iterdir():
        if not track_dir.is_dir():
            continue
        for f in track_dir.glob("*_mano.json"):
            frame_num = int(f.stem.split("_")[0])
            with open(f) as fh:
                is_right = json.load(fh).get("is_right", 0)
            if is_right:
                right_frames.add(frame_num)
            else:
                left_frames.add(frame_num)

    all_frames = set(range(1, WINDOW + 1))
    left_ok = all_frames <= left_frames
    right_ok = all_frames <= right_frames
    left_covered = sum(1 for i in all_frames if i in left_frames)
    right_covered = sum(1 for i in all_frames if i in right_frames)
    return left_ok and right_ok, left_covered, right_covered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ee_val", required=True)
    parser.add_argument("--take", required=True)
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--root", default="/work/courses/digital_human/team7/cooking_vids_uni")
    parser.add_argument("--overwrite", action="store_true", help="Re-cut and re-run even if output exists")
    parser.add_argument("--dry_run", action="store_true", help="Print what would be done without running Dyn-HaMR")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    videos_dir = root / "videos"

    data = torch.load(args.ee_val, map_location="cpu", weights_only=False)

    sequences = []
    for seq_name, seq_data in data.items():
        if not seq_name.startswith(args.take + "___"):
            continue
        take_name, start_30, end_30 = seq_name.split("___")
        start_30, end_30 = int(start_30), int(end_30)
        sequences.append((seq_name, take_name, start_30, end_30))

    if not sequences:
        print(f"No validation sequences found for take: {args.take}")
        return

    print(f"Found {len(sequences)} validation sequence(s) for: {args.take}")

    kept = []
    discarded = []

    for seq_name, take_name, seq_start_30, seq_end_30 in sorted(sequences):
        num_10fps_frames = (seq_end_30 - seq_start_30) // 3
        num_windows = num_10fps_frames // WINDOW

        print(f"\nSequence: {seq_name}")
        print(f"  10fps frames: {num_10fps_frames}, non-overlapping {WINDOW}-frame windows: {num_windows}")

        for w in range(num_windows):
            start_idx = w * WINDOW
            clip_name = make_seq_name(take_name, seq_start_30, seq_end_30, start_idx)
            video_path = videos_dir / f"{clip_name}.mp4"
            track_preds_dir = root / "dynhamr" / "track_preds" / clip_name

            print(f"\n  Window {w} (start_idx={start_idx}): {clip_name}")

            # Step 1: cut the video if needed
            if video_path.exists() and not args.overwrite:
                print(f"  Video already exists, skipping cut.")
            else:
                if args.dry_run:
                    print(f"  [dry_run] Would cut video -> {video_path}")
                else:
                    cut_video(args.input_video, clip_name, start_idx, video_path)

            # Step 2: run Dyn-HaMR if needed
            already_ran = track_preds_dir.exists() and any(track_preds_dir.iterdir())
            if already_ran and not args.overwrite:
                print(f"  Track preds already exist, skipping Dyn-HaMR.")
            else:
                if args.dry_run:
                    print(f"  [dry_run] Would run Dyn-HaMR on {clip_name}")
                else:
                    run_dynhamr(clip_name, root)

            # Step 3: check coverage
            if args.dry_run:
                print(f"  [dry_run] Would check track coverage.")
                continue

            result = check_all_frames_tracked(root, clip_name)
            if result is False:
                print(f"  No track_preds found — discarding.")
                if video_path.exists():
                    video_path.unlink()
                discarded.append(clip_name)
            else:
                all_tracked, left_covered, right_covered = result
                if all_tracked:
                    print(f"  Both hands tracked for all {WINDOW} frames — keeping.")
                    kept.append(clip_name)
                else:
                    print(f"  Incomplete tracking (left={left_covered}/{WINDOW}, right={right_covered}/{WINDOW}) — discarding.")
                    if video_path.exists():
                        video_path.unlink()
                    discarded.append(clip_name)

    if not args.dry_run:
        print(f"\n{'='*60}")
        print(f"Kept:     {len(kept)}")
        for n in kept:
            print(f"  {n}")
        print(f"Discarded: {len(discarded)}")
        for n in discarded:
            print(f"  {n}")


if __name__ == "__main__":
    main()
