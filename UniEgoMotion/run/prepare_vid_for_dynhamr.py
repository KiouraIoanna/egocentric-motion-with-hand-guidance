import argparse
import os
import cv2


def parse_seq_name(seq_name):
    take_name, seq_start, seq_end = seq_name.split("___")
    return take_name, int(seq_start), int(seq_end)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--seq_name", required=True)
    parser.add_argument("--start_idx", type=int, required=True)
    parser.add_argument("--window", type=int, default=80)
    parser.add_argument("--out", required=True)
    parser.add_argument("--out_fps", type=float, default=10.0)
    args = parser.parse_args()

    take_name, seq_start_30, seq_end_30 = parse_seq_name(args.seq_name)

    frame_indices = [
        seq_start_30 + 3 * i
        for i in range(args.start_idx, args.start_idx + args.window)
    ]

    if frame_indices[-1] > seq_end_30:
        raise ValueError(
            f"Requested window exceeds seq end: "
            f"last={frame_indices[-1]}, seq_end={seq_end_30}"
        )

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input_video}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    writer = None
    written = 0

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()

        if not ok:
            raise RuntimeError(f"Could not read frame {frame_idx}")

        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.out, fourcc, args.out_fps, (w, h))

        writer.write(frame)
        written += 1

    cap.release()

    if writer is not None:
        writer.release()

    print("Take:", take_name)
    print("Input video:", args.input_video)
    print("Output:", args.out)
    print("First original frame:", frame_indices[0])
    print("Last original frame:", frame_indices[-1])
    print("Written frames:", written)
    print("Output fps:", args.out_fps)


if __name__ == "__main__":
    main()