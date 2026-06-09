"""
Extract 2D wrist positions from egocentric MP4 clips using MediaPipe Hand Landmarker.

Output per clip: {seq_key}_2d_wrist_guidance.npz containing:
  left_wrist_2d   (T, 2)  — pixel (u, v) for left wrist
  right_wrist_2d  (T, 2)
  left_valid_2d   (T,)    — bool: MediaPipe detected a hand this frame
  right_valid_2d  (T,)
  left_conf_2d    (T,)    — detection confidence
  right_conf_2d   (T,)
  image_width, image_height

Usage (batch):
  python extract_2d_wrist_guidance.py \
      --video_dir /work/courses/digital_human/team7/cooking_vids_uni/videos \
      --out_dir   /work/courses/digital_human/team7/cooking_vids_uni/hand_guidance_2d

Usage (single clip):
  python extract_2d_wrist_guidance.py \
      --video /path/to/seq_key_uem80.mp4
"""

import argparse
import os
import glob
import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

DEFAULT_MODEL = "/tmp/hand_landmarker.task"
DEFAULT_OUT_DIR = "/work/courses/digital_human/team7/cooking_vids_uni/hand_guidance_2d"


def build_landmarker(model_path, min_det_conf=0.4, min_track_conf=0.4, num_hands=2):
    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=min_det_conf,
        min_hand_presence_confidence=min_det_conf,
        min_tracking_confidence=min_track_conf,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def detect_hands_in_video(video_path, model_path, min_det_conf=0.4, min_track_conf=0.4):
    """
    Run MediaPipe HandLandmarker on every frame of the video.
    Returns arrays of shape (T,2) for pixel coords and (T,) for validity/conf.

    Egocentric note: the Aria camera faces outward so the wearer's right hand
    appears on the right side of the image — MediaPipe handedness is correct.
    If two detections claim the same side, keep the higher-confidence one.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    left_wrist_2d  = np.zeros((n_frames, 2), dtype=np.float32)
    right_wrist_2d = np.zeros((n_frames, 2), dtype=np.float32)
    left_valid     = np.zeros(n_frames, dtype=bool)
    right_valid    = np.zeros(n_frames, dtype=bool)
    left_conf      = np.zeros(n_frames, dtype=np.float32)
    right_conf     = np.zeros(n_frames, dtype=np.float32)

    # frame timestamp in ms (needed for VIDEO mode)
    frame_ms = 0
    frame_interval_ms = int(1000.0 / fps) if fps > 0 else 100

    with build_landmarker(model_path, min_det_conf, min_track_conf) as landmarker:
        for fi in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect_for_video(mp_image, frame_ms)
            frame_ms += frame_interval_ms

            if not result.hand_landmarks:
                continue

            for landmarks, handedness_list in zip(result.hand_landmarks,
                                                   result.handedness):
                label = handedness_list[0].category_name   # "Left" or "Right"
                conf  = handedness_list[0].score

                # landmark 0 = WRIST (normalised [0,1])
                wrist = landmarks[0]
                u = wrist.x * W
                v = wrist.y * H

                if label == "Left":
                    if conf > left_conf[fi]:
                        left_wrist_2d[fi] = [u, v]
                        left_valid[fi]    = True
                        left_conf[fi]     = conf
                else:
                    if conf > right_conf[fi]:
                        right_wrist_2d[fi] = [u, v]
                        right_valid[fi]    = True
                        right_conf[fi]     = conf

    cap.release()

    print(f"  frames={n_frames}  {W}x{H}  fps={fps}")
    if left_valid.any():
        print(f"  left  valid={left_valid.sum()}/{n_frames}  "
              f"mean_conf={left_conf[left_valid].mean():.3f}")
    else:
        print(f"  left  valid=0/{n_frames}")
    if right_valid.any():
        print(f"  right valid={right_valid.sum()}/{n_frames}  "
              f"mean_conf={right_conf[right_valid].mean():.3f}")
    else:
        print(f"  right valid=0/{n_frames}")

    return left_wrist_2d, right_wrist_2d, left_valid, right_valid, left_conf, right_conf, W, H


def process_video(video_path, out_dir, model_path, force=False,
                  min_det_conf=0.4, min_track_conf=0.4):
    seq_key  = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{seq_key}_2d_wrist_guidance.npz")

    if os.path.exists(out_path) and not force:
        print(f"[SKIP] {seq_key}")
        return out_path

    print(f"\n[PROC] {seq_key}")
    lw2d, rw2d, lv, rv, lc, rc, W, H = detect_hands_in_video(
        video_path, model_path, min_det_conf, min_track_conf)

    np.savez_compressed(
        out_path,
        seq_key=seq_key,
        left_wrist_2d=lw2d,   right_wrist_2d=rw2d,
        left_valid_2d=lv,     right_valid_2d=rv,
        left_conf_2d=lc,      right_conf_2d=rc,
        image_width=W,         image_height=H,
    )
    print(f"  saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default=None,
                        help="Directory containing *_uem80.mp4 clips (batch mode).")
    parser.add_argument("--video", default=None,
                        help="Single MP4 (single mode).")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Path to hand_landmarker.task model file.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min_det_conf",   type=float, default=0.4)
    parser.add_argument("--min_track_conf", type=float, default=0.4)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        print("Download with:")
        print("  wget -O /tmp/hand_landmarker.task "
              "https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    if args.video:
        process_video(args.video, args.out_dir, args.model,
                      force=args.force,
                      min_det_conf=args.min_det_conf,
                      min_track_conf=args.min_track_conf)
    elif args.video_dir:
        videos = sorted(glob.glob(os.path.join(args.video_dir, "*_uem80.mp4")))
        if not videos:
            print(f"[WARN] No *_uem80.mp4 in {args.video_dir}")
            return
        print(f"Found {len(videos)} clips.")
        for v in videos:
            process_video(v, args.out_dir, args.model,
                          force=args.force,
                          min_det_conf=args.min_det_conf,
                          min_track_conf=args.min_track_conf)
    else:
        parser.error("Provide --video or --video_dir.")

    print("\nDone.")


if __name__ == "__main__":
    main()
