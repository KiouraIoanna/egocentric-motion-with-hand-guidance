# run/run_all_selected_extracts.py

from pathlib import Path
import subprocess
import sys

ROOT = Path("/work/courses/digital_human/team7/cooking_vids_uni")
EE_VAL = "/work/courses/digital_human/team7/ee4d_motion_uniegomotion/uniegomotion/ee_val.pt"
TAKE = "indiana_cooking_09_2"

kept_file = ROOT / "dynhamr" / f"kept_{TAKE}.txt"
clips = [x.strip() for x in kept_file.read_text().splitlines() if x.strip()]

tasks = [
    ("dataset/export_uem_npz.py", "uem_export.npz"),
    ("dataset/export_uem_joints.py", "uem_joints.npz"),
    ("dataset/export_uem_hands.py", "uem_hands.npz"),
    ("dataset/project_uem_hands.py", "uem_hands_projected_test.npz"),
]

for seq_key in clips:
    out_dir = ROOT / "uem_out" / seq_key
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(seq_key)

    for script, out_name in tasks:
        out_path = out_dir / out_name

        cmd = [
            sys.executable,
            script,
            "--pkl", EE_VAL,
            "--out", str(out_path),
            "--seq_key", seq_key,
        ]

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    compare_cmd = [
        sys.executable,
        "run/compare_wrist.py",
        "--uem_hands", str(out_dir / "uem_hands.npz"),
        "--dynhamr", str(ROOT / "dynhamr" / "hamer_out" / seq_key / "results" / f"demo_{seq_key}.pkl"),
        "--out", str(out_dir / "wrist_difference.npz"),
    ]

    print("Running:", " ".join(compare_cmd))
    subprocess.run(compare_cmd, check=True)