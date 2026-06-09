# Egocentric Motion Reconstruction with Hand Guidance

Improving full-body egocentric motion reconstruction by guiding
[UniEgoMotion](https://github.com/...) with hand information from
[Dyn-HaMR](https://github.com/ZhengdiYu/Dyn-HaMR) and MediaPipe.

Course project for the Digital Humans course. We take UniEgoMotion — a
diffusion model that reconstructs SMPL-X body motion from egocentric (Aria)
video — and inject hand cues at two stages: as guidance inside the diffusion
sampling loop, and as a post-inference optimization that fuses the cues,
re-poses the arms, and transfers finger articulation. Every cue is modulated by
a per-frame reliability estimate so the body follows the hands only where the
hand evidence is dependable.

## Hand sources

- **Dyn-HaMR** — 3D hand pose and wrist trajectories (3D guidance + MANO finger transfer).
- **MediaPipe Hands** — 2D wrist pixel detections (2D reprojection guidance).

## Repository layout
my_coord_attempt/        Our method, ablations, and analysis (main contribution)
UniEgoMotion/            UniEgoMotion model + our run/eval scripts (UniEgoMotion/run)
Dyn-HaMR/dyn-hamr/       Dyn-HaMR pipeline source (third-party libs gitignored)
utils/                   Shared helpers



Large data, model weights, datasets, rendered videos, and vendored third-party
libraries (`Dyn-HaMR/third-party`, `Dyn-HaMR/test`) are excluded via
`.gitignore` and must be obtained separately.

## Key scripts

| Script | Purpose |
|---|---|
| `my_coord_attempt/run_merged.py` | Main pipeline: UniEgoMotion + Dyn-HaMR/MediaPipe guidance + post-inference SMPL-X arm fit |
| `my_coord_attempt/run_merged.sbatch` | SLURM job to run the full pipeline on a clip set |
| `my_coord_attempt/run_merged_ablation.py` | Pipeline variant exposing ablation flags + decomposed (root-relative / Procrustes) wrist metrics |
| `my_coord_attempt/run_ablation_dyn_hamr.sbatch` | Multi-arm ablation (Dyn-HaMR 3D, MediaPipe 2D, finger pose) |
| `my_coord_attempt/eval_root_relative.py` | Post-hoc world / root-relative / Procrustes wrist-error report |
| `my_coord_attempt/fix_projection_errors.py` | Recompute projection-error weights with offset/intrinsic corrections |
| `UniEgoMotion/run/cut_and_run_dynhamr.py` | Extract 80-frame clips from Ego-Exo4D takes and run Dyn-HaMR |
| `UniEgoMotion/run/extract_img_feats.py` | DINOv2 image features for UniEgoMotion conditioning |

## Method summary

1. **Hand extraction** — Dyn-HaMR (3D) and MediaPipe (2D) on the Aria frames.
2. **Alignment** — map Dyn-HaMR's monocular wrists into the UniEgoMotion world frame via a per-clip similarity transform.
3. **Reliability weighting** — per-frame trust from velocity/acceleration/separation consistency (and optional projection-error weighting).
4. **In-diffusion guidance** — blend the predicted wrist channels toward Dyn-HaMR targets on late denoising steps.
5. **Post-inference fusion** — optimize fused wrist targets (Dyn-HaMR + smoothness + two-hand geometry).
6. **Arm re-posing + finger transfer** — fit SMPL-X arm rotations to the fused 3D targets and the MediaPipe 2D detections, then replace UniEgoMotion's coarse hand pose with Dyn-HaMR's MANO fingers.

## Setup

This project relies on the UniEgoMotion and Dyn-HaMR environments and external
assets (SMPL-X body model, Ego-Exo4D videos, pretrained checkpoints) that are
not included in the repository. 

### **UniEgoMotion** set up:
1. Run the following: 

```
conda create --name uem python=3.10
conda activate uem

# Install Pytorch with your own CUDA version
pip3 install torch --index-url https://download.pytorch.org/whl/cu118

pip3 install pytorch_lightning==2.4.0
pip3 install -r requirements.txt
```

We had some issues with the blendify package, which we ended up installing on its own, using a more recent version. If you have similar issues, run ```pip3 install -r requirements_no_blendify.txt``` and install a later version of blendify instead.

2. Download SMPL-X model here https://smpl-x.is.tue.mpg.de/ and set a proper path in get_smpl function in UniEgoMotion/dataset/smpl_utils.py

3. From DATASET.md follow the second link to download the processed and filtered EE4D-Motion data, DINOv2 features, and other metadata for running UniEgoMotion.

4. Create a new directory called UniEgoMotion/exp/ and follow this link https://huggingface.co/datasets/chaitanya100100/uniegomotion/tree/main to download the pretrained model. Place it in the new directory.

### MediaPipe (2D wrist detection) set up

MediaPipe is installed via pip into the same `uem` conda environment used by
UniEgoMotion:

```
conda activate uem
pip install mediapipe==0.10.35
```

We use the **MediaPipe Tasks** Hand Landmarker (`mediapipe.tasks.python.vision`).
The model bundle is downloaded separately from Google's model storage:

```
wget -O /tmp/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```


2D wrist detections are then extracted with
`my_coord_attempt/extract_2d_wrist_guidance.py`, keeping only the wrist landmark
(landmark 0) per hand. Tested with `mediapipe==0.10.35` on Python 3.10.

### **Dyn-HaMR** set up:

main link: https://github.com/ZhengdiYu/Dyn-HaMR

Inside Dyn-HaMR create the evironement with the \scripts: 

```
source install_pip.sh
```
or from conda: 
```
source install_conda.sh
```

### The **dataset**:

We used videos from the UniEgoMotion validation dataset. These can be obtained by gaining access to the EgoExo4D dataset (https://ego-exo4d-data.org). Once credentials are acquired, you can run the following command in the root folder to see the names of all videos in the validation dataset:

```
python3 -c "import torch; print('\n'.join(sorted({k.split('___')[0] for k in torch.load('ee4d_motion_uniegomotion/uniegomotion/ee_val.pt', map_location='cpu', weights_only=False)})))"
```

You can pick the videos you want. Afterwards, you need to install the CLI downloader:

```pip install ego4d```

And configure your credentials with:

```
pip install awscli
aws configure
```

Now you need to find the uids of your selected videos:

```
python - <<'PY'   
import json                                                      
p="/path/to/takes.json"

with open(p, "r") as f:
    takes = json.load(f)

for t in takes:
    if t["take_name"] == <YOUR_VIDEO_NAME>:  
        print("TAKE UID:", t["take_uid"])
        print("ROOT DIR:", t["root_dir"])
        print("DURATION:", t.get("duration_sec"))
        print("BEST EXO:", t.get("best_exo"))
        print()
PY 
```

And finally you can download the egocentric video as follows:

```
egoexo \          
  -o <YOU_OUTPUT_DIR> \
  --parts takes \
  --uids <YOUR_VIDEO_UID> \
  --views ego
  ```

This will download a folder of videos. The one we use is in the frame_aligned_videos/ folder and will be called aria01_214-1.mp4. You will need to create a new directory in the repo root titled cooking_vids_uni/videos/ to place this.


## Data Preprocessing
The videos in the validation dataset of UniEgoMotion are not contained as a whole. Specific ranges of frames have been preserved, while others have been deemed inadequate. To inspect for your selected videos which frames have been selected, you can run in the UniEgoMotion folder:

```
python run/inspect_val_frames.py \
       --ee_val /path/to/ee4d_motion_uniegomotion/uniegomotion/ee_val.pt \
       --take <YOUR_VIDEO_NAME>
```

We must only use these specific ranges when running our experiments. UniEgoMotion also runs on 80-frame clips, so we need to divide these ranges in 80-frame clips as well. Then we need to run Dyn-HaMR on them. This is what the cut_and_run_dynhamr.py script does, while also checking which videos have 80 valid Dyn-HaMR frames for both hands. The videos kept in the cooking_vids_uni/videos folder are the ones with 80 valid frames for both hands. Dyn-HaMR results however are kept for all checked clips regardless of valid frames. To run for your video, edit the name in cut_and_run_dynhamr.sbatch and run from the repo root:

```sbatch Uniegomotion/run/cut_and_run_dynhamr.sbatch```

## Running our implementation

### Guidance with MediaPipe

#First run MediaPipe on the desired preprocessed video as follows

You can then use the sbatch file my_coord_attempt/run_guided_mediapipe.sh chosing your vidoe after the flag --example. It must have a format like: georgiatech_bike_06_10___4314___6195:240

### Guidance with Dyn-HaMR and refinement with MediaPipe

Run the DynHaMR on the desired video

```
python run_opt.py \
  data=video_driod \
  run_opt=True \
  data.seq=<clip_name> \
  is_static=False \
  data.root=/work/courses/digital_human/team7/cooking_vids_uni
```

clip name has a similar format to "indiana_cooking_09_2___4314___6195st0_uem80"

You can then use the sbatch file my_coord_attempt/run_merged_test.sh chosing your vidoe after the flag --example. It must have a format like: georgiatech_bike_06_10___4314___6195:240

