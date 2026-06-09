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

For UniEgoMotion set up:
1. Create and activate conda environment: 

```
conda create --name uem python=3.10
conda activate uem

# Install Pytorch with your own CUDA version
pip3 install torch --index-url https://download.pytorch.org/whl/cu118

pip3 install pytorch_lightning==2.4.0
pip3 install -r requirements.txt
```

2. Download SMPL-X model here https://smpl-x.is.tue.mpg.de/ and set a proper path in get_smpl function in UniEgoMotion/dataset/smpl_utils.py

3. From DATASET.md follow the second link to download the processed and filtered EE4D-Motion data, DINOv2 features, and other metadata for running UniEgoMotion.

4. Create a new directory called UniEgoMotion/exp/ and follow this link https://huggingface.co/datasets/chaitanya100100/uniegomotion/tree/main to download the pretrained model. Place it in the new directory.



