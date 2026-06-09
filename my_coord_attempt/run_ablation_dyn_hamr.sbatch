#!/bin/bash
#SBATCH --account=digital_human
#SBATCH --time=12:00:00
#SBATCH --gpus=2080ti:1
#SBATCH --mem=32G
#SBATCH --output=/work/courses/digital_human/team7/UniEgoMotion/logs/ablation_%j.out
#SBATCH --error=/work/courses/digital_human/team7/UniEgoMotion/logs/ablation_%j.err
#SBATCH --job-name=dyn_ablation
#SBATCH --exclude=studgpu-node21

# Dyn-HaMR ablation, corrected to use the SAME config as run_merged_test.sh
# (the run that produced the good results) as the FULL baseline. Key point vs
# the earlier attempt: the FULL arm enables --use_projection_error_guidance,
# which applies Dyn-HaMR's 3D wrist only on the per-frame indices where UEM is
# wrong (precomputed projection errors). Dyn-HaMR is noisy on average but better
# than UEM on its bad frames; the projection-error weighting is what extracts
# that. Ablating without it (the previous run) tested the wrong configuration.
#
# Arms:
#   A. full      : run_merged_test.sh config (proj-error-weighted Dyn-HaMR 3D +
#                  finger + diffusion + MediaPipe 2D reproj).
#   B. noproj    : full MINUS --use_projection_error_guidance. Dyn-HaMR 3D falls
#                  back to uniform + reliability-gated. A vs B = value of the
#                  per-frame projection-error weighting itself.
#   C. nodyn     : --no_dyn_hamr (w_dyn=0, no diffusion guidance, no finger pose),
#                  proj flags off. MediaPipe 2D reproj STILL ON. A vs C = value of
#                  Dyn-HaMR on top of the 2D signal.
#   D. uemonly   : --no_dyn_hamr AND --w_reproj_2d 0. No Dyn-HaMR and no MediaPipe
#                  2D either -> raw UEM + smpl_opt pose prior/smoothing only.
#                  A vs D = value of ALL hand guidance vs nothing; C vs D = value
#                  of the MediaPipe 2D reproj alone.
#   E. dynno2d   : full Dyn-HaMR (proj-error-weighted 3D + finger + diffusion) but
#                  --w_reproj_2d 0. Dyn-HaMR with MediaPipe 2D removed. E vs D =
#                  value of Dyn-HaMR ALONE (no 2D); A vs E = value of MediaPipe 2D
#                  on top of Dyn-HaMR.
#
# Each arm writes *_traj_align_eval.npz into a separate dir (via EVAL_SUFFIX) and
# tags filenames (via --run_tag). eval_root_relative.py then prints world /
# root-relative / Procrustes wrist MPJPE for A and C plus their delta.

echo "Starting job on $(hostname)"
nvidia-smi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate uem

cd /work/courses/digital_human/team7/UniEgoMotion
mkdir -p logs

REPO=/work/courses/digital_human/team7
RUNNER=$REPO/my_coord_attempt/run_merged_ablation.py
EVALER=$REPO/my_coord_attempt/eval_root_relative.py
EXP=./exp/uem_v4b_dinov2

SUFFIX_FULL="_ablfull"
SUFFIX_NOPROJ="_ablnoproj"
SUFFIX_NODYN="_ablnodyn"
SUFFIX_UEMONLY="_abluemonly"
SUFFIX_DYNNO2D="_abldynno2d"

EXAMPLES="\
  georgiatech_bike_06_10___4314___6195:0 \
  georgiatech_bike_06_10___4314___6195:240 \
  georgiatech_bike_06_10___4314___6195:320 \
  georgiatech_bike_06_10___4314___6195:400 \
  indiana_cooking_09_2___10257___11112:160 \
  indiana_cooking_09_2___11658___11901:0 \
  indiana_cooking_09_2___13536___13797:0 \
  indiana_cooking_09_2___15660___16143:80 \
  indiana_cooking_09_2___17802___18048:0 \
  indiana_cooking_09_2___3888___4272:0 \
  indiana_cooking_09_2___9249___9495:0"

# Shared args common to all three arms (mirrors run_merged_test.sh).
COMMON_ARGS="\
  --hand_guidance_dir $REPO/cooking_vids_uni/hand_guidance \
  --guidance_2d_dir   $REPO/cooking_vids_uni/hand_guidance_2d \
  --no_traj_align \
  --save_eval_npz \
  --run_guidance_opt \
  --run_smpl_opt \
  --w_reproj_2d 5.0 \
  --reproj_2d_min_conf 0.5 \
  --diffusion_wrist_guidance_strength 0.05 \
  --diffusion_wrist_guidance_start_frac 0.7 \
  --w_acc 0.1"

# Projection-error guidance flags (exactly as run_merged_test.sh).
PROJ_ARGS="\
  --use_projection_error_guidance \
  --projection_error_dir $REPO/my_coord_attempt/projection_errors \
  --proj_3d_good 0.05 \
  --proj_3d_bad 0.20 \
  --proj_px_good 30 \
  --proj_px_bad 150"

echo "=============================================================="
echo "[ARM A] FULL = run_merged_test.sh config -> ee4d_vis${SUFFIX_FULL}"
echo "=============================================================="
PYTHONPATH=. python $RUNNER \
  --example $EXAMPLES \
  $COMMON_ARGS \
  --run_diffusion_wrist_guidance \
  $PROJ_ARGS \
  --run_tag ablfull \
  CONFIG ./config/uem.yaml \
  TRAIN.EXP_PATH $EXP/ \
  TRAIN.EVAL_SUFFIX $SUFFIX_FULL \
  MODEL.CKPT_PATH last_ckpt

echo "=============================================================="
echo "[ARM B] NO PROJ-ERROR WEIGHTING -> ee4d_vis${SUFFIX_NOPROJ}"
echo "=============================================================="
PYTHONPATH=. python $RUNNER \
  --example $EXAMPLES \
  $COMMON_ARGS \
  --run_diffusion_wrist_guidance \
  --run_tag ablnoproj \
  CONFIG ./config/uem.yaml \
  TRAIN.EXP_PATH $EXP/ \
  TRAIN.EVAL_SUFFIX $SUFFIX_NOPROJ \
  MODEL.CKPT_PATH last_ckpt

echo "=============================================================="
echo "[ARM C] NO DYN-HAMR (2D reproj only) -> ee4d_vis${SUFFIX_NODYN}"
echo "=============================================================="
PYTHONPATH=. python $RUNNER \
  --example $EXAMPLES \
  $COMMON_ARGS \
  --no_dyn_hamr \
  --run_tag ablnodyn \
  CONFIG ./config/uem.yaml \
  TRAIN.EXP_PATH $EXP/ \
  TRAIN.EVAL_SUFFIX $SUFFIX_NODYN \
  MODEL.CKPT_PATH last_ckpt

echo "=============================================================="
echo "[ARM D] UEM ONLY (no Dyn-HaMR, no MediaPipe 2D) -> ee4d_vis${SUFFIX_UEMONLY}"
echo "=============================================================="
PYTHONPATH=. python $RUNNER \
  --example $EXAMPLES \
  $COMMON_ARGS \
  --no_dyn_hamr \
  --w_reproj_2d 0 \
  --run_tag abluemonly \
  CONFIG ./config/uem.yaml \
  TRAIN.EXP_PATH $EXP/ \
  TRAIN.EVAL_SUFFIX $SUFFIX_UEMONLY \
  MODEL.CKPT_PATH last_ckpt

echo "=============================================================="
echo "[ARM E] DYN-HAMR, NO MediaPipe 2D -> ee4d_vis${SUFFIX_DYNNO2D}"
echo "=============================================================="
PYTHONPATH=. python $RUNNER \
  --example $EXAMPLES \
  $COMMON_ARGS \
  --run_diffusion_wrist_guidance \
  $PROJ_ARGS \
  --w_reproj_2d 0 \
  --run_tag abldynno2d \
  CONFIG ./config/uem.yaml \
  TRAIN.EXP_PATH $EXP/ \
  TRAIN.EVAL_SUFFIX $SUFFIX_DYNNO2D \
  MODEL.CKPT_PATH last_ckpt

echo "=============================================================="
echo "[REPORT 1] FULL (A) vs NO-DYN-HAMR (C): total Dyn-HaMR value"
echo "=============================================================="
PYTHONPATH=. python $EVALER \
  --full_dir     $EXP/ee4d_vis${SUFFIX_FULL} \
  --ablation_dir $EXP/ee4d_vis${SUFFIX_NODYN}

echo "=============================================================="
echo "[REPORT 2] FULL (A) vs NO-PROJ-WEIGHTING (B): value of weighting"
echo "=============================================================="
PYTHONPATH=. python $EVALER \
  --full_dir     $EXP/ee4d_vis${SUFFIX_FULL} \
  --ablation_dir $EXP/ee4d_vis${SUFFIX_NOPROJ}

echo "=============================================================="
echo "[REPORT 3] FULL (A) vs UEM-ONLY (D): value of ALL hand guidance"
echo "=============================================================="
PYTHONPATH=. python $EVALER \
  --full_dir     $EXP/ee4d_vis${SUFFIX_FULL} \
  --ablation_dir $EXP/ee4d_vis${SUFFIX_UEMONLY}

echo "=============================================================="
echo "[REPORT 4] NO-DYN-HAMR (C) vs UEM-ONLY (D): value of MediaPipe 2D alone"
echo "=============================================================="
PYTHONPATH=. python $EVALER \
  --full_dir     $EXP/ee4d_vis${SUFFIX_NODYN} \
  --ablation_dir $EXP/ee4d_vis${SUFFIX_UEMONLY}

echo "=============================================================="
echo "[REPORT 5] DYN-NO-2D (E) vs UEM-ONLY (D): value of Dyn-HaMR ALONE"
echo "=============================================================="
PYTHONPATH=. python $EVALER \
  --full_dir     $EXP/ee4d_vis${SUFFIX_DYNNO2D} \
  --ablation_dir $EXP/ee4d_vis${SUFFIX_UEMONLY}

echo "=============================================================="
echo "[REPORT 6] FULL (A) vs DYN-NO-2D (E): value of MediaPipe 2D on top of Dyn-HaMR"
echo "=============================================================="
PYTHONPATH=. python $EVALER \
  --full_dir     $EXP/ee4d_vis${SUFFIX_FULL} \
  --ablation_dir $EXP/ee4d_vis${SUFFIX_DYNNO2D}
