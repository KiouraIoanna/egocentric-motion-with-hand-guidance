#!/bin/bash
#SBATCH --account=digital_human
#SBATCH --time=08:00:00
#SBATCH --gpus=2080ti:1
#SBATCH --mem=32G
#SBATCH --output=/work/courses/digital_human/team7/UniEgoMotion/logs/ablation_projfix_%j.out
#SBATCH --error=/work/courses/digital_human/team7/UniEgoMotion/logs/ablation_projfix_%j.err
#SBATCH --job-name=projfix_abl
#SBATCH --exclude=studgpu-node21

# Re-test the projection-error-weighted Dyn-HaMR guidance using the CORRECTED
# projection-error files (my_coord_attempt/projection_errors_fixed, produced by
# fix_projection_errors.py). In the original run the weights were 1.0 everywhere
# (offset/intrinsics bugs) so proj-guidance was inert and arm A == arm B. With
# the fix the per-frame weights vary (21% trust UEM, 29% trust Dyn, 50% blend),
# so this comparison finally tests the mechanism.
#
# Arms (everything else = run_merged_test.sh config):
#   AFIX. proj-guidance ON, reading projection_errors_fixed  -> ee4d_vis_projfix_afix
#   B.    proj-guidance OFF (uniform/gated Dyn-HaMR)          -> ee4d_vis_projfix_b
# AFIX vs B = the real value of the (now functional) projection-error weighting.

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
PROJ_FIXED_DIR=$REPO/my_coord_attempt/projection_errors_fixed

SUFFIX_AFIX="_projfix_afix"
SUFFIX_B="_projfix_b"

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

COMMON_ARGS="\
  --hand_guidance_dir $REPO/cooking_vids_uni/hand_guidance \
  --guidance_2d_dir   $REPO/cooking_vids_uni/hand_guidance_2d \
  --no_traj_align \
  --save_eval_npz \
  --run_guidance_opt \
  --run_smpl_opt \
  --run_diffusion_wrist_guidance \
  --w_reproj_2d 5.0 \
  --reproj_2d_min_conf 0.5 \
  --diffusion_wrist_guidance_strength 0.05 \
  --diffusion_wrist_guidance_start_frac 0.7 \
  --w_acc 0.1"

echo "=============================================================="
echo "[ARM AFIX] proj-guidance ON (FIXED npz) -> ee4d_vis${SUFFIX_AFIX}"
echo "=============================================================="
PYTHONPATH=. python $RUNNER \
  --example $EXAMPLES \
  $COMMON_ARGS \
  --use_projection_error_guidance \
  --projection_error_dir $PROJ_FIXED_DIR \
  --proj_3d_good 0.05 --proj_3d_bad 0.20 \
  --proj_px_good 30   --proj_px_bad 150 \
  --run_tag projfixafix \
  CONFIG ./config/uem.yaml \
  TRAIN.EXP_PATH $EXP/ \
  TRAIN.EVAL_SUFFIX $SUFFIX_AFIX \
  MODEL.CKPT_PATH last_ckpt

echo "=============================================================="
echo "[ARM B] proj-guidance OFF -> ee4d_vis${SUFFIX_B}"
echo "=============================================================="
PYTHONPATH=. python $RUNNER \
  --example $EXAMPLES \
  $COMMON_ARGS \
  --run_tag projfixb \
  CONFIG ./config/uem.yaml \
  TRAIN.EXP_PATH $EXP/ \
  TRAIN.EVAL_SUFFIX $SUFFIX_B \
  MODEL.CKPT_PATH last_ckpt

echo "=============================================================="
echo "[REPORT] AFIX (proj-weighting ON, fixed) vs B (off)"
echo "=============================================================="
PYTHONPATH=. python $EVALER \
  --full_dir     $EXP/ee4d_vis${SUFFIX_AFIX} \
  --ablation_dir $EXP/ee4d_vis${SUFFIX_B}
