#!/bin/bash
#SBATCH --account=digital_human
#SBATCH --output=/work/courses/digital_human/team7/my_coord_attempt/projection_errors_%j.out
#SBATCH --error=/work/courses/digital_human/team7/my_coord_attempt/projection_errors_%j.err

set -euo pipefail

cd /work/courses/digital_human/team7/UniEgoMotion
source ~/miniconda3/etc/profile.d/conda.sh
conda activate uem

OUT_DIR=/work/courses/digital_human/team7/my_coord_attempt/projection_errors
# Eval NPZs saved by run_merged.py --save_eval_npz (abldynno2d = no 2D reproj, clean baseline)
EVAL_DIR=/work/courses/digital_human/team7/UniEgoMotion/exp/uem_v4b_dinov2/ee4d_vis_abldynno2d
EVAL_SUFFIX=abldynno2d

EXAMPLES=(
  georgiatech_bike_06_10___4314___6195:0
  georgiatech_bike_06_10___4314___6195:240
  georgiatech_bike_06_10___4314___6195:320
  georgiatech_bike_06_10___4314___6195:400
  indiana_cooking_09_2___10257___11112:160
  indiana_cooking_09_2___11658___11901:0
  indiana_cooking_09_2___13536___13797:0
  indiana_cooking_09_2___15660___16143:80
  indiana_cooking_09_2___17802___18048:0
  indiana_cooking_09_2___3888___4272:0
  indiana_cooking_09_2___9249___9495:0
)

for EX in "${EXAMPLES[@]}"; do
  SEQ_NAME="${EX%%:*}"
  START_IDX="${EX##*:}"

  EVAL_NPZ="${EVAL_DIR}/${SEQ_NAME}_st${START_IDX}_${EVAL_SUFFIX}_traj_align_eval.npz"
  # Output name must match what run_merged.py constructs in load_projection_error_guidance_weights
  OUT_NPZ="${OUT_DIR}/${SEQ_NAME}st${START_IDX}_uem80_projection_error.npz"
  OUT_CSV="${OUT_DIR}/${SEQ_NAME}st${START_IDX}_uem80_projection_error.csv"

  echo "===================================="
  echo "Example: $EX"
  echo "Eval: $EVAL_NPZ"
  echo "Out:  $OUT_NPZ"

  PYTHONPATH=/work/courses/digital_human/team7/UniEgoMotion:${PYTHONPATH:-} python -u /work/courses/digital_human/team7/UniEgoMotion/run/project_compare_dynhamr_uem_hands.py \
    --eval_npz "$EVAL_NPZ" \
    --out "$OUT_NPZ" \
    --csv "$OUT_CSV"
done