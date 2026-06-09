#!/bin/bash
#SBATCH --account=digital_human
#SBATCH --time=01:00:00
#SBATCH --gpus=2080ti:1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=24G
#SBATCH --output=/work/courses/digital_human/team7/UniEgoMotion/logs/diff_hands_%j.out
#SBATCH --error=/work/courses/digital_human/team7/UniEgoMotion/logs/diff_hands_%j.err
#SBATCH --job-name=hand_diff



echo "================================================="
echo "Dyn-HaMR vs UniEgoMotion hand comparison"
echo "================================================="

hostname
date

# -------------------------------------------------
# ACTIVATE ENVIRONMENT
# -------------------------------------------------

source ~/miniconda3/etc/profile.d/conda.sh
conda activate uem

# -------------------------------------------------
# GO TO PROJECT
# -------------------------------------------------

cd /work/courses/digital_human/team7/UniEgoMotion

echo "Working directory:"
pwd

# -------------------------------------------------
# INPUT
# -------------------------------------------------

# CHANGE THIS
EXAMPLE="georgiatech_bike_06_10___2181___2964st80_uem80"

OUT_DIR="/work/scratch/$USER/hand_comparison"

mkdir -p ${OUT_DIR}

# -------------------------------------------------
# RUN
# -------------------------------------------------

python run/compare_dynhamr_uem_hands.py \
    --example indiana_cooking_09_2___10257___11112:160 \
    --out_root ${OUT_DIR}

# -------------------------------------------------
# FINISHED
# -------------------------------------------------

echo "================================================="
echo "Finished"
echo "================================================="

echo "Results:"
ls -lh ${OUT_DIR}

date