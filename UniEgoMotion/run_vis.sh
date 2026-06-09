#!/bin/bash
#SBATCH --job-name=uem_vis
#SBATCH --account=digital_human
#SBATCH --constraint=2080ti
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/uem_vis-%j.out
#SBATCH --error=logs/uem_vis-%j.err

cd /work/courses/digital_human/team7/UniEgoMotion

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /work/scratch/ayopan/envs/uem

nvidia-smi

export PYTHONPATH=.
export PYOPENGL_PLATFORM=egl

python run/vis_uem.py \
CONFIG ./config/uem.yaml \
TRAIN.EXP_PATH ./exp/uem_v4b_dinov2/ \
MODEL.CKPT_PATH last_ckpt