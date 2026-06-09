#!/bin/bash
#SBATCH --account=digital_human_jobs
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --constraint=1080ti
#SBATCH --output=/work/courses/digital_human/team7/UniEgoMotion/logs/merged_%j.out
#SBATCH --error=/work/courses/digital_human/team7/UniEgoMotion/logs/merged_%j.err
#SBATCH --job-name=merged

echo "Starting job on $(hostname)"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate uem

cd /work/courses/digital_human/team7/UniEgoMotion
mkdir -p logs

export PYTHONPATH=/work/courses/digital_human/team7/UniEgoMotion:$PYTHONPATH

python -u /work/courses/digital_human/team7/my_coord_attempt/run_merged.py \
  --example georgiatech_bike_06_10___4314___6195:0 \
    georgiatech_bike_06_10___4314___6195:240 \
    georgiatech_bike_06_10___4314___6195:320 \
    georgiatech_bike_06_10___4314___6195:400 \
    indiana_cooking_09_2___9249___9495:0 \
    indiana_cooking_09_2___10257___11112:160 \
    indiana_cooking_09_2___11658___11901:0 \
    indiana_cooking_09_2___13536___13797:0 \
    indiana_cooking_09_2___15660___16143:80 \
    indiana_cooking_09_2___17802___18048:0 \
    indiana_cooking_09_2___3888___4272:0 \
    georgiatech_bike_06_10___4314___6195:0 \
  --hand_guidance_dir /work/courses/digital_human/team7/cooking_vids_uni/hand_guidance \
  --guidance_2d_dir /work/courses/digital_human/team7/cooking_vids_uni/hand_guidance_2d \
  --no_traj_align \
  --save_eval_npz \
  --run_guidance_opt \
  --run_diffusion_wrist_guidance \
  --run_smpl_opt \
  --w_reproj_2d 5.0 \
  --reproj_2d_min_conf 0.5 \
  --diffusion_wrist_guidance_strength 0.05 \
  --diffusion_wrist_guidance_start_frac 0.7 \
  --w_acc 0.1 \
  --use_projection_error_guidance \
  --projection_error_dir /work/courses/digital_human/team7/my_coord_attempt/projection_errors \
  --proj_3d_good 0.05 \
  --proj_3d_bad 0.20 \
  --proj_px_good 30 \
  --proj_px_bad 150 \
  CONFIG ./config/uem.yaml \
  TRAIN.EXP_PATH ./exp/uem_v4b_dinov2/ \
  MODEL.CKPT_PATH last_ckpt


#    georgiatech_bike_06_10___4314___6195:0 \
#    georgiatech_bike_06_10___4314___6195:240 \
#    georgiatech_bike_06_10___4314___6195:320 \
#    georgiatech_bike_06_10___4314___6195:400 \
#    indiana_cooking_09_2___3888___4272:0 \
#    indiana_cooking_09_2___9249___9495:0 \
#    indiana_cooking_09_2___10257___11112:160 \
#    indiana_cooking_09_2___11658___11901:0 \
#    indiana_cooking_09_2___13536___13797:0 \
#    indiana_cooking_09_2___15660___16143:80 \
#    indiana_cooking_09_2___17802___18048:0 \
