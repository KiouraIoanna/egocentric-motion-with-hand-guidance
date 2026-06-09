#!/bin/bash
#SBATCH --account=digital_human_jobs
#SBATCH --time=12:00:00
#SBATCH --gpus=2080ti:1
#SBATCH --mem=32G
#SBATCH --output=/work/courses/digital_human/team7/UniEgoMotion/logs/cut_dynhamr-upenn_1_2-%j.out
#SBATCH --job-name=cut_dynhamr_indiana

cd /work/courses/digital_human/team7

module load cuda/11.8
source /work/courses/digital_human/team7/Dyn-HaMR/.dynhamr/bin/activate

export LD_LIBRARY_PATH=/work/courses/digital_human/team7/Dyn-HaMR/.dynhamr/lib/python3.10/site-packages/torch/lib:/cluster/data/cuda/x86_64/11.8/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

echo "GPU:"
nvidia-smi

echo "Torch check:"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

echo "Starting cut + Dyn-HaMR pipeline for upenn_0711_Cooking_1_2"

python UniEgoMotion/run/cut_and_run_dynhamr.py \
  --ee_val ee4d_motion_uniegomotion/uniegomotion/ee_val.pt \
  --take upenn_0711_Cooking_1_2 \
  --input_video cooking_vids_uni/videos/upenn_0711_Cooking_1_2.mp4 \
  --root cooking_vids_uni

# Flags:
#   --dry_run    preview clips without running Dyn-HaMR
#   --overwrite  re-cut and re-run clips that already exist
