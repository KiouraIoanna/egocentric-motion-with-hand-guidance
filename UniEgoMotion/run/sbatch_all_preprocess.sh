#!/bin/bash
#SBATCH --job-name=uem_postprocess
#SBATCH --account=digital_human
#SBATCH --time=08:00:00
#SBATCH --gpus=2080ti:1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=32G
#SBATCH --output=/work/scratch/ayopan/uem_postprocess-%j.out
#SBATCH --error=/work/scratch/ayopan/uem_postprocess-%j.err

echo "=================================================="
echo "Starting UniEgoMotion postprocess job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Time: $(date)"
echo "=================================================="

# CUDA
module load cuda/11.8

# Activate environment
source /work/courses/digital_human/team7/Dyn-HaMR/.dynhamr/bin/activate

# Go to project
cd /work/courses/digital_human/team7/UniEgoMotion

echo "Python: $(which python)"
python --version

echo "=================================================="
echo "Running postprocess pipeline"
echo "=================================================="

python run/run_all_uem_postprocess.py

echo "=================================================="
echo "Finished at $(date)"
echo "=================================================="