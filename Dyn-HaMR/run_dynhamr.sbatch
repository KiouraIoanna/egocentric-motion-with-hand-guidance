#!/bin/bash
#SBATCH --account=digital_human
#SBATCH --time=01:00:00
#SBATCH --gpus=2080ti:1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=24G
#SBATCH --output=/work/courses/digital_human/team7/Dyn-HaMR/slurm-%j.out

cd /work/courses/digital_human/team7/Dyn-HaMR

module load cuda/11.8
source .dynhamr/bin/activate

export LD_LIBRARY_PATH=/work/courses/digital_human/team7/Dyn-HaMR/.dynhamr/lib/python3.10/site-packages/torch/lib:/cluster/data/cuda/x86_64/11.8/lib64:$LD_LIBRARY_PATH

echo "Hostname:"
hostname

echo "GPU:"
nvidia-smi

echo "Torch test:"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

echo "Compiled modules test:"
python -c "import droid_backends, lietorch; print('compiled modules ok')"
