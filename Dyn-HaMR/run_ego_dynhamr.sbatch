#!/bin/bash
#SBATCH --account=digital_human
#SBATCH --time=02:00:00
#SBATCH --gpus=2080ti:1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=24G
#SBATCH --output=/work/courses/digital_human/team7/Dyn-HaMR/demo-%j.out
#SBATCH --job-name=ego_dynhamr

cd /work/courses/digital_human/team7/Dyn-HaMR

module load cuda/11.8
source .dynhamr/bin/activate

export LD_LIBRARY_PATH=/work/courses/digital_human/team7/Dyn-HaMR/.dynhamr/lib/python3.10/site-packages/torch/lib:/cluster/data/cuda/x86_64/11.8/lib64:$LD_LIBRARY_PATH

echo "GPU check"
nvidia-smi

echo "Torch check"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "Compiled modules check"
python -c "import droid_backends, lietorch; print('compiled modules ok')"

echo "MANO check"
python -c "import mano; print('mano ok')"

echo "Starting Dyn-HaMR demo"
cd dyn-hamr

python run_opt.py \
  data=video_driod \
  run_opt=True \
  data.seq=indiana_cooking_09_2___8145___9156st80_uem80 \
  is_static=False \
  data.root=/work/courses/digital_human/team7/cooking_vids_uni


