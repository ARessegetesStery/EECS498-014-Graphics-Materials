# HW_NeRF

# Allocate GPU
For A140 GPU:
salloc --cpus-per-task=4 --gpus=1 --mem-per-gpu=44GB --partition=spgpu --time=0-02:00:00 --account=eecs498s014f24_class 

For V100 GPU:
salloc --cpus-per-task=4 --gpus=1 --mem-per-gpu=24GB --partition=gpu --time=0-02:00:00 --account=eecs498s014f24_class 

## Install 
module load python3.9-anaconda
conda create -n nerf python=3.9
conda activate nerf
cd nerf_homework/

conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia

python -m pip install -r requirements.txt

## Execute
python train.py