# Diffusion_HW

## Install
conda create -n diffusion python=3.10

conda activate diffusion

pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt


## GPU resources:
salloc --cpus-per-task=4 --gpus=1 --mem-per-gpu=44GB --partition=spgpu --time=0-02:00:00 --account=eecs498s014f24_class 

squeue --account=eecs498s014f24_class


## Execution
python execute_DDPM_DDIM.py --scheduler_name DDPM

python execute_DDPM_DDIM.py --scheduler_name DDIM

python execute_SDS_VSD.py --generation_mode=sds --store_dir SDS_results

python execute_SDS_VSD.py --generation_mode=sds --store_dir SDS_results --guidance_scale=100

python execute_SDS_VSD.py --generation_mode=vsd --store_dir VSD_results
