# CG_GAN_hw


## Installation
module load python3.9-anaconda

conda create -n GAN python=3.9

conda activate GAN

pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

<!-- pip install --force-reinstall charset-normalizer==3.1.0

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch -->


## GPU resources:
salloc --cpus-per-task=4 --gpus=1 --mem-per-gpu=44GB --partition=spgpu --time=0-02:00:00 --account=eecs498s014f24_class 

squeue --account=eecs498s014f24_class


## Dataset download
python dataset_download.py


## Checkpoint Download
wget https://github.com/um-graphics/um-graphics.github.io/releases/download/HW5/checkpoint.zip

unzip checkpoint.zip

rm checkpoint.zip       # Save space


## Execute
python train.py --outdir training-runs --data many_shot_dog --use_r1_regularization False --use_diffaug False --checkpoint_path checkpoint/part1-10K.pkl

python train.py --outdir training-runs --data many_shot_dog --use_r1_regularization True --use_diffaug False --checkpoint_path checkpoint/part2-10K.pkl

python train.py --outdir training-runs --data many_shot_dog --use_r1_regularization True --use_diffaug True --checkpoint_path checkpoint/part3-10K.pkl
