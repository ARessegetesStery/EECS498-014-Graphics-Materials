# HW_Gaussian

## Execution
python train.py --single_image_fitting
python train.py


## GS package
git clone https://github.com/nerfstudio-project/gsplat.git --recursive
cd gsplat
module load cuda/12.1.
module load gcc/10.3.0
python -m pip install -e .
python -m pip install -r examples/requirements.txt
python examples/datasets/download_dataset.py

## SSH Forwarding
- On the local machine, run:
```
ssh -L {local_port}:localhost:{server_port} {your unique name}@greatlakes.arc-ts.umich.edu
```
- On the login node, run:
```
ssh -L {server_port}:localhost:{server_port_compute} {node_address} 
```

- On the Computing Node, run:
```
cd examples/

python python simple_trainer.py default \
    --data_dir data/360_v2/garden/ --data_factor 4 \
    --result_dir ./results/garden--port {server_port_compute} --max_steps 20000
```