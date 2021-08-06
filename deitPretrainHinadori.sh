#!/bin/bash
#YBATCH -r am_8
#SBATCH -N 1
#SBATCH -J PreTFrac1k
#SBATCH --time=120:00:00
#SBATCH --output output/%j-PreTFrac1k.out

## Pyenv loading
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

## Modules
. /etc/profile.d/modules.sh
module purge
module load openmpi/3.1.6 cuda/11.1 cudnn/cuda-11.1/8.0

export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"

#wandb disabled

wandb enabled

# Put a comment to avoid vi bash format reading
echo "######################### START ########################################"
cat deitPretrainHinadori.sh

############################ Benchmark with 8-GPUs on IMNET
############################ WANDB
#python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="192.168.205.10" --master_port=1234 --use_env main.py  --model deit_small_patch16_224 --batch-size 256 --data-path /mnt/nfs/datasets/ILSVRC2012  --data-set IMNET --dist_url env://mnt/nfs/datasets/ILSVRC2012/

############# Fractal 1k Kataoka
#### Color
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /mnt/nfs/datasets/FractalDB/FractalDB-1k-Color --data-set FRACTAL1k --batch-size 544 --output_dir preTrains/preTrain_Tiny_244_Fractal1k/ --train-only --lr=3.0e-4 --warmup-epochs=10

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_384 --input-size 384 --data-path /mnt/nfs/datasets/FractalDB/FractalDB-1k-Color --data-set FRACTAL1k --batch-size 32 --output_dir çpreTrains/preTrain_Tiny_244_Fractal1k/ --train-only --lr=3.0e-4 --warmup-epochs=10

#### Gray
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /mnt/nfs/datasets/FractalDB/FractalDB-1k-Gray --data-set FRACTAL1k --batch-size 544 --output_dir preTrains/Tiny_244_Fractal1k-Gray/ --train-only --lr=3.0e-4 --warmup-epochs=10

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /mnt/nfs/datasets/FractalDB/FractalDB-1k-Gray --data-set FRACTAL1k --batch-size 544 --resumeid 2qnslbn3 --resume preTrains/Tiny_244_Fractal1k-Gray/checkpoint.pth --output_dir preTrains/Tiny_244_Fractal1k-Gray_fromEpoch300/ --train-only --lr=3.0e-4 --warmup-epochs=10 --start_epoch 300  --epochs 1000

############### Using EdRender FractalDB
### Fractal 1k Patch-Gray
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /mnt/nfs/datasets/FractalDB/FractalDB-1000_PATCHGRAY --data-set FRACTAL1k --output_dir preTrains/Tiny_244_DefaultHyper_EdREnder_Fractal1k-Gray/ --train-only 

### Fractal 1k Patch-Color
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /mnt/nfs/datasets/FractalDB/FractalDB-1000_PATCHCOLOR --data-set FRACTAL1k --output_dir preTrains/Tiny_244_DefaultHyper_EdREnder_Fractal1k-Gray/ --train-only 


#python pretrain.py \
#data.set.root=$DATAROOT \
#model=deit_tiny_patch16_224 \
#optim.args.lr=3.0e-4 \
#scheduler.args.warmup_epochs=10

## Debuggin purposes???
echo "code=$?"