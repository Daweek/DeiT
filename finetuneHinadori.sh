#!/bin/bash
#---------YBATCH -r dgx-a100_8
#YBATCH -r am_4
#SBATCH -N 1
#SBATCH -J FrkFineTune
#SBATCH --time=120:00:00
#SBATCH --output output/%j-FracFineTune.out

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
cat finetuneHinadori.sh

############################ WANDB
############ First trail with my data, first version of Fractal 1k, 100 instances Gray.
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_384 --input-size 384 --data-path data/ --data-set CIFAR --dist_url env:// --finetune preTrain/preTrain_base_384_Fractal1k.pth --batch-size 24 --drop-path=0.0 --opt sgd --momentum=0.9 --lr=0.01 --weight-decay=1.0e-4 --warmup-epochs=10

## From Kataoka models and parameters...
#8GPUS
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path data/ --data-set CIFAR --finetune preTrain/deitt16_224_fractal1k_lr3e-4_300ep.pth --batch-size 96 --drop-path=0.0 --opt sgd --momentum=0.9 --lr=0.01 --weight-decay=1.0e-4 --warmup-epochs=10
#4GPUS
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path data/ --data-set CIFAR --finetune preTrain/deitt16_224_fractal1k_lr3e-4_300ep.pth --batch-size 96 --drop-path=0.0 --opt sgd --momentum=0.9 --lr=0.01 --weight-decay=1.0e-4 --warmup-epochs=10

#python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path data/ --data-set CIFAR --finetune preTrain/deitt16_224_fractal1k_lr3e-4_300ep.pth --batch-size 96 --drop-path=0.0 --opt sgd --momentum=0.9 --lr=0.01 --weight-decay=1.0e-4 --warmup-epochs=10

###################### My preTrain on Frac 1k Color
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path data/ --data-set CIFAR --finetune preTrains/preTrain_Tiny_244_Fractal1k/checkpoint.pth --batch-size 96 --drop-path=0.0 --opt sgd --momentum=0.9 --lr=0.01 --weight-decay=1.0e-4 --warmup-epochs=10

## With DeiT original Tiny model and Parameters from github
#python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path data/ --data-set CIFAR --finetune preTrain/deit_tiny_patch16_224-a1311bcf.pth --batch-size 512 --drop-path=0.0 --opt sgd --momentum=0.9 --lr=0.01 --weight-decay=1.0e-4 --warmup-epochs=10

###################### My preTrain on Frac 1k Gray
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path data/ --data-set CIFAR --finetune preTrains/Tiny_244_Fractal1k-Gray/checkpoint.pth --batch-size 96 --drop-path=0.0 --opt sgd --momentum=0.9 --lr=0.01 --weight-decay=1.0e-4 --warmup-epochs=10

## Debuggin purposes???
echo "code=$?"