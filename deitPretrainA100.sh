#!/bin/bash 
#$ -cwd
#$ -l rt_AF=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -o output/$JOB_ID-imnet21k.out

## Pyenv loading
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

## Modules
. /etc/profile.d/modules.sh
module purge
module load openmpi/3.1.6 cuda/11.1/11.1.1 cudnn/8.0/8.0.5 nccl/2.8/2.8.4-1

export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"

#wandb disabled

wandb enabled

# Put a comment to avoid vi bash format reading
echo "######################### START ########################################"
cat deitPretrainA100.sh

############################ Benchmark with 8-GPUs on IMNET
############################ WANDB
################## ILSVRC 2012 -- SMALL model
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/imnet/ILSVRC2012  --data-set IMNET --dist_url env://groups/gca50014/imnet/ILSVRC2012 --batch-size 1024


## Fake + ILSVRC 2012 2K classes 1.3kimgs -- SMALL model
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_small_patch16_224 --input-size 224 --data-path /groups/gca50014/imnet/Fake_v1+ILSVRC2012-2kClass-1.3kimgs/  --data-set FakeReal2kClass --dist_url env://groups/gca50014/imnet/Fake_v1+ILSVRC2012-2kClass-1.3kimgs/ --batch-size 512 --resume preTrains/preTrain_small_244_2K/checkpoint.pth --output_dir preTrain_small_244_2K-Epoch219/ --resumeid 5o6vgcgh 


## Fake + ILSVRC 2012 2K classes 1.3kimgs -- BASE_384 model
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_384 --input-size 384 --data-path /groups/gca50014/imnet/Fake_v1+ILSVRC2012-2kClass-1.3kimgs/  --data-set FakeReal2kClass --dist_url env://groups/gca50014/imnet/Fake_v1+ILSVRC2012-2kClass-1.3kimgs/ --batch-size 32 --resume preTrains/preTrain_base_384_2K-Epoch48/checkpoint.pth --output_dir preTrains/preTrain_base_384_2K-Epoch95 --resumeid qyd4hktq --start_epoch 95


## Fractal 1k
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/Fractal1k-pointcolor/ --data-set FRACTAL1k --dist_url env://groups/gca50014/Fractal/Fractal1k-pointcolor/ --batch-size 1024 --output_dir preTrains/preTrain_Tiny_244_Fractal1k/

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_384 --input-size 384 --data-path /groups/gca50014/Fractal/Fractal1k-pointcolor/ --data-set FRACTAL1k --dist_url env://groups/gca50014/Fractal/Fractal1k-pointcolor/ --batch-size 32 --output_dir preTrains/preTrain_base_384_Fractal1k/


################### ImageNet 21k
### TINY_224
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/imnet/ImageNet21k/ --data-set IMNET21k --dist_url env://groups/gca50014/imnet/ImageNet21k/ --batch-size 1024 --train-only --output_dir preTrains/tiny_imnet21k_Epoch0

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224  --input-size 224 --data-path /groups/gca50014/imnet/ImageNet21k/ --data-set IMNET21k --dist_url env://groups/gca50014/imnet/ImageNet21k/ --batch-size 1024 --train-only --resumeid 2vdfeiv9 --resume preTrains/tiny_imnet21k_Epoch1/checkpoint.pth  --output_dir preTrains/tiny_imnet21k_fromEpoch11 --start_epoch 11

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224  --input-size 224 --data-path /groups/gca50014/imnet/ImageNet21k/ --data-set IMNET21k --dist_url env://groups/gca50014/imnet/ImageNet21k/ --batch-size 1024 --train-only --resumeid 2vdfeiv9 --resume preTrains/tiny_imnet21k_fromEpoch11/checkpoint.pth  --output_dir preTrains/tiny_imnet21k_fromEpoch58 --start_epoch 58

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224  --input-size 224 --data-path /groups/gca50014/imnet/ImageNet21k/ --data-set IMNET21k --dist_url env://groups/gca50014/imnet/ImageNet21k/ --batch-size 1024 --train-only --resumeid 2vdfeiv9 --resume preTrains/tiny_imnet21k_fromEpoch58/checkpoint.pth  --output_dir preTrains/tiny_imnet21k_fromEpoch110 --start_epoch 110

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224  --input-size 224 --data-path /groups/gca50014/imnet/ImageNet21k/ --data-set IMNET21k --batch-size 1024 --train-only --resumeid 2vdfeiv9 --resume preTrains/tiny_imnet21k_fromEpoch110/checkpoint.pth  --output_dir preTrains/tiny_imnet21k_fromEpoch153 --start_epoch 153


### BASE_384
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_384 --input-size 384 --data-path /groups/gca50014/imnet/ImageNet21k/ --data-set IMNET21k --dist_url env://groups/gca50014/imnet/ImageNet21k/ --batch-size 56 --train-only --output_dir preTrains/base384_imnet21k_fromEpoch0

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_384 --input-size 384 --data-path /groups/gca50014/imnet/ImageNet21k/ --data-set IMNET21k --dist_url env://groups/gca50014/imnet/ImageNet21k/ --batch-size 56 --train-only --resumeid 22msm8g3 --resume preTrains/base384_imnet21k_fromEpoch0/checkpoint.pth --output_dir preTrains/base384_imnet21k_fromEpoch11 --start_epoch 11

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_384 --input-size 384 --data-path /groups/gca50014/imnet/ImageNet21k/ --data-set IMNET21k --dist_url env://groups/gca50014/imnet/ImageNet21k/ --batch-size 56 --train-only --resumeid 22msm8g3 --resume preTrains/base384_imnet21k_fromEpoch11/checkpoint.pth --output_dir preTrains/base384_imnet21k_fromEpoch22 --start_epoch 22

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_384 --input-size 384 --data-path /groups/gca50014/imnet/ImageNet21k/ --data-set IMNET21k  --batch-size 56 --train-only --resumeid 22msm8g3 --resume preTrains/base384_imnet21k_fromEpoch22/checkpoint.pth --output_dir preTrains/base384_imnet21k_fromEpoch33 --start_epoch 33

#################### ImageNet 21k - P
### TINY_224
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/imnet/ImageNet-21K-P/trainval --data-set IMNET21k-P --batch-size 1024 --output_dir preTrains/tiny_21k-P_fromEpoch0

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/imnet/ImageNet-21K-P/trainval --data-set IMNET21k-P --batch-size 1024 --resumeid 1rdfn1m5 --resume preTrains/tiny_21k-P_fromEpoch0/checkpoint.pth  --output_dir preTrains/tiny_21k-P_fromEpoch56 --start_epoch 56

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/imnet/ImageNet-21K-P/trainval --data-set IMNET21k-P --batch-size 1024 --resumeid 1rdfn1m5 --resume preTrains/tiny_21k-P_fromEpoch56/checkpoint.pth  --output_dir preTrains/tiny_21k-P_fromEpoch96 --start_epoch 96

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/imnet/ImageNet-21K-P/trainval --data-set IMNET21k-P --batch-size 1024 --resumeid 1rdfn1m5 --resume preTrains/tiny_21k-P_fromEpoch96/checkpoint.pth  --output_dir preTrains/tiny_21k-P_fromEpoch132 --start_epoch 132

### BASE_384
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_384 --input-size 384 --data-path /groups/gca50014/imnet/ImageNet-21K-P/trainval --data-set IMNET21k-P --batch-size 56 --output_dir preTrains/base384_21k-P_fromEpoch0

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_384 --input-size 384 --data-path /groups/gca50014/imnet/ImageNet-21K-P/trainval --data-set IMNET21k-P --batch-size 56 --resumeid 23cqmeup --resume preTrains/base384_21k-P_fromEpoch0/checkpoint.pth  --output_dir preTrains/base384_21k-P_fromEpoch12 --start_epoch 12

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_384 --input-size 384 --data-path /groups/gca50014/imnet/ImageNet-21K-P/trainval --data-set IMNET21k-P --batch-size 56 --resumeid 23cqmeup --resume preTrains/base384_21k-P_fromEpoch12/checkpoint.pth  --output_dir preTrains/base384_21k-P_fromEpoch24 --start_epoch 24

python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_384 --input-size 384 --data-path /groups/gca50014/imnet/ImageNet-21K-P/trainval --data-set IMNET21k-P --batch-size 56 --resumeid 23cqmeup --resume preTrains/base384_21k-P_fromEpoch24/checkpoint.pth  --output_dir preTrains/base384_21k-P_fromEpoch35 --start_epoch 35


##

## Trainning CIFAR from scracth
#python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path data/ --data-set CIFAR --batch-size 96 --drop-path=0.0 --opt sgd --momentum=0.9 --lr=0.01 --weight-decay=1.0e-4 --warmup-epochs=10 --output_dir preTrains/tiny224_CIFAR10



## Debuggin purposes???
echo "code=$?"