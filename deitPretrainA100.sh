#!/bin/bash 
#$ -cwd
#$ -l rt_AF=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -o output/$JOB_ID-Fractal1k.out

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

############################ Benchmark with 8-GPUs on IMNETls
############################ WANDB
################## ILSVRC 2012 -- SMALL model
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/imnet/ILSVRC2012  --data-set IMNET --dist_url env://groups/gca50014/imnet/ILSVRC2012 --batch-size 1024

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/imnet/ILSVRC2012  --data-set IMNET --output_dir preTrains/Tiny_224_DeafaultHyper_IMNET1k_for50Epochs/ --epochs 50


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

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_384 --input-size 384 --data-path /groups/gca50014/imnet/ImageNet-21K-P/trainval --data-set IMNET21k-P --batch-size 56 --resumeid 23cqmeup --resume preTrains/base384_21k-P_fromEpoch24/checkpoint.pth  --output_dir preTrains/base384_21k-P_fromEpoch35 --start_epoch 35

###################### Fractal Original Data Base from Kataoka
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/org/FractalDB-1k-Gray --data-set FRACTAL1k --batch-size 96 --output_dir preTrains/Tiny_244_DSfromKataoka_momentum_bs96_Fractal1k-Gray/ --aa rand-m9-mstd0.5-inc1   --train-only --opt momentum --lr=3.0e-4 --warmup-epochs=10 --epochs 300

## Trainning CIFAR from scracth
#python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path data/ --data-set CIFAR --batch-size 96 --drop-path=0.0 --opt sgd --momentum=0.9 --lr=0.01 --weight-decay=1.0e-4 --warmup-epochs=10 --output_dir preTrains/tiny224_CIFAR10



################## PreTrain with Sora's Hyperparameters on Kataokas DB
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/org/FractalDB-1k-Gray --data-set FRACTAL1k --opt adamw --batch-size 64 --epochs 300 --cooldown-epochs 0 --lr 6.0e-4 --sched cosine --warmup-epochs 10 --weight-decay 0.05 --smoothing 0.1 --drop-path 0.1 --aa rand-m9-mstd0.5-inc1 --repeated-aug --mixup 0.8 --cutmix 1.0 --reprob 0.25 --remode pixel --train-interpolation bicubic --hflip 0.0 --output_dir preTrains/Tiny_244_SORAsHyper_KataokasDB/ --train-only


################## PreTrain with Sora's Hyperparameters on EdRender DB
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/FractalDB-1000_PATCHGRAY --data-set FRACTAL1k --opt adamw --batch-size 64 --epochs 300 --cooldown-epochs 0 --lr 6.0e-4 --sched cosine --warmup-epochs 10 --weight-decay 0.05 --smoothing 0.1 --drop-path 0.1 --aa rand-m9-mstd0.5-inc1 --repeated-aug --mixup 0.8 --cutmix 1.0 --reprob 0.25 --remode pixel --train-interpolation bicubic --hflip 0.0 --output_dir preTrains/Tiny_244_SORAsHyper_EdRernderDB/ --train-only

# Sora san's hyperparameters

#fractalDB_pretrain.py /groups/gcd50691/datasets/FractalDB-1k-Gray --model vit_deit_tiny_patch16_224 --opt adamw --batch-size 64 --epochs 300 --cooldown-epochs 0 --lr 6.0e-4 --sched cosine --warmup-epochs 10 --weight-decay 0.05 --smoothing 0.1 --drop-path 0.1 --aa rand-m9-mstd0.5-inc1 --repeated-aug --mixup 0.8 --cutmix 1.0 --reprob 0.25 --remode pixel --interpolation bicubic --hflip 0.0 --eval-metric loss --log-wandb --output train_result --experiment PreTraining_vit_deit_tiny_patch16_224_fractalDB_1k_gray -j 8


################# Pre train with default hyperparameters
## Kataokas DB 1k
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/org/FractalDB-1k-Gray --data-set FRACTAL1k  --output_dir preTrains/Tiny_244_DefaultHyper_KataokasDB/ --train-only

## Kataokas DB 10k - Color
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/org/FractalDB-10k-Color --data-set FRACTAL10k  --output_dir preTrains/Tiny_244_DefaultHyper_KataokasDB_10kColor/ --train-only

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/org/FractalDB-10k-Color --data-set FRACTAL10k --train-only --resumeid 26vpvjya --resume preTrains/Tiny_244_DefaultHyper_KataokasDB_10kColor/checkpoint.pth --output_dir preTrains/Tiny_244_DefaultHyper_KataokasDB_10kColor_fromEp48/ --start_epoch 48

## EdRender DB 10k Patch Gray
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/edRender/x256/FractalDB-10000_PATCHGRAY/ --data-set FRACTAL10k  --output_dir preTrains/Tiny_244_DefaultHyper_EdRenderDB_10kGray/ --train-only


## EdRender DB- Gray
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/edRender/FractalDB-1000_PATCHGRAY --data-set FRACTAL1k  --output_dir preTrains/Tiny_244_DefaultHyper_KataokasDB_Gray/ --train-only

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/edRender/FractalDB-1000_PATCHGRAY --data-set FRACTAL1k  --output_dir preTrains/Tiny_244_DefaultHyper_EdRenderDB_Gray_Ep50/ --train-only --epochs 50

## EdRender DB- Color
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/FractalDB-1000_PATCHCOLOR --data-set FRACTAL1k  --output_dir preTrains/Tiny_244_DefaultHyper_KataokasDB_Color/ --train-only

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/edRender/FractalDB-1000_PATCHCOLOR --data-set FRACTAL1k  --output_dir preTrains/Tiny_244_DefaultHyper_KataokasDB_Color_50Epochs/ --train-only --epochs 50


## EdRender DB- Gray x256
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/edRender/x256/FractalDB-1000_PATCHGRAY --data-set FRACTAL1k  --output_dir preTrains/Tiny_244_DefaultHyper_EdRenderDB_Gray_x256/ --train-only

## EdRender 21k i676 x256 
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/edRender/x256/FractalDB-21000_PATCHGRAY_i676 --data-set FRACTAL21k_i676 --output_dir preTrains/Tiny_244_DefaultHyper_Fractal21Ki676_8GPUs/ --train-only

## EdRender 21k i1k x256 
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/edRender/x256/FractalDB-21000_PATCHGRAY --data-set FRACTAL21k_i676 --output_dir preTrains/Tiny_244_DefaultHyper_Fractal21k_i1k_8GPUs/ --train-only 

## One node to pretrain Fractal 21k
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/edRender/x256/FractalDB-21000_PATCHGRAY_i676 --data-set FRACTAL21k_i676 --train-only --resumeid 2rplgkys --resume preTrains/Tiny_244_DefaultHyper_Fractal21Ki676_fromE222/checkpoint.pth --output_dir preTrains/Tiny_244_DefaultHyper_Fractal21Ki676_fromE278_8gpus --start_epoch 278 --batch-size 1024

## One node to pretrain Fractal 21k, 1k class
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/edRender/x256/FractalDB-21000_PATCHGRAY --data-set FRACTAL21k_i676 --train-only --resumeid 14k2n2jm --resume preTrains/Tiny_244_DefaultHyper_Fractal21Ki1000_fromE236/checkpoint.pth --output_dir preTrains/Tiny_244_DefaultHyper_Fractal21Ki1000_fromE252/ --start_epoch 252 --batch-size 1024

## Debuggin purposes???
echo "code=$?"