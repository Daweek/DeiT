#!/bin/bash 
#$ -cwd
#$ -l rt_AF=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -o output/$JOB_ID-FineTuneA100.out

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
cat finetuneA100.sh

############################ Benchmark with 8-GPUs on IMNET
############################ WANDB

## Fake + ILSVRC 2012 2K classes 1.3kimgs -- SMALL model
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_small_patch16_224 --input-size 224 --data-path /groups/gca50014/imnet/ILSVRC2012  --data-set IMNET --dist_url env://groups/gca50014/imnet/ILSVRC2012  --finetune preTrains/preTrain_small_244_2K/checkpoint.pth --batch-size 256

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_small_patch16_224 --input-size 224 --data-path /groups/gca50014/imnet/ILSVRC2012  --data-set IMNET --dist_url env://groups/gca50014/imnet/ILSVRC2012  --finetune preTrains/preTrain_small_244_2K/checkpoint.pth --batch-size 512

## Fake + ILSVRC 2012 2K classes 1.3kimgs -- BASE_384 model
# Cechkpoint from 127 epochs
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_384 --input-size 384 --data-path /groups/gca50014/imnet/ILSVRC2012 --data-set IMNET --dist_url env://groups/gca50014/imnet/ILSVRC2012 --finetune preTrains/preTrain_base_384_2K-Epoch95/checkpoint.pth --batch-size 32

# Cechkpoint from 127 epochs
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_384 --input-size 384 --data-path /groups/gca50014/imnet/ILSVRC2012 --data-set IMNET --dist_url env://groups/gca50014/imnet/ILSVRC2012 --finetune preTrains/preTrain_base_384_2K-Epoch95/checkpoint.pth --batch-size 56

########################### Fine Tune on CIFAR with Fractals

## Fine tune with adamw using Kataoka model on CIFAR
#python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path data/ --data-set CIFAR --finetune preTrains/fractal/deitt16_224_fractal1k_lr3e-4_300ep.pth --batch-size 96 --drop-path=0.0  --lr=0.01 --weight-decay=1.0e-4 --warmup-epochs=10

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path data/ --data-set CIFAR --finetune preTrains/fractal/deitt16_224_fractal1k_lr3e-4_300ep.pth --batch-size 32 --drop-path=0.0 --momentum=0.9 --lr=0.01 --weight-decay=1.0e-4 --warmup-epochs=10

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path data/ --data-set CIFAR --finetune preTrains/fractal/deitt16_224_fractal1k_lr3e-4_300ep.pth --batch-size 32 --drop-path=0.0 --opt momentum --momentum=0.9 --lr=0.01 --weight-decay=1.0e-4 --warmup-epochs=10

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_384 --input-size 384 --data-path data/ --data-set CIFAR --dist_url env:// --finetune preTrain/preTrain_base_384_Fractal1k.pth --batch-size 32 --drop-path=0.0 --momentum=0.9 --lr=0.01 --weight-decay=1.0e-4 --warmup-epochs=10


################################# Fine Tune with Sora's params

### Kataoka's model with Sora's params
#python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_tiny_patch16_224 --input-size 224  --finetune preTrains/fractal/deitt16_224_fractal1k_lr3e-4_300ep.pth --data-set CIFAR --data-path data/  --input-size 224 --opt sgd --batch-size 48 --epochs 1000 --cooldown-epochs 0 --lr 0.01 --sched cosine --warmup-epochs 5 --weight-decay 0.0001 --smoothing 0.1 --aa rand-m9-mstd0.5-inc1 --repeated-aug --mixup 0.8 --cutmix 1.0

### Kataoka's model with Default Hyperparameters
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224  --finetune preTrains/fractal/deitt16_224_fractal1k_lr3e-4_300ep.pth --data-set CIFAR --data-path data/  --input-size 224


### Fine tune Imnet21k model with Sora's params
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224  --finetune preTrains/tiny_imnet21k_fromEpoch153/checkpoint.pth --data-set CIFAR --data-path data/  --input-size 224 --opt sgd --batch-size 48 --epochs 1000 --cooldown-epochs 0 --lr 0.01 --sched cosine --warmup-epochs 5 --weight-decay 0.0001 --smoothing 0.1 --aa rand-m9-mstd0.5-inc1 --repeated-aug --mixup 0.8 --cutmix 1.0

### Fune tune Imnet21k model with default hyperparemeters
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224  --finetune preTrains/tiny_imnet21k_fromEpoch153/checkpoint.pth --data-set CIFAR --data-path data/  --input-size 224 

### Fine tune Imnet21k model with GITHUB hyperparemeters
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224  --finetune preTrains/tiny_imnet21k_fromEpoch153/checkpoint.pth --data-set CIFAR --data-path data/  --input-size 224  --batch-size 512 --drop-path=0.0 --opt sgd --momentum=0.9 --lr=0.01 --weight-decay=1.0e-4 --warmup-epochs=10

## Fine Tune using pre-train edgar with KataokasDB
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224  --finetune preTrains/Tiny_244_Fractal1k-Gray_fromEpoch300/checkpoint.pth --data-set CIFAR --data-path data/  --input-size 224


## Fine Tune using pre-train edgar with Edgar's DB
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224  --finetune preTrains/Tiny_244_EdREnder_Fractal1k-Gray/checkpoint.pth --data-set CIFAR --data-path data/  --input-size 224

## Fine tune with EdRender DB Color
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224  --finetune preTrains/Tiny_244_DefaultHyper_EdREnder_Fractal1k-Color/checkpoint.pth --data-set CIFAR --data-path data/  --input-size 224

## Fine tune with EdRender DB Color - 300 Epochs pre trained
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224  --finetune preTrains/Tiny_244_DefaultHyper_EdREnder_Fractal1k-Color-ep300/Tiny_244_DefaultHyper_EdREnder_Fractal1k-Color/checkpoint.pth --data-set CIFAR --data-path data/  --input-size 224

## Fine tune with EdRender DB Color - 50 Epochs pre trained
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224  --finetune preTrains/Tiny_244_DefaultHyper_KataokasDB_Color_50Epochs/checkpoint.pth --data-set CIFAR --data-path data/  --input-size 224

## Fine tune with EdRender DB Color - 50 Epochs pre trained
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224  --finetune preTrains/Tiny_244_DefaultHyper_EdRenderDB_Gray_Ep50/checkpoint.pth --data-set CIFAR --data-path data/  --input-size 224

## Fine tune with EdRender DB Gray
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224  --finetune preTrains/Tiny_244_DefaultHyper_EdREnder_Fractal1k-Gray/checkpoint.pth --data-set CIFAR --data-path data/  --input-size 224

## Fine tune with EdRender DB Gray - 300 Epochs pre trained
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224  --finetune preTrains/Tiny_244_DefaultHyper_EdREnder_Fractal1k-Gray-ep300/Tiny_244_DefaultHyper_EdREnder_Fractal1k-Gray/checkpoint.pth --data-set CIFAR --data-path data/  --input-size 224

## Fine tune with EdRender DB Gray x256 - 300 Epochs pre trained
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224  --finetune preTrains/Tiny_244_DefaultHyper_EdRenderDB_Gray_x256/checkpoint.pth --data-set CIFAR --data-path data/  --input-size 224

## Fine Tune with kataoka's FractalDB 10k - 47 epochs.
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224  --finetune preTrains/Tiny_244_DefaultHyper_KataokasDB_10kColor/checkpoint.pth --data-set CIFAR --data-path data/  --input-size 224

############################################################# FINE tune on Imnet 1k

## Train from scracth Imnet 1k
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-set IMNET --data-path /groups/gca50014/imnet/ILSVRC2012

## Fine Tune Imnet 1k for 50 epochs
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --finetune preTrains/Tiny_224_DeafaultHyper_IMNET1k_for50Epochs/checkpoint.pth --data-set CIFAR --data-path data/



## Fine tune with pre-train tiny model from Facebook
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224  --resume preTrains/deit_tiny_patch16_224-a1311bcf.pth --data-set IMNET --data-path /groups/gca50014/imnet/ILSVRC2012

## Fine tune with pre-train base model from Facebook
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_384 --input-size 384  --resume preTrains/deit_base_patch16_384-8de9b5d1.pth --data-set IMNET --data-path /groups/gca50014/imnet/ILSVRC2012 --batch-size 32

## Fine tune ImageNet 1k from pre-train Tiny ImageNett 21k
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224  --finetune preTrains/tiny_imnet21k_fromEpoch153/checkpoint.pth --data-set IMNET --data-path /groups/gca50014/imnet/ILSVRC2012  

## Fine tune ImageNet 1k from pre-train Base384 ImageNett 21k
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_384 --input-size 384 --finetune preTrains/base384_imnet21k_fromEpoch33/checkpoint.pth --data-set IMNET --data-path /groups/gca50014/imnet/ILSVRC2012 --batch-size 8


############################################################# Fine Tunning on Fractal SORA's script
#train.py ./ --pretrained --pretrained-path ./train_result/PreTraining_vit_deit_tiny_patch16_224_fake1k/model_best.pth.tar --dataset CIFAR10 --num-classes 10 --model vit_deit_tiny_patch16_224 --input-size 3 224 224 --opt sgd --batch-size 48 --epochs 1000 --cooldown-epochs 0 --lr 0.01 --sched cosine --warmup-epochs 5 --weight-decay 0.0001 --smoothing 0.1 --aa rand-m9-mstd0.5-inc1 --repeated-aug --mixup 0.8 --cutmix 1.0 --log-wandb --output train_result --experiment finetuning_vit_deit_tiny_patch16_224_1k_to_CIFAR10 -j 4


#python finetune.py \
#ckpt=$CKPT \
#data=$DATA \
#data.set.root=$DATAROOT \
#data.transform.re_prob=0 \
#data.loader.batch_size=96 \
#model=deit_tiny_patch16_224 \
#model.drop_path_rate=0.0 \
#optim=momentum \
#optim.args.lr=0.01 \
#optim.args.weight_decay=1.0e-4 \
#scheduler.args.warmup_epochs=10


###################################### FineTune with 10k and 21k
######## EdRender FractalDB-10k to CIFAR10
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path data/ --data-set CIFAR --finetune preTrains/Tiny_244_DefaultHyper_EdRenderDB_10kGray_64GPUS/checkpoint.pth 

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-set IMNET --data-path /groups/gca50014/imnet/ILSVRC2012 --finetune preTrains/Tiny_244_DefaultHyper_EdRenderDB_10kGray_64GPUS/checkpoint.pth --batch-size 128

######## ImageNet 21k Fall_11
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path data/ --data-set CIFAR --finetune preTrains/Tiny_244_DefaultHyper_21Kfall/checkpoint.pth

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-set IMNET --data-path /groups/gca50014/imnet/ILSVRC2012 --finetune preTrains/Tiny_244_DefaultHyper_21Kfall/checkpoint.pth --batch-size 256

######## FineTune Kataoka's 10k Color 
## - CIFAR10
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path data/ --data-set CIFAR --finetune preTrains/Tiny_244_DefaultHyper_KataokasDB_10kColor_fromEp109/checkpoint.pth

## - Imnet 1k
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-set IMNET --data-path /groups/gca50014/imnet/ILSVRC2012 --finetune preTrains/Tiny_244_DefaultHyper_KataokasDB_10kColor_fromEp109/checkpoint.pth --batch-size 256

#python -m torch.distributed.launch --nproc_per_node=8  --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-set IMNET --data-path /groups/gca50014/imnet/ILSVRC2012 --finetune preTrains/Tiny_244_DefaultHyper_KataokasDB_10kColor_fromEp109/checkpoint.pth --output_dir preTrains/Tiny_244_DefaultHyper_Finetune_FractalKataokas10ktoImnet1k_8aGPUs/

#python -m torch.distributed.launch --nproc_per_node=8  --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-set IMNET --data-path /groups/gca50014/imnet/ILSVRC2012 --resumeid 1kvhqmlj --resume preTrains/Tiny_244_DefaultHyper_Finetune_FractalKataokas10ktoImnet1k_8aGPUs/checkpoint.pth --output_dir preTrains/Tiny_244_DefaultHyper_Finetune_FractalKataokas10ktoImnet1k_8aGPUs_Ep200/ --start_epoch 200


######## From Facebook Model
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-set IMNET --data-path /groups/gca50014/imnet/ILSVRC2012 --resume preTrains/deit_tiny_patch16_224-a1311bcf.pth --batch-size 256

###### FractalDB 21k, i676
## Imnet 1k
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-set IMNET --data-path /groups/gca50014/imnet/ILSVRC2012 --finetune preTrains/Tiny_244_DefaultHyper_Fractal21Ki676_from286/checkpoint.pth --output_dir preTrains/Tiny_244_DefaultHyper_FineTune21k_i676_8gpus

python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-set IMNET --data-path /groups/gca50014/imnet/ILSVRC2012 --resumeid 31z708ib --resume preTrains/Tiny_244_DefaultHyper_FineTune21k_i676_8gpus/checkpoint.pth --output_dir preTrains/Tiny_244_DefaultHyper_FineTune21k_i676_8gpus_Ep201 --start_epoch 201


## CIFAR10
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path data/ --data-set CIFAR --finetune preTrains/Tiny_244_DefaultHyper_Fractal21Ki676_from286/checkpoint.pth --output_dir preTrains/Tiny_244_DefaultHyper_FineTune21ki676_toCIFAR10_8gpus

##### FractalDB 21k, i1k
## Imnet 1k
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-set IMNET --data-path /groups/gca50014/imnet/ILSVRC2012 --finetune preTrains/Tiny_244_DefaultHyper_Fractal21Ki1000_fromE293/checkpoint.pth --output_dir preTrains/Tiny_244_DefaultHyper_FineTune21k_i1k_8gpus

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-set IMNET --data-path /groups/gca50014/imnet/ILSVRC2012 --resumeid 3qrz6pb8 --resume preTrains/Tiny_244_DefaultHyper_FineTune21k_i1k_8gpus/checkpoint.pth --output_dir preTrains/Tiny_244_DefaultHyper_FineTune21k_i1k_8gpus_Ep207 --start_epoch 207


## CIFAR10
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path data/ --data-set CIFAR  --finetune preTrains/Tiny_244_DefaultHyper_Fractal21Ki1000_fromE293/checkpoint.pth --output_dir preTrains/Tiny_244_DefaultHyper_FineTune21ki1k_toCIFAR10_8gpus

## Debuggin purposes???
echo "code=$?"