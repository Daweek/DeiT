#!/bin/bash 
#$ -cwd
#$ -l rt_AF=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -o output/$JOB_ID-FineTun-2kto1kReal.out

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

wandb disabled

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

## Fine tune with adamw using Kataoka model on CIFAR
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path data/ --data-set CIFAR --finetune preTrains/fractal/deitt16_224_fractal1k_lr3e-4_300ep.pth --batch-size 96 --drop-path=0.0  --lr=0.01 --weight-decay=1.0e-4 --warmup-epochs=10


python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_384 --input-size 384 --data-path data/ --data-set CIFAR --dist_url env:// --finetune preTrain/preTrain_base_384_Fractal1k.pth --batch-size 32 --drop-path=0.0 --momentum=0.9 --lr=0.01 --weight-decay=1.0e-4 --warmup-epochs=10





python finetune.py \
ckpt=$CKPT \
data=$DATA \
data.set.root=$DATAROOT \
data.transform.re_prob=0 \
data.loader.batch_size=96 \
model=deit_tiny_patch16_224 \
model.drop_path_rate=0.0 \
optim=momentum \
optim.args.lr=0.01 \
optim.args.weight_decay=1.0e-4 \
scheduler.args.warmup_epochs=10


  

## Debuggin purposes???
echo "code=$?"