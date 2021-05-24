#!/bin/bash 
#$ -cwd
#$ -l rt_AF=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -o output/$JOB_ID-FakeFULL-1K.out

## Pyenv loading
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

## Modules
. /etc/profile.d/modules.sh
module purge
module load openmpi/3.1.6 cuda/11.1/11.1.1 cudnn/8.0/8.0.5 nccl/2.8/2.8.4-1

# No buffered python
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"

wandb enabled

# Put a comment to avoid vi bash format reading
echo "######################### START ########################################"
cat deitPretrainA100.sh

############################ Benchmark with 8-GPUs on IMNET
################################# - WANDB
## Fake + ILSVRC 2012 1K classes 2.6kimgs -- SMALL model
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_small_patch16_224 --input-size 224 --data-path /groups/gca50014/imnet/Fake_v1+ILSVRC2012-1kClass-2.6kimgs/  --data-set FakeReal1k --dist_url env://groups/gca50014/imnet/Fake_v1+ILSVRC2012-1kClass-2.6kimgs/ --batch-size 320

## Fake + ILSVRC 2012 1K classes 2.6kimgs -- BASE model
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_224 --input-size 224 --data-path /groups/gca50014/imnet/Fake_v1+ILSVRC2012-1kClass-2.6kimgs/  --data-set FakeReal1k --dist_url env://groups/gca50014/imnet/Fake_v1+ILSVRC2012-1kClass-2.6kimgs/ --batch-size 256

## Fake + ILSVRC 2012 1K classes 2.6kimgs -- BASE_384 model
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_384 --input-size 384 --data-path /groups/gca50014/imnet/Fake_v1+ILSVRC2012-1kClass-2.6kimgs/  --data-set FakeReal1k --dist_url env://groups/gca50014/imnet/Fake_v1+ILSVRC2012-1kClass-2.6kimgs/ --batch-size 56

## Fake + ILSVRC 2012 1K classes 2.6kimgs -- BASE_384 >HARDESTILATION model
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_distilled_patch16_384 --input-size 384 --data-path /groups/gca50014/imnet/Fake_v1+ILSVRC2012-1kClass-2.6kimgs/  --data-set FakeReal1k --dist_url env://groups/gca50014/imnet/Fake_v1+ILSVRC2012-1kClass-2.6kimgs/ --distillation-type hard --teacher-model regnety_160 --teacher-path teacher/regnety_160-a5fe301d.pth --batch-size 32


## Fake + ILSVRC 2012 2K classes 1.3kimgs -- SMALL model
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_small_patch16_224 --input-size 224 --data-path /groups/gca50014/imnet/Fake_v1+ILSVRC2012-2kClass-1.3kimgs/  --data-set FakeReal2kClass --dist_url env://groups/gca50014/imnet/Fake_v1+ILSVRC2012-2kClass-1.3kimgs/ --batch-size 512 --output_dir preTrain_small_244_2K/

## Fake + ILSVRC 2012 2K classes 1.3kimgs -- BASE model
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_224 --input-size 224 --data-path /groups/gca50014/imnet/Fake_v1+ILSVRC2012-2kClass-1.3kimgs/  --data-set FakeReal2kClass --dist_url env://groups/gca50014/imnet/Fake_v1+ILSVRC2012-2kClass-1.3kimgs/ --batch-size 256

## Fake + ILSVRC 2012 2K classes 1.3kimgs -- BASE_384 model
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_384 --input-size 384 --data-path /groups/gca50014/imnet/Fake_v1+ILSVRC2012-2kClass-1.3kimgs/  --data-set FakeReal2kClass --dist_url env://groups/gca50014/imnet/Fake_v1+ILSVRC2012-2kClass-1.3kimgs/ --batch-size 32

## Fake + ILSVRC 2012 2K classes 1.3kimgs -- BASE_384->HARDESTILATION model
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_distilled_patch16_384 --input-size 384 --data-path /groups/gca50014/imnet/Fake_v1+ILSVRC2012-2kClass-1.3kimgs/  --data-set FakeReal2kClass --dist_url env://groups/gca50014/imnet/Fake_v1+ILSVRC2012-2kClass-1.3kimgs/ --distillation-type hard --teacher-model regnety_160 --teacher-path teacher/regnety_160-a5fe301d.pth --batch-size 32



## Fake FULL 1k -- SMALL model
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_small_patch16_224 --input-size 224 --data-path /groups/gca50014/imnet/FakeImageNet1k_v1  --data-set FakeReal1k --dist_url env://groups/gca50014/imnet/FakeImageNet1k_v1 --batch-size 512

## Fake FULL 1k -- BASE model
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_224 --input-size 224 --data-path /groups/gca50014/imnet/FakeImageNet1k_v1  --data-set FakeReal1k --dist_url env://groups/gca50014/imnet/FakeImageNet1k_v1 --batch-size 256

## Fake FULL 1k -- BASE_384 model
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_384 --input-size 384 --data-path /groups/gca50014/imnet/FakeImageNet1k_v1  --data-set FakeReal1k --dist_url env://groups/gca50014/imnet/FakeImageNet1k_v1 --batch-size 56

## Fake FULL 1k -- BASE_384->HARDESTILATION model
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_distilled_patch16_384 --input-size 384 --data-path /groups/gca50014/imnet/FakeImageNet1k_v1  --data-set FakeReal1k --dist_url env://groups/gca50014/imnet/FakeImageNet1k_v1 --distillation-type hard --teacher-model regnety_160 --teacher-path teacher/regnety_160-a5fe301d.pth --batch-size 32


## ILSVRC 2012 -- SMALL model
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_small_patch16_224 --input-size 224 --data-path /groups/gca50014/imnet/ILSVRC2012  --data-set IMNET --dist_url env://groups/gca50014/imnet/ILSVRC2012 --batch-size 256

## ILSVRC 2012 -- BASE model
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_224 --input-size 224 --data-path /groups/gca50014/imnet/ILSVRC2012  --data-set IMNET --dist_url env://groups/gca50014/imnet/ILSVRC2012 --batch-size 64

## ILSVRC 2012 -- BASE_384 model
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_384 --input-size 384 --data-path /groups/gca50014/imnet/ILSVRC2012  --data-set IMNET --dist_url env://groups/gca50014/imnet/ILSVRC2012 --batch-size 32

## ILSVRC 2012 -- BASE_384->HARDESTILATION model
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_distilled_patch16_384 --input-size 384 --data-path /groups/gca50014/imnet/ILSVRC2012  --data-set IMNET --dist_url env://groups/gca50014/imnet/ILSVRC2012 --distillation-type hard --teacher-model regnety_160 --teacher-path teacher/regnety_160-a5fe301d.pth --batch-size 32


## Debuggin purposes???
echo "code=$?"