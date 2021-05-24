#!/bin/bash 
#$ -cwd
#$ -l rt_AF=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -o output/$JOB_ID-Pre-FineTun-2K.out

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
cat finetuneA100.sh


############################ Benchmark with 8-GPUs on IMNET
################################# - WANDB

## Fake + ILSVRC 2012 2K classes 1.3kimgs -- SMALL model
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_small_patch16_224 --input-size 224 --data-path /groups/gca50014/imnet/Fake_v1+ILSVRC2012-2kClass-1.3kimgs/  --data-set FakeReal2kClass --dist_url env://groups/gca50014/imnet/Fake_v1+ILSVRC2012-2kClass-1.3kimgs/ --batch-size 512 --output_dir preTrain_small_244_2K-CheckP/

## Fake + ILSVRC 2012 2K classes 1.3kimgs -- BASE_384 model
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_384 --input-size 384 --data-path /groups/gca50014/imnet/Fake_v1+ILSVRC2012-2kClass-1.3kimgs/  --data-set FakeReal2kClass --dist_url env://groups/gca50014/imnet/Fake_v1+ILSVRC2012-2kClass-1.3kimgs/ --batch-size 32 --output_dir preTrain_base_384_2K/


## Resume work...
## Fake + ILSVRC 2012 2K classes 1.3kimgs -- SMALL model
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_small_patch16_224 --input-size 224 --resume preTrain_small_244_2K/checkpoint.pth --data-path /groups/gca50014/imnet/Fake_v1+ILSVRC2012-2kClass-1.3kimgs/  --data-set FakeReal2kClass --dist_url env://groups/gca50014/imnet/Fake_v1+ILSVRC2012-2kClass-1.3kimgs/ --batch-size 512 --output_dir preTrain_small_244_2K-E[229]/ 

## Fake + ILSVRC 2012 2K classes 1.3kimgs -- BASE_384 model
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_384 --input-size 384 --resume preTrain_base_384_2K/checkpoint.pth --data-path /groups/gca50014/imnet/Fake_v1+ILSVRC2012-2kClass-1.3kimgs/  --data-set FakeReal2kClass --dist_url env://groups/gca50014/imnet/Fake_v1+ILSVRC2012-2kClass-1.3kimgs/ --batch-size 32 --output_dir preTrain_base_384_2K-E[48]/


## Debuggin purposes???
echo "code=$?"