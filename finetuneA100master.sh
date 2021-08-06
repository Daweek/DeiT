#!/bin/bash 
#$ -cwd
#$ -l rt_AF=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -o output/$JOB_ID-MultiMaster.out

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
cat pretrainA100master.sh

############################ Catch IP
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep -w inet | cut -d " " -f 6 | cut -d "/" -f 1)

nodos=8
echo "\n"
echo $MASTER_ADDR

######################## Fine tune from KATAOKA's 10k Color to Imnet 1k
#python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${nodos} --node_rank=0 --master_addr=$MASTER_ADDR --master_port=1234 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-set IMNET --data-path /groups/gca50014/imnet/ILSVRC2012 --finetune preTrains/Tiny_244_DefaultHyper_KataokasDB_10kColor_fromEp109/checkpoint.pth --output_dir preTrains/Tiny_244_DefaultHyper_Finetune_Fractal10ktoImnet1k/

## Fractal 21k_i646 to Imnet 1k
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${nodos} --node_rank=0 --master_addr=$MASTER_ADDR --master_port=1234 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-set IMNET --data-path /groups/gca50014/imnet/ILSVRC2012 --finetune preTrains/Tiny_244_DefaultHyper_Fractal21Ki676_from286/checkpoint.pth --output_dir preTrains/Tiny_244_DefaultHyper_FineTune21k_i676 --batch-size 128

## Debuggin purposes???
echo "code=$?"