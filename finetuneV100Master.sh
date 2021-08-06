#!/bin/bash 
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -o output/$JOB_ID-FinetuneMASTER.out

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
cat finetuneV100Master.sh

export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep -w inet | cut -d " " -f 6 | cut -d "/" -f 1)

nodos=32
echo "\n"
echo $MASTER_ADDR

############################ Benchmark with 4-GPUs on IMNET
############################ WANDB
## ILSVRC 2012 -- SMALL model
#python -m torch.distributed.launch --nproc_per_node=4 --nnodes=${nodos} --node_rank=0 --master_addr=$MASTER_ADDR --master_port=1234 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-set IMNET --data-path /groups/gca50014/imnet/ILSVRC2012 --finetune preTrains/Tiny_244_DefaultHyper_EdRenderDB_10kGray_64GPUS/checkpoint.pth

############################ Pre-Train FractalDB-21K_i676
#python -m torch.distributed.launch --nproc_per_node=4 --nnodes=${nodos} --node_rank=0 --master_addr=$MASTER_ADDR --master_port=1234 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/edRender/x256/FractalDB-21000_PATCHGRAY_i676 --data-set FRACTAL21k_i676 --output_dir preTrains/Tiny_244_DefaultHyper_Fractal21Ki676/ --train-only

############################ Pre-Train FractalDB-21K_i1k
#python -m torch.distributed.launch --nproc_per_node=4 --nnodes=${nodos} --node_rank=0 --master_addr=$MASTER_ADDR --master_port=1234 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/edRender/x256/FractalDB-21000_PATCHGRAY --data-set FRACTAL21k_i676 --output_dir preTrains/Tiny_244_DefaultHyper_Fractal21k_i1k/ --train-only 

python -m torch.distributed.launch --nproc_per_node=4 --nnodes=${nodos} --node_rank=0 --master_addr=$MASTER_ADDR --master_port=1234 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/edRender/x256/FractalDB-21000_PATCHGRAY --data-set FRACTAL21k_i676 --train-only --resumeid 24p5bth8 --resume preTrains/tiny_imnet21k_fromEpoch238/checkpoint.pth --output_dir preTrains/tiny_imnet21k_fromEpoch247 --start_epoch 247


########################### Continue Kataokas 10k Color ---- PRETRAIN
#python -m torch.distributed.launch --nproc_per_node=4 --nnodes=${nodos} --node_rank=0 --master_addr=$MASTER_ADDR --master_port=1234 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/org/FractalDB-10k-Color --data-set FRACTAL10k --data-set FRACTAL10k --resumeid 26vpvjya --resume preTrains/Tiny_244_DefaultHyper_KataokasDB_10kColor_fromEp48/checkpoint.pth  --output_dir preTrains/Tiny_244_DefaultHyper_KataokasDB_10kColor_fromEp109/ --train-only --start_epoch 109


## Debuggin purposes???
echo "code=$?"