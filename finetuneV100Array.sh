#!/bin/bash 
#$ -cwd
#$ -l rt_F=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -o output/$JOB_ID-FinetuneARRAY.out

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

#wandb enabled

# Put a comment to avoid vi bash format reading
echo "######################### START ########################################"
cat finetuneV100Array.sh

nodos=32
echo $SGE_TASK_ID

############################ Benchmark with 4-GPUs on IMNET
############################ WANDB

## ILSVRC 2012 -- SMALL model
#python -m torch.distributed.launch --nproc_per_node=4 --nnodes=${nodos} --master_addr="10.0.2.23" --node_rank=$SGE_TASK_ID --master_port=1234 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-set IMNET --data-path /groups/gca50014/imnet/ILSVRC2012 --finetune preTrains/Tiny_244_DefaultHyper_EdRenderDB_10kGray_64GPUS/checkpoint.pth

############################ Pre-Train FractalDB-21K_i676
#python -m torch.distributed.launch --nproc_per_node=4 --nnodes=${nodos} --master_addr="10.0.3.17" --node_rank=$SGE_TASK_ID  --master_port=1234 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/edRender/x256/FractalDB-21000_PATCHGRAY_i676 --data-set FRACTAL21k_i676 --output_dir preTrains/Tiny_244_DefaultHyper_Fractal21Ki676/ --train-only



############################ Pre-Train FractalDB-21K_i1k
#python -m torch.distributed.launch --nproc_per_node=4 --nnodes=${nodos} --master_addr="10.0.4.23" --node_rank=$SGE_TASK_ID  --master_port=1234 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/edRender/x256/FractalDB-21000_PATCHGRAY --data-set FRACTAL21k_i676 --output_dir preTrains/Tiny_244_DefaultHyper_Fractal21k_i1k/ --train-only 

#python -m torch.distributed.launch --nproc_per_node=4 --nnodes=${nodos} --master_addr="10.0.3.8" --node_rank=$SGE_TASK_ID  --master_port=1234 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/edRender/x256/FractalDB-21000_PATCHGRAY --data-set FRACTAL21k_i676 --train-only --resumeid 24p5bth8 --resume preTrains/tiny_imnet21k_fromEpoch238/checkpoint.pth --output_dir preTrains/tiny_imnet21k_fromEpoch247 --start_epoch 247




## Debuggin purposes???
echo "code=$?"