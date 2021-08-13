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
echo " "
echo $MASTER_ADDR

############################ Benchmark with 8-GPUs on IMNET
############################ WANDB

#python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${nodos} --node_rank=0 --master_addr=$MASTER_ADDR --master_port=1234 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/edRender/x256/FractalDB-10000_PATCHGRAY/ --data-set FRACTAL10k --output_dir preTrains/Tiny_244_DefaultHyper_EdRenderDB_10kGray_64GPUS/ --train-only

############################ Pre-Train FractalDB-21K_i1000
#python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${nodos} --node_rank=0 --master_addr=$MASTER_ADDR --master_port=1234 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/edRender/x256/FractalDB-21000_PATCHGRAY_i676 --data-set FRACTAL21k_i676 --output_dir preTrains/Tiny_244_DefaultHyper_Fractal21Ki676_A100/ --train-only

#python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${nodos} --node_rank=0 --master_addr=$MASTER_ADDR --master_port=1234 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/edRender/x256/FractalDB-21000_PATCHGRAY --data-set FRACTAL21k_i676 --train-only --resumeid 14k2n2jm --resume preTrains/Tiny_244_DefaultHyper_Fractal21Ki676_A100/checkpoint.pth --output_dir preTrains/Tiny_244_DefaultHyper_Fractal21Ki1000_fromE68/ --start_epoch 68

#python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${nodos} --node_rank=0 --master_addr=$MASTER_ADDR --master_port=1234 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/edRender/x256/FractalDB-21000_PATCHGRAY --data-set FRACTAL21k_i676 --train-only --resumeid 14k2n2jm --resume preTrains/Tiny_244_DefaultHyper_Fractal21Ki1000_fromE68/checkpoint.pth --output_dir preTrains/Tiny_244_DefaultHyper_Fractal21Ki1000_fromE236/ --start_epoch 236


#python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${nodos} --node_rank=0 --master_addr=$MASTER_ADDR --master_port=1234 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/edRender/x256/FractalDB-21000_PATCHGRAY --data-set FRACTAL21k_i676 --train-only --resumeid 14k2n2jm --resume preTrains/Tiny_244_DefaultHyper_Fractal21Ki1000_fromE236/checkpoint.pth --output_dir preTrains/Tiny_244_DefaultHyper_Fractal21Ki1000_fromE252/ --start_epoch 252 --batch-size 8

#python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${nodos} --node_rank=0 --master_addr=$MASTER_ADDR --master_port=1234 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/edRender/x256/FractalDB-21000_PATCHGRAY --data-set FRACTAL21k_i676 --train-only --resumeid 14k2n2jm --resume preTrains/Tiny_244_DefaultHyper_Fractal21Ki1000_fromE252/checkpoint.pth --output_dir preTrains/Tiny_244_DefaultHyper_Fractal21Ki1000_fromE293/ --start_epoch 293 --batch-size 4

############################ Pre-Train FractalDB-21K_i676
#python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${nodos} --node_rank=0 --master_addr=$MASTER_ADDR --master_port=1234 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/edRender/x256/FractalDB-21000_PATCHGRAY_i676 --data-set FRACTAL21k_i676 --train-only --resumeid 2rplgkys --resume preTrains/Tiny_244_DefaultHyper_Fractal21Ki676/checkpoint.pth --output_dir preTrains/Tiny_244_DefaultHyper_Fractal21Ki676_fromE222 --start_epoch 222

#python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${nodos} --node_rank=0 --master_addr=$MASTER_ADDR --master_port=1234 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/edRender/x256/FractalDB-21000_PATCHGRAY_i676 --data-set FRACTAL21k_i676 --train-only --resumeid 2rplgkys --resume preTrains/Tiny_244_DefaultHyper_Fractal21Ki676_fromE222/checkpoint.pth --output_dir preTrains/Tiny_244_DefaultHyper_Fractal21Ki676_fromE278 --start_epoch 278 --batch-size 8

#python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${nodos} --node_rank=0 --master_addr=$MASTER_ADDR --master_port=1234 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/edRender/x256/FractalDB-21000_PATCHGRAY_i676 --data-set FRACTAL21k_i676 --train-only --resumeid 2rplgkys --resume preTrains/Tiny_244_DefaultHyper_Fractal21Ki676_fromE278/checkpoint.pth --output_dir preTrains/Tiny_244_DefaultHyper_Fractal21Ki676_from286 --start_epoch 286 --batch-size 4


#----------------From 222 on another Wandb----------------------
#python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${nodos} --node_rank=0 --master_addr=$MASTER_ADDR --master_port=1234 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/edRender/x256/FractalDB-21000_PATCHGRAY_i676 --data-set FRACTAL21k_i676 --train-only --resume preTrains/Tiny_244_DefaultHyper_Fractal21Ki676/checkpoint.pth --output_dir preTrains/Tiny_244_DefaultHyper_Fractal21Ki676_fromE222_cero2 --epochs 300 


############################ Pre-Train FractalDB-50K
#python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${nodos} --node_rank=0 --master_addr=$MASTER_ADDR --master_port=1234 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/edRender/x256/FractalDB-50000_PATCHGRAY --data-set FRACTAL50k --output_dir preTrains/Tiny_244_DefaultHyper_Fractal50k/ --train-only


#python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${nodos} --node_rank=0 --master_addr=$MASTER_ADDR --master_port=1234 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/edRender/x256/FractalDB-50000_PATCHGRAY --data-set FRACTAL50k --resumeid 22zry7zp --resume preTrains/Tiny_244_DefaultHyper_Fractal50k/checkpoint.pth --output_dir preTrains/Tiny_244_DefaultHyper_Fractal50k_E69/ --start_epoch 69 --train-only

#python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${nodos} --node_rank=0 --master_addr=$MASTER_ADDR --master_port=1234 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/edRender/x256/FractalDB-50000_PATCHGRAY --data-set FRACTAL50k --resumeid 22zry7zp --resume preTrains/Tiny_244_DefaultHyper_Fractal50k_E69/checkpoint.pth --output_dir preTrains/Tiny_244_DefaultHyper_Fractal50k_E118/ --start_epoch 118 --train-only


python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${nodos} --node_rank=0 --master_addr=$MASTER_ADDR --master_port=1234 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-path /groups/gca50014/Fractal/edRender/x256/FractalDB-50000_PATCHGRAY --data-set FRACTAL50k --resumeid 22zry7zp --resume preTrains/Tiny_244_DefaultHyper_Fractal50k_E118/checkpoint.pth --output_dir preTrains/Tiny_244_DefaultHyper_Fractal50k_E165/ --start_epoch 165 --train-only

###################################### FINE TUNE #################################################
## Facebook model Imnet 1k with bs 512, 2 nodes
#python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${nodos} --node_rank=0 --master_addr=$MASTER_ADDR --master_port=1234 --use_env main.py --model deit_tiny_patch16_224 --input-size 224 --data-set IMNET --data-path /groups/gca50014/imnet/ILSVRC2012 --resume preTrains/deit_tiny_patch16_224-a1311bcf.pth --output_dir preTrains/Tiny_244_DefaultHyper_FineTuneFacebookModel --batch-size 512




## Debuggin purposes???
echo "code=$?"
