#!/usr/bin/env bash
#SBATCH --job-name=train-simple-clip-disabled-ph
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=h100:1
#SBATCH --gpus-per-task=h100:1
#SBATCH --mem=512G
#SBATCH --time=1:00:00
#SBATCH --output=/mnt/ps/home/CORP/yassir.elmesbahi/project/unimodality_pipeline/out/train-simple-clip-disabled-ph.out
#SBATCH --error=/mnt/ps/home/CORP/yassir.elmesbahi/project/unimodality_pipeline/out/train-simple-clip-disabled-ph.out 

# In a SLURM job, you CANNOT use `conda activate` and instead MUST use:
#source ${HOME_DIR}/.bashrc
#source ${HOME_DIR}/miniforge3/etc/profile.d/conda.sh
#mamba activate unimodality

### General
export EXP_NAME="train-simple-clip-disabled-ph"



### Environment
export CUDA_HOME='/cm/shared/apps/cuda12.1/toolkit/12.1.1'
export FORCE_CUDA="1"
export OMP_NUM_THREADS=8

### JOBS
export CUDA_VISIBLE_DEVICES=0
export NODE_RANK=$SLURM_NODEID
export N_NODES=$SLURM_NNODES
export GPUS_PER_NODE=${SLURM_GPUS_PER_NODE#*:}
export NUM_PROCS=$((N_NODES * GPUS_PER_NODE))
export WORLD_SIZE=$NUM_PROCS
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((((RANDOM<<15)|RANDOM)%60001+5110)) # Port must be 0-65535

### PATHS
export HOME_DIR="/mnt/ps/home/CORP/yassir.elmesbahi"
export PROJ_NAME="unimodality_pipeline"
export PROJ_DIR="${HOME_DIR}/project/${PROJ_NAME}"
export ONDEMAND_DIR="${HOME_DIR}/ondemand"
export OUTPUT_DIR="${ONDEMAND_DIR}/${PROJ_NAME}/${EXP_NAME}"
export DATA_DIR="/mnt/ps/home/CORP/ihab.bendidi/ondemand/yassir_unimodality"
export TX_DATA_PATH="${DATA_DIR}/huvec_compounds.h5ad"
export OBSM_KEY="X_uce_4_layers"
export PH_DATA_PATH="${DATA_DIR}/HUVEC-tvn_v11_prox_bias_reduced_PHENOM1-2023-09-28_smiles_4splits_v3_filtered_transcriptomics_molphenix_embeds.parquet"
export TX_EVAL_DATA_PATH="${PROJ_DIR}/data/replogle_2022.h5ad"
export EVAL_OBSM_KEY="test"
export RUNNER="${PROJ_DIR}/${PROJ_NAME}/tests/run_clip.py"


### WANDB
export TRACKER_NAME="wandb"
export WANDB_API_KEY="ceeb005a3731a69cc1377d12184e2c28ede292bf"
export WANDB_DISABLED=false
export WANDB_SILENT=true
export WANDB_LOG_MODEL="true"
export WANDB_PROJECT="UNIMODALITY_PROJECT"
export WANDB_HTTP_TIMEOUT=120
export WANDB_INIT_TIMEOUT=120
export WANDB__SERVICE_WAIT=300
export WANDB_DEBUG=true
export SANDBOX_DIR="${HOME_DIR}/sandbox"
export WANDB_DATA_DIR="${SANDBOX_DIR}/.wandb"
export WANDB_CACHE_DIR="${WANDB_DATA_DIR}/cache"


### TRAINING PARAMETERS
export BATCH_SIZE=128
export N_EPOCH=1
export PH_ENCODER_LR=1e-3
export TX_ENCODER_LR=1e-3
export WEIGHT_DECAY=1e-3
export LR_SCHEDULER_PATIENCE=5
export LR_SCHEDULER_FACTOR=0.1

export RUNNER_ARGS=" \
    --tx_data_path ${TX_DATA_PATH} \
    --ph_data_path ${PH_DATA_PATH} \
    --obsm_key ${OBSM_KEY} \
    --tx_eval_data_path ${TX_EVAL_DATA_PATH} \
    --eval_obsm_key ${EVAL_OBSM_KEY} \
    --n_gpus ${WORLD_SIZE} \
    --n_epochs ${N_EPOCH} \
    --batch_size ${BATCH_SIZE} \
    --ph_encoder_lr ${PH_ENCODER_LR} \
    --tx_encoder_lr ${TX_ENCODER_LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --lr_scheduler_patience ${LR_SCHEDULER_PATIENCE} \
    --lr_scheduler_factor ${LR_SCHEDULER_FACTOR} \
    --exp_name ${EXP_NAME} \
    --wandb_name ${WANDB_PROJECT} \
    --wandb_dir ${WANDB_DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --ph_disabled \
    --do_predict \
    "


export PYTHON_LAUNCHER="python \
"


# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="${PYTHON_LAUNCHER} ${RUNNER} ${RUNNER_ARGS}" 
echo "===>>> Running command '${CMD}'"
srun --jobid $SLURM_JOBID --export=ALL  $CMD
