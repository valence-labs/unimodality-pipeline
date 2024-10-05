#!/usr/bin/env bash
#SBATCH --job-name=predict-simple-clip
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=h100:1
#SBATCH --gpus-per-task=h100:1
#SBATCH --mem=512G
#SBATCH --time=1:00:00
#SBATCH --output=/mnt/ps/home/CORP/yassir.elmesbahi/project/unimodality_pipeline/out/predict-simple-clip.out
#SBATCH --error=/mnt/ps/home/CORP/yassir.elmesbahi/project/unimodality_pipeline/out/predict-simple-clip.out 

# In a SLURM job, you CANNOT use `conda activate` and instead MUST use:
#source ${HOME_DIR}/.bashrc
#source ${HOME_DIR}/miniforge3/etc/profile.d/conda.sh
#mamba activate unimodality

### General
export EXP_NAME="predict-simple-clip"


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
export CKPT_PATH="${ONDEMAND_DIR}/${PROJ_NAME}/train-simple-clip/best_ckpt.ckpt"
export TX_DATA_PATH="${PROJ_DIR}/data/replogle_2022.h5ad"
export OBSM_KEY="test"
export RUNNER="${PROJ_DIR}/${PROJ_NAME}/tests/run_inference.py"


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


export RUNNER_ARGS=" \
    --exp_name ${EXP_NAME} \
    --ckpt_path ${CKPT_PATH} \
    --tx_data_path ${TX_DATA_PATH} \
    --obsm_key ${OBSM_KEY} \
    --n_gpus ${WORLD_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --output_dir ${OUTPUT_DIR} \
    "
#--n_samples ${N_SAMPLES} \

export PYTHON_LAUNCHER="python \
"


# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="${PYTHON_LAUNCHER} ${RUNNER} ${RUNNER_ARGS}" 
echo "===>>> Running command '${CMD}'"
srun --jobid $SLURM_JOBID --export=ALL  $CMD
