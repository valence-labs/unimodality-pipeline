#!/usr/bin/env bash
#SBATCH --job-name=augmented_clip_42_1e-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=h100:1
#SBATCH --gpus-per-task=h100:1
#SBATCH --mem=512G
#SBATCH --time=5:00:00
#SBATCH --output=/mnt/ps/home/CORP/yassir.elmesbahi/project/unimodality_pipeline/out/augmented_clip_42_1e-3.out
#SBATCH --error=/mnt/ps/home/CORP/yassir.elmesbahi/project/unimodality_pipeline/out/augmented_clip_42_1e-3.out

#SBATCH --array=1-225%50  # Update 225 to the total number of combinations


# Define parameter lists
SEEDS=(42 45 66 88 129)
PH_ENCODER_LR_LIST=(1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9 1e-10)
METHODS=('clip' 'kd' 'c2kd' 'sslc2kd' 'ph_supervised' 'shake' 'vicreg' 'sigclip' 'dcca' 'cka_clip' 'clipped_dino')

# Calculate total combinations
total_seeds=${#seed_list[@]}
total_tx_lrs=${#tx_encoder_lr_list[@]}
total_class_lrs=${#classifier_lr_list[@]}

total_jobs=$((total_seeds * total_tx_lrs * total_class_lrs))

# Calculate indices for each parameter
task_id=$((SLURM_ARRAY_TASK_ID - 1))
seed_index=$((task_id / (total_tx_lrs * total_class_lrs)))
tx_lr_index=$(( (task_id / total_class_lrs) % total_tx_lrs ))
class_lr_index=$((task_id % total_class_lrs))

# Get the parameter values
seed=${seed_list[$seed_index]}
tx_encoder_lr=${tx_encoder_lr_list[$tx_lr_index]}
classifier_lr=${classifier_lr_list[$class_lr_index]}


# In a SLURM job, you CANNOT use `conda activate` and instead MUST use:
#source ${HOME_DIR}/.bashrc
#source ${HOME_DIR}/miniforge3/etc/profile.d/conda.sh
#mamba activate unimodality

### General
export SEED=42
export PH_ENCODER_LR=1e-3
export METHOD='clip'
export EXP_NAME="ablation_augmented_${METHOD}_${SEED}_${PH_ENCODER_LR}"


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
export TX_DATA_PATH="${DATA_DIR}/paired_train.h5ad"
export OBSM_KEY="scVI"
export PH_DATA_PATH="${DATA_DIR}/HUVEC-tvn_v11_prox_bias_reduced_PHENOM1-2023-09-28_smiles_4splits_v3_filtered_transcriptomics_molphenix_embeds.parquet"
export TX_EVAL_DATA_PATH="${PROJ_DIR}/data/crispr_l1000.h5ad"
export TX_PRED_DATA_DIR="/mnt/ps/home/CORP/yassir.elmesbahi/project/unimodality_pipeline/data/predictions"
export EVAL_OBSM_KEY="scVI"
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



### Training parameters
export BATCH_SIZE=1024
#export N_EPOCHS=50
export N_EPOCHS=1
export GAMMA=0.1
export WLAMBDA=1000.0
export LAMBDA_PRESERVE_TX=1000.0
export LAMBDA_PRESERVE_PH=1000.0
export SAVE_EMB_NAME='Clip'
export ITERS=6
export TX_OUTPUT_SIZE=768
export PH_OUTPUT_SIZE=768
export KRC_THRESHOLD=0.0
export MIN_LR=1e-6
export TEMPERATURE_KD=2
export TX_ENCODER_LR=1e-3


export WORLD_SIZE=1
# Use the exported lambda values
export RUNNER_ARGS=" \
    --tx_data_path ${TX_DATA_PATH} \
    --ph_data_path ${PH_DATA_PATH} \
    --tx_eval_data_path ${TX_EVAL_DATA_PATH} \
    --tx_pred_data_dir ${TX_PRED_DATA_DIR} \
    --obsm_key ${OBSM_KEY} \
    --eval_obsm_key ${EVAL_OBSM_KEY} \
    --n_gpus ${WORLD_SIZE} \
    --exp_name ${EXP_NAME} \
    --wandb_name ${WANDB_PROJECT} \
    --wandb_dir ${WANDB_DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --n_epochs ${N_EPOCHS} \
    --gamma ${GAMMA} \
    --Wlambda ${WLAMBDA} \
    --lambda_preserve_tx ${LAMBDA_PRESERVE_TX} \
    --lambda_preserve_ph ${LAMBDA_PRESERVE_PH} \
    --save_emb_name ${SAVE_EMB_NAME} \
    --iters ${ITERS} \
    --do_predict \
    --tx_output_size ${TX_OUTPUT_SIZE} \
    --ph_output_size ${PH_OUTPUT_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --krc_threshold ${KRC_THRESHOLD} \
    --min_lr ${MIN_LR} \
    --temperature_KD ${TEMPERATURE_KD} \
    --tx_encoder_lr ${TX_ENCODER_LR} \
    --ph_disabled \
    --ph_encoder_lr ${PH_ENCODER_LR} \
    --seed ${SEED} \
    --method ${METHOD} \
    "

export PYTHON_LAUNCHER="python \
"


# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="${PYTHON_LAUNCHER} ${RUNNER} ${RUNNER_ARGS}" 
echo "===>>> Running command '${CMD}'"
#srun --jobid $SLURM_JOBID --export=ALL  $CMD
$CMD
