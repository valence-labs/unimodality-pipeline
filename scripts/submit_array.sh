#!/bin/bash
#SBATCH --job-name=explore_tvn_clipped_dino
#SBATCH --output=./terminal/explore_tvn_clipped_dino_%A_%a.out
#SBATCH --error=./error/explore_tvn_clipped_dino_%A_%a.err
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=700G
#SBATCH --time=5:00:00
#SBATCH --array=1-225%50  # Update 225 to the total number of combinations


# Define parameter lists
seed_list=(42 45 66 88 129)
tx_encoder_lr_list=(0.01 0.005 0.001 0.0005 0.0001)
classifier_lr_list=(1.0 0.5 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001)

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

echo "Running job with seed=$seed, tx_encoder_lr=$tx_encoder_lr, classifier_lr=$classifier_lr"

# Export variables
export SEED=$seed
export TX_ENCODER_LR=$tx_encoder_lr
export CLASSIFIER_LR=$classifier_lr
export EXP_NAME="explore_tvn_clipped_dino_3"

### General
export EXP_NAME="${EXP_NAME}"

### JOBS
export CUDA_VISIBLE_DEVICES=0
export NODE_RANK=0
export N_NODES=1
export GPUS_PER_NODE=1
export NUM_PROCS=1
export WORLD_SIZE=1
export MASTER_ADDR="localhost"
export MASTER_PORT=$((((RANDOM<<15)|RANDOM)%60001+5110)) # Port must be 0-65535

### PATHS
export HOME_DIR="/mnt/ps/home/CORP/ihab.bendidi"
export PROJ_NAME="unimodality_pipeline"
export PROJ_DIR="${HOME_DIR}/sandbox/code/${PROJ_NAME}"
export ONDEMAND_DIR="${HOME_DIR}/ondemand"
export OUTPUT_DIR="${PROJ_DIR}/experiments/${EXP_NAME}"
export DATA_DIR="/rxrx/scratch/sandbox/ihab.bendidi/unimodality_data/Tx/paired_train.h5ad"
export TX_DATA_PATH="/rxrx/scratch/sandbox/ihab.bendidi/unimodality_data/Tx/paired_train.h5ad"
export OBSM_KEY="scVI"
export PH_DATA_PATH="/mnt/ps/home/CORP/ihab.bendidi/ondemand/yassir_unimodality/HUVEC-tvn_v11_prox_bias_reduced_PHENOM1-2023-09-28_smiles_4splits_v3_filtered_transcriptomics_molphenix_embeds.parquet"
export TX_EVAL_DATA_PATH="/rxrx/scratch/sandbox/ihab.bendidi/biomodal_codebase/datasets/eval/crispr_l1000.h5ad"
export EVAL_OBSM_KEY="scVI"
export RUNNER="${PROJ_DIR}/${PROJ_NAME}/tests/run_clip.py"

### WANDB
export TRACKER_NAME="wandb"
export WANDB_DISABLED=false
export WANDB_SILENT=true
export WANDB_LOG_MODEL="true"
export WANDB_PROJECT="test_unimodality"
export WANDB_DATA_DIR="${PROJ_DIR}/wandb"
export WANDB_HTTP_TIMEOUT=120
export WANDB_INIT_TIMEOUT=120
export WANDB__SERVICE_WAIT=300
export WANDB_DEBUG=true
export WANDB_CACHE_DIR="${WANDB_DATA_DIR}/cache"

# Use the exported lambda values
export RUNNER_ARGS=" \
    --tx_data_path ${TX_DATA_PATH} \
    --ph_data_path ${PH_DATA_PATH} \
    --obsm_key ${OBSM_KEY} \
    --tx_eval_data_path ${TX_EVAL_DATA_PATH} \
    --eval_obsm_key ${EVAL_OBSM_KEY} \
    --n_gpus ${WORLD_SIZE} \
    --n_epochs 10 \
    --exp_name ${EXP_NAME} \
    --wandb_name ${WANDB_PROJECT} \
    --wandb_dir ${WANDB_DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --gamma 0.1 \
    --Wlambda 1000.0 \
    --lambda_preserve_tx 500.0 \
    --lambda_preserve_ph 1000.0 \
    --save_emb_name ClippedDINO5 \
    --iters 6 \
    --do_predict \
    --ph_output_size 768 \
    --tx_output_size 768 \
    --batch_size=256 \
    --krc_threshold=0.0 \
    --min_lr=1e-8 \
    --ph_classifier_lr=${CLASSIFIER_LR} \
    --ph_encoder_lr=1e-9 \
    --temperature_KD=0.7 \
    --temperature=0.1 \
    --tx_classifier_lr=${CLASSIFIER_LR} \
    --tx_encoder_lr=${TX_ENCODER_LR} \
    --proj_size=65536 \
    --lambda_kl_tx=1000.0 \
    --lambda_kl_ph=100 \
    --seed ${SEED} \
    --method clipped_dino \
    "

export PYTHON_LAUNCHER="python"

# Construct the command
export CMD="${PYTHON_LAUNCHER} ${RUNNER} ${RUNNER_ARGS}"
echo "===>>> Running command '${CMD}'"

# Run the command
$CMD
