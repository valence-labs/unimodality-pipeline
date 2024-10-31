#!/usr/bin/env bash
#SBATCH --job-name=test_clipped_dino
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=h100:1
#SBATCH --gpus-per-task=h100:1
#SBATCH --mem=200G
#SBATCH --time=5:00:00
#SBATCH --output=./terminal/test_clipped_dino%j.out
#SBATCH --error=./error/test_clipped_dino%j.err 

### General
export EXP_NAME="${EXP_NAME}"  # Use the exported EXP_NAME

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
    --lambda_preserve_tx 500.0   \
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
    --ph_encoder_lr=1e-9  \
    --temperature_KD=0.7 \
    --temperature=0.1 \
    --tx_classifier_lr=${CLASSIFIER_LR} \
    --tx_encoder_lr=${TX_ENCODER_LR} \
    --proj_size=65536  \
    --lambda_kl_tx=1000.0   \
    --lambda_kl_ph=100 \
    --seed ${SEED} \
    --method clipped_dino \
    "

export PYTHON_LAUNCHER="python"

# Construct the command
export CMD="${PYTHON_LAUNCHER} ${RUNNER} ${RUNNER_ARGS}" 
echo "===>>> Running command '${CMD}'"

# Run the command
srun --jobid $SLURM_JOBID --export=ALL $CMD
