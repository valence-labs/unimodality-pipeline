#!/usr/bin/env bash
#SBATCH --job-name=augmented_clip_ablation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=a100:1
#SBATCH --gpus-per-task=a100:1
#SBATCH --mem=512G
#SBATCH --time=5:00:00
#SBATCH --output=/mnt/ps/home/CORP/yassir.elmesbahi/project/unimodality_pipeline/out/augmented_clip_ablation_%j.out
#SBATCH --error=/mnt/ps/home/CORP/yassir.elmesbahi/project/unimodality_pipeline/out/augmented_clip_ablation_%j.out

#SBATCH --array=1-350%50  # Update 225 to the total number of combinations

# Define parameter lists
export SEEDS=(42 45 66 88 129)
export PH_ENCODER_LR_LIST=(1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9 1e-10)
#export METHODS=('clip' 'kd' 'c2kd' 'sslc2kd' 'ph_supervised' 'shake' 'vicreg' 'sigclip' 'dcca' 'cka_clip' 'clipped_dino')
export METHODS=('clip' 'vicreg' 'sigclip' 'shake' 'c2kd' 'dcca')

# Calculate total combinations
export N_SEEDS=${#SEEDS[@]}
export N_PH_ENCODER_LR=${#PH_ENCODER_LR_LIST[@]}
export N_METHODS=${#METHODS[@]}

export N_JOBS=$((N_SEEDS * N_PH_ENCODER_LR * N_METHODS))

# Calculate indices for each parameter
export TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))
export SEED_ID=$((TASK_ID / (N_PH_ENCODER_LR * N_METHODS)))
export PH_ENCODER_LR_ID=$(( (TASK_ID / N_METHODS) % N_PH_ENCODER_LR ))
export METHOD_ID=$((TASK_ID % N_METHODS))

export SEED=${SEEDS[$SEED_ID]}
export PH_ENCODER_LR=${PH_ENCODER_LR_LIST[$PH_ENCODER_LR_ID]}
export METHOD=${METHODS[$METHOD_ID]}

# In a SLURM job, you CANNOT use `conda activate` and instead MUST use:
#source ${HOME_DIR}/.bashrc
#source ${HOME_DIR}/miniforge3/etc/profile.d/conda.sh
#mamba activate unimodality

### General
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
export LAMBDA_PRESERVE_TX=1000.0
export LAMBDA_PRESERVE_PH=1000.0
export TX_OUTPUT_SIZE=768
export PH_OUTPUT_SIZE=768
export KRC_THRESHOLD=0.0


declare -A BATCH_SIZE=(
    ["clip"]=1024
    ["vicreg"]=1024
    ["sigclip"]=1024
    ["shake"]=1024
    ["kd"]=1024
    ["c2kd"]=1024
    ["dcca"]=2048
)

declare -A N_EPOCHS=(
    ["clip"]=50
    ["vicreg"]=10
    ["sigclip"]=10
    ["shake"]=10
    ["kd"]=10
    ["c2kd"]=30
    ["dcca"]=50
)

declare -A GAMMA=(
    ["clip"]=0.1
    ["vicreg"]=1000.0
    ["sigclip"]=1000.0
    ["shake"]=1000.0
    ["kd"]=1000.0
    ["c2kd"]=0.1
    ["dcca"]=0.1
)

declare -A WLAMBDA=(
    ["clip"]=1000.0
    ["vicreg"]=0.1
    ["sigclip"]=0.1
    ["shake"]=0.1
    ["kd"]=0.1
    ["c2kd"]=1000.0
    ["dcca"]=1000.0
)

declare -A SAVE_EMB_NAME=(
    ["clip"]='Clip'
    ["vicreg"]='VicReg'
    ["sigclip"]='Sigclip'
    ["shake"]='SHAKE'
    ["kd"]='KD'
    ["c2kd"]='C2kdOptim'
    ["dcca"]='DCC'
)

declare -A ITERS=(
    ["clip"]=6
    ["vicreg"]=10
    ["sigclip"]=10
    ["shake"]=10
    ["kd"]=10
    ["c2kd"]=6
    ["dcca"]=6
)

declare -A MIN_LR=(
    ["clip"]=1e-6
    ["vicreg"]=1e-10
    ["sigclip"]=1e-10
    ["shake"]=1e-10
    ["kd"]=1e-10
    ["c2kd"]=1e-7
    ["dcca"]=1e-10
)

declare -A PH_CLASSIFIER_LR=(
    ["clip"]=
    ["vicreg"]=1e-7
    ["sigclip"]=1e-7
    ["shake"]=1e-7
    ["kd"]=1e-7
    ["c2kd"]=1e-3
    ["dcca"]=
)

declare -A TEMPERATURE_KD=(
    ["clip"]=2
    ["vicreg"]=9
    ["sigclip"]=9
    ["shake"]=9
    ["kd"]=9
    ["c2kd"]=2
    ["dcca"]=2
)

declare -A TX_CLASSIFIER_LR=(
    ["clip"]=1e-3
    ["vicreg"]=1e-3
    ["sigclip"]=1e-3
    ["shake"]=1e-3
    ["kd"]=1e-3
    ["c2kd"]=1e-3
    ["dcca"]=1e-3
)

declare -A TX_ENCODER_LR=(
    ["clip"]=1e-3
    ["vicreg"]=1e-1
    ["sigclip"]=1e-1
    ["shake"]=1e-1
    ["kd"]=1e-1
    ["c2kd"]=1e-1
    ["dcca"]=1e-6
)

declare -A PRETRAINED_WEIGHTS=(
    ["clip"]=
    ["vicreg"]=
    ["sigclip"]=
    ["shake"]='/mnt/ps/home/CORP/yassir.elmesbahi/project/unimodality_pipeline/data/save_ph_encoder/save_ph_encoder/epoch=03-val_loss=5.42-val_acc=0.00.ckpt'
    ["kd"]='/mnt/ps/home/CORP/yassir.elmesbahi/project/unimodality_pipeline/data/save_ph_encoder/save_ph_encoder/epoch=03-val_loss=5.42-val_acc=0.00.ckpt'
    ["c2kd"]=
    ["dcca"]=
)

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
    --n_epochs ${N_EPOCHS[$METHOD]} \
    --gamma ${GAMMA[$METHOD]} \
    --Wlambda ${WLAMBDA[$METHOD]} \
    --lambda_preserve_tx ${LAMBDA_PRESERVE_TX[$METHOD]} \
    --lambda_preserve_ph ${LAMBDA_PRESERVE_PH[$METHOD]} \
    --save_emb_name ${SAVE_EMB_NAME[$METHOD]} \
    --iters ${ITERS[$METHOD]} \
    --do_predict \
    --tx_output_size ${TX_OUTPUT_SIZE} \
    --ph_output_size ${PH_OUTPUT_SIZE} \
    --batch_size ${BATCH_SIZE[$METHOD]} \
    --krc_threshold ${KRC_THRESHOLD} \
    --min_lr ${MIN_LR[$METHOD]} \
    --ph_classifier_lr ${PH_CLASSIFIER_LR[$METHOD]} \
    --temperature_KD ${TEMPERATURE_KD[$METHOD]} \
    --tx_classifier_lr ${TX_CLASSIFIER_LR[$METHOD]} \
    --tx_encoder_lr ${TX_ENCODER_LR[$METHOD]} \
    --ph_encoder_lr ${PH_ENCODER_LR} \
    --pretrained_weights ${PRETRAINED_WEIGHTS[$METHOD]} \
    --seed ${SEED} \
    --method ${METHOD} \
    "

export PYTHON_LAUNCHER="python \
"


# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="${PYTHON_LAUNCHER} ${RUNNER} ${RUNNER_ARGS}" 
echo "===>>> Running command '${CMD}'"
srun --jobid $SLURM_JOBID --export=ALL  $CMD
