#!/bin/bash
# Define lists of lambda_preserve_ph and lambda_preserve_tx
seed_list=(42 45 66 88)
tx_encoder_lr_list=(0.01 0.005 0.001 0.0005 0.0001)
classifier_lr_list=(1.0 0.5 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001)
# Loop over combinations
for seed in "${seed_list[@]}"; do
    for tx_encoder_lr in "${tx_encoder_lr_list[@]}"; do
        for classifier_lr in "${classifier_lr_list[@]}"; do
            # Update EXP_NAME to include lambda values
            export EXP_NAME="explore_tvn_clipped_dino_3"
            # Submit the job with exported variables
            sbatch --export=ALL,SEED=$seed,TX_ENCODER_LR=$tx_encoder_lr,CLASSIFIER_LR=$classifier_lr,EXP_NAME=${EXP_NAME} ./scripts/run_exp.sh
        done
    done
done

