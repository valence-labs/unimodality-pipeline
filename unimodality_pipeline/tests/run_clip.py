import os
import logging
from typing import List
import argparse
import wandb


import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from lightning.pytorch.strategies import DDPStrategy

from unimodality_pipeline.setups.clip_module import ClipModule
from unimodality_pipeline.setups.c2kd_module import C2KDModule
from unimodality_pipeline.setups.ssl_c2kd_module import SslC2KDModule
from unimodality_pipeline.setups.ph_supervised_module import PhSupervisedModule
from unimodality_pipeline.setups.shake_module import ShakeModule
from unimodality_pipeline.setups.kd_module import KDModule
from unimodality_pipeline.setups.vicreg_module import VicRegModule
from unimodality_pipeline.setups.sigclip_module import SigClipModule
from unimodality_pipeline.setups.cka_clip_module import CKAClipModule
from unimodality_pipeline.setups.dcca_module import DCCA
from unimodality_pipeline.setups.clipped_dino_module import ClippedDINO

from unimodality_pipeline.datasets.basic_dataset_module import MultiModalDataModule
from unimodality_pipeline.tools.constants import ACTIVATIONS_KEYS
from unimodality_pipeline.eval.evaluate import evaluate

def main():
    parser = argparse.ArgumentParser(description='Script to train models')
    parser.add_argument('--tx_data_path', type=str, help='Path to Tx data', default="/mnt/ps/home/CORP/ihab.bendidi/ondemand/yassir_unimodality/paired_train.h5ad")
    parser.add_argument('--ph_data_path', type=str, help='Path to Ph data', default="/mnt/ps/home/CORP/ihab.bendidi/ondemand/yassir_unimodality/HUVEC-tvn_v11_prox_bias_reduced_PHENOM1-2023-09-28_smiles_4splits_v3_filtered_transcriptomics_molphenix_embeds.parquet")
    parser.add_argument('--obsm_key', type=str, help='Obsm key', default='scVI')
    parser.add_argument('--n_eval_samples', type=int, help='Number of samples to evaluate. If None,  all eval dataset will be used.', default=None)
    
    parser.add_argument('--tx_eval_data_path', type=str, help='Path to evaluation Tx data', default="/mnt/ps/home/CORP/yassir.elmesbahi/project/unimodality_pipeline/data/predictions/crispr_l1000.h5ad")
    parser.add_argument('--tx_pred_data_dir', type=str, help='Path to prediction Tx data directory', default="/mnt/ps/home/CORP/yassir.elmesbahi/project/unimodality_pipeline/data/predictions")
    parser.add_argument('--eval_obsm_key', type=str, help='Obsm key for evaluation data', default='scVI')
    parser.add_argument('--pred_obsm_key', type=str, help='Obsm key for prediction data', default='scVI')

    parser.add_argument('--tx_input_size', type=int, help='Tx encoder input size', default=256)
    parser.add_argument('--tx_hidden_dims', type=List[int], help='Tx encoder hidden dimensions', default=[1024, 1024])
    parser.add_argument('--tx_output_size', type=int, help='Tx encoder output size', default=768)
    parser.add_argument('--tx_output_activation', type=str, help='Tx encoder output activation', default='linear')
    parser.add_argument('--tx_activations', type=List[str], help='Tx encoder activations', default=['relu'], choices=ACTIVATIONS_KEYS)
    parser.add_argument('--tx_disabled', action='store_true', help='Disables Tx encoder. If True, Tx embedding will be passed as is to the loss function.')
    parser.add_argument('--tx_frozen', action='store_true', help='Freeze Tx encoder')
    
    parser.add_argument('--ph_input_size', type=int, help='Ph encoder input size', default=768)
    parser.add_argument('--ph_hidden_dims', type=List[int], help='Ph encoder hidden dimensions', default=[1024, 1024])
    parser.add_argument('--ph_output_size', type=int, help='Ph encoder output size', default=768)
    parser.add_argument('--ph_output_activation', type=str, help='Ph encoder output activation', default='linear')
    parser.add_argument('--ph_activations', type=List[str], help='Ph encoder activations', default=['relu'], choices=ACTIVATIONS_KEYS)
    parser.add_argument('--ph_disabled', action='store_true', help='Disables Ph encoder. If True, Ph embedding will be passed as is to the loss function.')
    parser.add_argument('--ph_frozen', action='store_true', help='Freeze Ph encoder')
    
    parser.add_argument('--gather_distributed', type=bool, help='Loss hyper parameter', default=False)
    parser.add_argument('--normalize', type=bool, help='Loss hyper parameter', default=True)
    parser.add_argument('--temperature', type=float, help='Loss hyper parameter', default=0.1)
    
    parser.add_argument('--n_gpus', type=int, help='Number of GPUs', default=1)
    parser.add_argument('--fp16', action='store_true', help='Mixed precision')
    parser.add_argument('--num_workers', type=int, help='Number of workers in dataloaders', default=2)
    
    parser.add_argument('--n_epochs', type=int, help='Number of Epochs', default=20)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=1024)
    parser.add_argument('--ph_encoder_lr', type=float, help='Learning rate for phenomics encoder', default=1e-3)
    parser.add_argument('--tx_encoder_lr', type=float, help='Learning rate for transcriptomiccs encoder', default=1e-3)
    parser.add_argument('--weight_decay', type=float, help='Weight decay', default=1e-4)
    parser.add_argument('--lr_scheduler_patience', type=float, help='Scheduler hyperparameter', default=5)
    parser.add_argument('--lr_scheduler_factor', type=float, help='Scheduler hyperparameter', default=0.1)
    
    parser.add_argument('--exp_name', type=str, help='Experience name', required=True)
    parser.add_argument('--wandb_name', type=str, help='Wandb project name', required=True)
    parser.add_argument('--wandb_dir', type=str, help='Wandb cache', default='./wandb')
    parser.add_argument('--output_dir', type=str, help='Path where to save results',  default='./output')
    parser.add_argument('--ckpt_path', type=str, help='Path to weights from which to resume training',  default=None)
    
    parser.add_argument('--do_predict', action='store_true', help='Predict')
    parser.add_argument('--write_embs', action='store_true', default=False, help='whether to write embeddings for complete eval')
    parser.add_argument('--save_emb_name', type=str, default="SSLCMKD", help='Qualifier of saved embeddings')

    parser.add_argument('--momentum', type=float, default=0.95, help='Momentum for SGD optimizer')
    parser.add_argument('--min_lr', type=float, default=0.000001, help='Minimum learning rate for Tx/Ph encoders')
    
    parser.add_argument('--iters', type=int, default=10, help='number of iterations for ot')
    parser.add_argument('--Wlambda', type=float, default=0.1, help='lambda for optimal transport')
    parser.add_argument('--gamma', type=float, default=1000.0, help='gamma for optimal transport')


    parser.add_argument('--lambda_preserve_tx', type=float, default=100.0, help='coefficient for preservation Tx loss')
    parser.add_argument('--lambda_preserve_ph', type=float, default=100.0, help='coefficient for preservation Ph loss')


    parser.add_argument('--temperature_KD', type=float, default=4.0, help='Temperature for distillation')
    parser.add_argument('--krc_threshold', type=float, default=0.2, help='Threshold for KRC in OFSD')
    parser.add_argument('--ph_classifier_lr', type=float, help='Learning rate for phenomics classifier', default=1e-3)
    parser.add_argument('--tx_classifier_lr', type=float, help='Learning rate for transcriptomiccs classifier', default=0.001)


    parser.add_argument('--lambda_kl_tx', type=float, default=100.0, help='coefficient for kl divergence Tx loss')
    parser.add_argument('--lambda_kl_ph', type=float, default=100.0, help='coefficient for kl divergence Ph loss')
    parser.add_argument('--proj_size', type=int, default=8192, help='output of dino head projection size')

    parser.add_argument('--alpha', type=float, default=100.0, help='coefficient for kl divergence Tx loss')
    parser.add_argument('--beta', type=float, default=100.0, help='coefficient for kl divergence Ph loss')

    parser.add_argument('--sim_loss_weight', type=float, default=25.0, help='Weight for the similarity (invariance) loss in VICReg.')
    parser.add_argument('--var_loss_weight', type=float, default=25.0, help='Weight for the variance loss in VICReg.')
    parser.add_argument('--cov_loss_weight', type=float, default=1.0, help='Weight for the covariance loss in VICReg.')

    # New arguments for SigCLIP
    parser.add_argument('--bias', type=float, default=0.0, help='Initial value for the bias term in SigCLIP loss.')
    parser.add_argument('--learnable_bias', type=bool, default=True, help='Whether the bias term is learnable.')
    parser.add_argument('--bias_lr', type=float, default=1e-3, help='Learning rate for the bias term if it is learnable.')

    parser.add_argument('--seed', type=int, default=42, help='Random Seed')
    parser.add_argument('--method', type=str, choices=['clip', 'kd', 'c2kd', 'sslc2kd', 'ph_supervised', 'shake', 'vicreg', 'sigclip', 'dcca', 'cka_clip', 'clipped_dino'], default='clipped_dino', help='Choose the method to use')
    parser.add_argument('--pretrained_weights', type=str, default=None, help='Pretrained weights for shake and kd')


    
    args = parser.parse_args()
    pl.seed_everything(args.seed)
    
    # Initialize WandB and update args with wandb.config
    wandb.init(config=vars(args), dir=args.wandb_dir, entity="valencelabs", project=args.wandb_name, name=args.exp_name)
    args = argparse.Namespace(**wandb.config)
    
    if args.tx_disabled and args.ph_disabled:
         raise ValueError(f"Tx and Ph encoders cannot be both disabled!")
    if args.tx_frozen and args.ph_frozen:
         raise ValueError(f"Tx and Ph encoders cannot be both frozen!")
    
    ### Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        )
    logger = logging.getLogger(__name__)

    logger.info(f"######################## Starting training script")

    logger.info(f">> Initializing the training module ...")
    # Initialize the system based on the method argument
    if args.method == 'clip':
        system = ClipModule(hparams = args)
    elif args.method == 'kd':
        system = KDModule(hparams = args)
    elif args.method == 'c2kd':
        system = C2KDModule(hparams = args)
    elif args.method == 'sslc2kd':
        system = SslC2KDModule(hparams = args)
    elif args.method == 'ph_supervised':
        system = PhSupervisedModule(hparams = args)
    elif args.method == 'shake':
        system = ShakeModule(hparams = args)
    elif args.method == 'vicreg':
        system = VicRegModule(hparams = args)
    elif args.method == "sigclip":
        system = SigClipModule(hparams = args)
    elif args.method == "cka_clip":
        system = CKAClipModule(hparams = args)
    elif args.method == "dcca":
        system = DCCA(hparams = args)
    elif args.method == "clipped_dino":
        system = ClippedDINO(hparams = args)
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    logger.info(f">> Initializing the data module ...")
    data_module = MultiModalDataModule(
        multimodal_tx_data_path = args.tx_data_path, 
        multimodal_tx_obsm_key = args.obsm_key, 
        multimodal_ph_data_path = args.ph_data_path, 
        evaluation_tx_data = args.tx_eval_data_path, 
        evaluation_tx_obsm_key = args.eval_obsm_key, 
        batch_size = args.batch_size, 
        num_workers = args.num_workers, 
    )
    
    logger.info(f">> Setup encoders training mode ...")
    ## Freezing one encoder at a time
    if args.tx_frozen:
        system.set_encoder_mode(encoder="tx",  train_mode=False)
    if args.ph_frozen:
        system.set_encoder_mode(encoder="ph",  train_mode=False)

    logger.info(f">> Preparing callbacks ...")
    ckpt_cb = ModelCheckpoint(
        dirpath=f'{os.path.join(args.output_dir, args.exp_name)}',
        monitor="val_loss",
        filename="{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
        save_top_k=1,
        mode="min"
        )
    pbar = TQDMProgressBar(refresh_rate=1)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [ckpt_cb, pbar, lr_monitor]
    #callbacks = [pbar, lr_monitor]

    wandb_logger = WandbLogger(
        project=args.wandb_name,
        name=args.exp_name,
        log_model="all",
        save_dir=args.wandb_dir,
        )
    logger.info(f">> Setting up the trainer ...")
    trainer = Trainer(
        max_epochs=args.n_epochs,
        callbacks=callbacks,
        logger=wandb_logger,
        enable_model_summary=False,
        precision=16 if args.fp16 else 32,
        accelerator='gpu',
        devices=args.n_gpus,
        log_every_n_steps=5,
        check_val_every_n_epoch=1,
        strategy=DDPStrategy(find_unused_parameters=False)
                if args.n_gpus > 1 else 'auto',
        )
    logger.info(f">> Training ...")
    trainer.fit(system, datamodule=data_module, ckpt_path=args.ckpt_path)
    
    logger.info(f">> Saving last checkpoint ...")
    trainer.save_checkpoint(os.path.join(args.output_dir, "best_ckpt.ckpt"), weights_only=False)

    if args.do_predict == True:
        logger.info(f">> Evaluating ...")
        evaluate(args, system, trainer, logger, wandb)
        
    
    logger.info(f"######################## DONE")
    

if __name__ == "__main__":
    main()

    