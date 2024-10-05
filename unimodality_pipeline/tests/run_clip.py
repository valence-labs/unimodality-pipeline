import os
import logging
from typing import List
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from lightning.pytorch.strategies import DDPStrategy

from unimodality_pipeline.setups.clip_module import ClipModule
from unimodality_pipeline.datasets.basic_dataset_module import MultiModalDataModule
from unimodality_pipeline.tools.constants import ACTIVATIONS_KEYS

def main():
    parser = ArgumentParser(description='Script to train models')
    parser.add_argument('--tx_data_path', type=str, help='Path to Tx data', default="/mnt/ps/home/CORP/ihab.bendidi/ondemand/yassir_unimodality/huvec_compounds.h5ad")
    parser.add_argument('--ph_data_path', type=str, help='Path to Ph data', default="/mnt/ps/home/CORP/ihab.bendidi/ondemand/yassir_unimodality/HUVEC-tvn_v11_prox_bias_reduced_PHENOM1-2023-09-28_smiles_4splits_v3_filtered_transcriptomics_molphenix_embeds.parquet")
    parser.add_argument('--obsm_key', type=str, help='Obsm key', default='X_uce_4_layers')
    
    parser.add_argument('--tx_eval_data_path', type=str, help='Path to evaluation Tx data', default="/mnt/ps/home/CORP/yassir.elmesbahi/project/unimodality_pipeline/data/replogle_2022.h5ad")
    parser.add_argument('--eval_obsm_key', type=str, help='Obsm key for evaluation data', default='test')

    parser.add_argument('--tx_input_size', type=int, help='Tx encoder input size', default=1280)
    parser.add_argument('--tx_hidden_dims', type=List[int], help='Tx encoder hidden dimensions', default=[256, 256])
    parser.add_argument('--tx_output_size', type=int, help='Tx encoder output size', default=128)
    parser.add_argument('--tx_output_activation', type=str, help='Tx encoder output activation', default='linear')
    parser.add_argument('--tx_activations', type=List[str], help='Tx encoder activations', default=['relu'], choices=ACTIVATIONS_KEYS)
    parser.add_argument('--tx_disabled', action='store_true', help='Disables Tx encoder. If True, Tx embedding will be passed as is to the loss function.')
    parser.add_argument('--tx_frozen', action='store_true', help='Freeze Tx encoder')
    
    parser.add_argument('--ph_input_size', type=int, help='Ph encoder input size', default=768)
    parser.add_argument('--ph_hidden_dims', type=List[int], help='Ph encoder hidden dimensions', default=[256, 256])
    parser.add_argument('--ph_output_size', type=int, help='Ph encoder output size', default=128)
    parser.add_argument('--ph_output_activation', type=str, help='Ph encoder output activation', default='linear')
    parser.add_argument('--ph_activations', type=List[str], help='Ph encoder activations', default=['relu'], choices=ACTIVATIONS_KEYS)
    parser.add_argument('--ph_disabled', action='store_true', help='Disables Ph encoder. If True, Ph embedding will be passed as is to the loss function.')
    parser.add_argument('--ph_frozen', action='store_true', help='Freeze Ph encoder')
    
    parser.add_argument('--gather_distributed', type=bool, help='Loss hyper parameter', default=False)
    parser.add_argument('--normalize', type=bool, help='Loss hyper parameter', default=True)
    parser.add_argument('--temperature', type=float, help='Loss hyper parameter', default=4.6052)
    
    parser.add_argument('--n_gpus', type=int, help='Number of GPUs', default=1)
    parser.add_argument('--fp16', action='store_true', help='Mixed precision')
    parser.add_argument('--num_workers', type=int, help='Number of workers in dataloaders', default=2)
    
    parser.add_argument('--n_epochs', type=int, help='Number of Epochs', default=10)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=128)
    parser.add_argument('--ph_encoder_lr', type=float, help='Learning rate for phenomics encoder', default=1e-3)
    parser.add_argument('--tx_encoder_lr', type=float, help='Learning rate for transcriptomiccs encoder', default=1e-3)
    parser.add_argument('--weight_decay', type=float, help='Weight decay', default=1e-5)
    parser.add_argument('--lr_scheduler_patience', type=float, help='Scheduler hyperparameter', default=5)
    parser.add_argument('--lr_scheduler_factor', type=float, help='Scheduler hyperparameter', default=0.1)
    
    parser.add_argument('--exp_name', type=str, help='Experience name', required=True)
    parser.add_argument('--wandb_name', type=str, help='Wandb project name', required=True)
    parser.add_argument('--wandb_dir', type=str, help='Wandb cache', default='./wandb')
    parser.add_argument('--output_dir', type=str, help='Path where to save results',  default='./output')
    parser.add_argument('--ckpt_path', type=str, help='Path to weights from which to resume training',  default=None)
    
    parser.add_argument('--do_predict', action='store_true', help='Predict')
    
            
    args = parser.parse_args()
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
    system = ClipModule(hparams = args)
    
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
        save_top_k=3,
        mode="min"
        )
    pbar = TQDMProgressBar(refresh_rate=1)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [ckpt_cb, pbar, lr_monitor]

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
        logger.info(f">> Predicting ...")
        #data_module.setup(stage="predict")
        predictions = trainer.predict(system, datamodule=data_module)
        logger.info(f">> Saving predictions ...")
        torch.save(predictions, os.path.join(args.output_dir, "predictions.pt"))
    
    logger.info(f"######################## DONE")
    

if __name__ == "__main__":
    main()

    