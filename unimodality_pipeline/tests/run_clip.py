import os

import logging
from typing import List
from argparse import ArgumentParser

# model
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

from unimodality_pipeline.models.mlp import MLP
from unimodality_pipeline.modules.clip_module import ClipModule
from unimodality_pipeline.tools.constants import ACTIVATIONS_KEYS

def main():
    parser = ArgumentParser(description='Script to train models')
    parser.add_argument('--tx_data_path', type=str, help='Path to Tx data', default="/mnt/ps/home/CORP/ihab.bendidi/ondemand/yassir_unimodality/huvec_compounds.h5ad")
    parser.add_argument('--ph_data_path', type=str, help='Path to Ph data', default="/mnt/ps/home/CORP/ihab.bendidi/ondemand/yassir_unimodality/HUVEC-tvn_v11_prox_bias_reduced_PHENOM1-2023-09-28_smiles_4splits_v3_filtered_transcriptomics_molphenix_embeds.parquet")
    parser.add_argument('--obsm_key', type=str, help='Obsm key', default='X_uce_4_layers')

    parser.add_argument('--tx_input_size', type=int, help='Tx encoder input size', default=128)
    parser.add_argument('--tx_hidden_dims', type=List[int], help='Tx encoder hidden dimensions', default=[256, 256])
    parser.add_argument('--tx_output_size', type=int, help='Tx encoder output size', default=128)
    parser.add_argument('--tx_output_activation', type=str, help='Tx encoder output activation', default='linear')
    parser.add_argument('--tx_activations', type=List[str], help='Tx encoder activations', default=['relu'], choices=ACTIVATIONS_KEYS)
    parser.add_argument('--tx_disabled', action='store_true', help='Disables Tx encoder. If True, Tx embedding will be passed as is to the loss function.')
    parser.add_argument('--tx_frozen', action='store_true', help='Freeze Tx encoder', default=False)
    
    parser.add_argument('--ph_input_size', type=int, help='Ph encoder input size', default=128)
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
    parser.add_argument('--wandb_dir', type=str, help='Wandb cache', default='./wandb')
    parser.add_argument('--output_dir', type=str, help='Path where to save results',  default='./output')
            
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

    logger.info(f"######################## STARTING")

    system = ClipModule(
        tx_encoder = None if args.tx_disabled else MLP(
            args.tx_input_size, 
            args.tx_hidden_dims,
            args.tx_activations, 
            args.tx_output_size,
            args.tx_output_activation
            ) , 
        ph_encoder = None if args.ph_disabled else MLP(
            args.ph_input_size, 
            args.ph_hidden_dims,
            args.ph_activations, 
            args.ph_output_size,
            args.ph_output_activation
            ),
        h_params = args)
    
    ## Freezing one encoder at a time
    if args.tx_frozen:
        system.ph_encoder_train_mode(encoder="tx",  train_mode=False)
    if args.ph_frozen:
        system.ph_encoder_train_mode(encoder="ph",  train_mode=False)

    ckpt_cb = ModelCheckpoint(
        dirpath=f'{os.path.join(args.output_dir, args.exp_name)}',
        monitor="val_loss",
        filename="{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
        save_top_k=3,
        mode="min"
        )
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [ckpt_cb, pbar]

    wandb_logger = WandbLogger(
        project=args.wandb_name,
        name=args.exp_name,
        log_model="all",
        save_dir=args.wandb_dir,
        )

    trainer = Trainer(max_epochs=args.n_epochs,
                      callbacks=callbacks,
                      logger=wandb_logger,
                      enable_model_summary=False,
                      precision=16 if args.fp16 else 32,
                      accelerator='auto',
                      devices=args.n_gpus,
                      strategy=DDPStrategy(find_unused_parameters=False)
                               if args.n_gpus > 1 else None,
                      num_sanity_val_steps=1)

    trainer.fit(system, ckpt_path=args.output_dir)

if __name__ == "__main__":
    main()

    