import os
import torch
import secrets

import logging
from typing import List
from argparse import ArgumentParser
from pytorch_lightning import Trainer


# model
from unimodality_pipeline.setups.clip_module import ClipModule
from unimodality_pipeline.datasets.basic_dataset_module import MultiModalDataModule

def main():
    parser = ArgumentParser(description='Script to evaluate models')
    parser.add_argument('--exp_name', type=str, help='Experience name', default=None)
    parser.add_argument('--ckpt_path', type=str, help='Path to Model checkpoint', required=True)
    parser.add_argument('--tx_data_path', type=str, help='Path to Tx data', default="/mnt/ps/home/CORP/ihab.bendidi/ondemand/yassir_unimodality/huvec_compounds.h5ad")
    parser.add_argument('--obsm_key', type=str, help='Obsm key', default='X_uce_4_layers')
    parser.add_argument('--n_gpus', type=int, help='Number of GPUs', default=1)
    parser.add_argument('--fp16', action='store_true', help='Mixed precision')
    parser.add_argument('--num_workers', type=int, help='Number of workers in dataloaders', default=2)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=128)
    parser.add_argument('--output_dir', type=str, help='Path where to save embeddings',  default='./output')            
    args = parser.parse_args()

    ### Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        )
    logger = logging.getLogger(__name__)

    logger.info(f"######################## Starting inference script")
    
    
    logger.info(f">> Loading model...")
    system = ClipModule.load_from_checkpoint(args.ckpt_path)
    system.eval()
    
    logger.info(f">> Loading data module...")
    data_module = MultiModalDataModule(
        multimodal_tx_data_path = None, 
        multimodal_tx_obsm_key = None, 
        multimodal_ph_data_path = None, 
        evaluation_tx_data = args.tx_data_path, 
        evaluation_tx_obsm_key = args.obsm_key, 
        batch_size = args.batch_size, 
        num_workers = args.num_workers, 
    )
    
    logger.info(f">> Instantiating trainer...")
    trainer = Trainer(
        enable_model_summary=False,
        precision=16 if args.fp16 else 32,
        accelerator='gpu',
        devices=args.n_gpus
        )
    logger.info(f">> Predicting...")
    predictions = trainer.predict(system, datamodule=data_module)
    
    output_file = args.exp_name if f"{args.exp_name}.pt" is not None else f"weights_{secrets.token_urlsafe(8)}.pt"
    output_file = os.path.join(args.output_dir, output_file)
    logger.info(f">> Saving predictions to '{output_file}'...")
    torch.save(predictions, f'{output_file}')
    
    logger.info(f"######################## Done")
    

if __name__ == "__main__":
    main()

    