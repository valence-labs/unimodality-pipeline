import torch
import logging
import torch.nn as nn
from typing import Optional, Dict, Any
from itertools import chain

from lightning import LightningModule
from torch.utils.data import DataLoader

from ..tools.clip_losses import ClipLoss
from ..datasets.basic_dataset import (
    MultimodalDataset,
    multimodal_collate_fn
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    #level=logging.DEBUG,
    )
logger = logging.getLogger(__name__)


class ClipModule(LightningModule):
    def __init__(
        self, 
        tx_encoder: Optional[nn.Module], 
        ph_encoder: Optional[nn.Module],
        h_params: Dict[str, Any],
        ):
        super().__init__()
        self.save_hyperparameters(h_params)
        
        # Encoders
        self.tx_encoder = tx_encoder
        self.ph_encoder = ph_encoder

        if self.tx_encoder is None and self.ph_encoder is None:
            raise ValueError(f"Both Tx and Img encoders are null. Please provide at least one model for either of them.")

        self.tx_encoder_train_mode = (self.tx_encoder is not None)
        self.ph_encoder_train_mode = (self.ph_encoder is not None)
        
        # Clip loss
        self.loss = ClipLoss(self.h_params.gather_distributed, self.h_params.normalize, self.h_params.inv_tau)
    
    def set_encoder_mode(self, encoder="ph",  train_mode=True):
        encoder_name = encoder.lower()
        if encoder_name not in ["ph","tx","all"]:
            raise ValueError(f"Wrong encoder name. Please select a value among '{['ph','tx','all']}'")
        if (encoder_name in ["ph", "all"]):
            if (self.ph_encoder is None) and (train_mode==True):
                raise ValueError(f"Ph encoder cannot be set to train mode as it is disabled!")
            elif self.ph_encoder is not None:
                for p in self.ph_encoder.parameters():
                    p.requires_grad = train_mode
                self.ph_encoder_train_mode = train_mode
        if (encoder_name in ["tx", "all"]):
            if (self.tx_encoder is None) and (train_mode==True):
                raise ValueError(f"Tx encoder cannot be set to train mode as it is disabled!")
            elif self.tx_encoder is not None:
                for p in self.tx_encoder.parameters():
                    p.requires_grad = train_mode
                self.tx_encoder_train_mode = train_mode
    
    def setup(self):
        logger.info('>> Setup():: Loading datasets ...')
        self.train_dataset = MultimodalDataset(self.h_params.tx_data_path, self.h_params.ph_data_path, self.h_params.obsm_key, mode='train')
        self.val_dataset = MultimodalDataset(self.h_params.tx_data_path, self.h_params.ph_data_path,  self.h_params.obsm_key, mode='test')
    
    def teardown(self):
        logger.info('>> Teardown():: ...')
        pass
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          collate_fn=multimodal_collate_fn,
                          shuffle=True,
                          num_workers=self.h_params.num_workers,
                          batch_size=self.h_params.batch_size,
                          pin_memory=True,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          collate_fn=multimodal_collate_fn,
                          shuffle=True,
                          num_workers=self.hparams.num_workers,
                          batch_size=self.h_params.batch_size,
                          pin_memory=True,
                          drop_last=True)

        
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x_tx, x_ph = batch
        loss = self.loss(
            self.tx_encoder(x_tx[batch_idx]) if self.tx_encoder is not None else x_tx[batch_idx],
            self.ph_encoder(x_ph[batch_idx]) if self.ph_encoder is not None else x_ph[batch_idx],
        ).mean()
        loss = self.all_gather(loss)
        self.log('train_loss', loss.mean())
        return loss

    def validation_step(self, batch, batch_idx):
        x_tx, x_ph = batch
        loss = self.loss(
            self.tx_encoder(x_tx[batch_idx]) if self.tx_encoder is not None else x_tx[batch_idx],
            self.ph_encoder(x_ph[batch_idx]) if self.ph_encoder is not None else x_ph[batch_idx],
        )
        loss = self.all_gather(loss)
        self.log('val_loss', loss.mean())
        return loss
    
    def configure_optimizers(self):
        parameters = []
        if self.ph_encoder_train_mode == True:
            parameters.append({"params": self.ph_encoder.parameters(), "lr": self.h_params.ph_encoder_lr})
        if self.tx_encoder_train_mode == True:
            parameters.append({"params": self.tx_encoder.parameters(), "lr": self.h_params.tx_encoder_lr})
        
        optimizer = torch.optim.Adam(parameters, weight_decay=self.h_params.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.h_params.lr_scheduler_patience,
            factor=self.h_params.lr_scheduler_factor,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }

