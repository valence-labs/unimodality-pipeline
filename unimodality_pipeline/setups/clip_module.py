import torch
import logging
import torch.nn as nn
from typing import Optional, Dict, Any
from pytorch_lightning import LightningModule
from ..tools.clip_losses import ClipLoss

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    )
logger = logging.getLogger(__name__)


class ClipModule(LightningModule):
    def __init__(
        self, 
        tx_encoder: Optional[nn.Module], 
        ph_encoder: Optional[nn.Module],
        hparams: Dict[str, Any],
        ):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        # Encoders
        self.tx_encoder = tx_encoder
        self.ph_encoder = ph_encoder

        if self.tx_encoder is None and self.ph_encoder is None:
            raise ValueError(f"Both Tx and Img encoders are null. Please provide at least one model for either of them.")

        self.tx_encoder_train_mode = (self.tx_encoder is not None)
        self.ph_encoder_train_mode = (self.ph_encoder is not None)
        
        # Clip loss
        self.loss = ClipLoss(self.hparams.gather_distributed, self.hparams.normalize, self.hparams.temperature)
    
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
    
    def setup(self, stage=None):
        pass
    
    def teardown(self, stage=None):
        pass
    
    def forward(self, x):
        # Methods takes only tx embeddings 
        return self.tx_encoder(x) if (self.tx_encoder is not None) else self.ph_encoder(x)

        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x_tx, x_ph = batch
        
        loss = self.loss(
            self.tx_encoder(x_tx) if self.tx_encoder is not None else x_tx,
            self.ph_encoder(x_ph) if self.ph_encoder is not None else x_ph,
        ).mean()
        if self.hparams.gather_distributed == True:
            loss = self.all_gather(loss).mean()
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_tx, x_ph = batch
        loss = self.loss(
            self.tx_encoder(x_tx) if self.tx_encoder is not None else x_tx,
            self.ph_encoder(x_ph) if self.ph_encoder is not None else x_ph,
        ).mean()
        if self.hparams.gather_distributed == True:
            loss = self.all_gather(loss).mean()
        self.log('val_loss', loss)
        return loss
    
    def predition_step(self, batch, batch_idx):
        return self.tx_encoder(batch) if self.tx_encoder is not None else batch
    
    def configure_optimizers(self):
        parameters = []
        if self.ph_encoder_train_mode == True:
            parameters.append({"params": self.ph_encoder.parameters(), "lr": self.hparams.ph_encoder_lr})
        if self.tx_encoder_train_mode == True:
            parameters.append({"params": self.tx_encoder.parameters(), "lr": self.hparams.tx_encoder_lr})
        
        optimizer = torch.optim.Adam(parameters, weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.hparams.lr_scheduler_patience,
            factor=self.hparams.lr_scheduler_factor,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }

