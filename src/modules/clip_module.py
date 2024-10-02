import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning import LightningModule
from typing import Optional
from itertools import chain


from ..tools.losses import ClipLoss
from ..models.double_mlp_encoders import MLP

class ClipModel(LightningModule):
    def __init__(
        self, 
        tx_encoder: Optional[nn.module] = None, 
        ph_encoder: Optional[nn.module] = None,
        gather_distributed: bool = False,
        normalize: bool = True,
        inv_tau: float = 4.6052
        ):
        super().__init__()
        
        # Encoders
        self.tx_encoder = tx_encoder
        self.ph_encoder = ph_encoder
        if self.tx_encoder is None and self.ph_encoder is None:
            raise ValueError(f"Both Tx and Img encoders are null. Please provide at least one model for either of them.")
        
        # Clip loss
        self.loss = ClipLoss(gather_distributed, normalize, inv_tau)
    
    def set_encoder_mode(self, encoder="ph",  train_mode=True):
        encoder_name = encoder.lower()
        if encoder_name not in ["ph","tx","both"]:
            raise ValueError(f"")
        if encoder_name == "ph" or encoder_name == "both":
            for p in self.ph_encoder.parameters():
                p.requires_grad = train_mode
        if encoder_name == "tx" or encoder_name == "both":
            for p in self.tx_encoder.parameters():
                p.requires_grad = train_mode
       
        
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x_tx, x_ph = batch
        return self.loss(
            self.tx_encoder(x_tx[batch_idx]) if self.tx_encoder is not None else x_tx[batch_idx],
            self.ph_encoder(x_ph[batch_idx]) if self.ph_encoder is not None else x_ph[batch_idx],
        )

    def configure_optimizers(self, lr=1e-3):
        optimizer = torch.optim.Adam(chain(self.tx_encoder.parameters(), self.ph_encoder.parameters()), lr=lr)
        return optimizer