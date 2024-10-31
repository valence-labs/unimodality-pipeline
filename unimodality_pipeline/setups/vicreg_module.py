import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Dict, Any
from pytorch_lightning import LightningModule
from ..models.mlp import MLP
from ..tools.vicreg_loss import VicRegLoss  
from ..eval.knn import WeightedKNNClassifier

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class VicRegModule(LightningModule):
    def __init__(
        self, 
        hparams: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        # Encoders
        self.tx_encoder = None if self.hparams.tx_disabled else MLP(
            self.hparams.tx_input_size, 
            self.hparams.tx_hidden_dims,
            self.hparams.tx_activations, 
            self.hparams.tx_output_size,
            self.hparams.tx_output_activation
        )
        self.ph_encoder = None if self.hparams.ph_disabled else MLP(
            self.hparams.ph_input_size, 
            self.hparams.ph_hidden_dims,
            self.hparams.ph_activations, 
            self.hparams.ph_output_size,
            self.hparams.ph_output_activation
        )
        
        if self.tx_encoder is None and self.ph_encoder is None:
            raise ValueError("Both Tx and Ph encoders are null. Please provide at least one model for either of them.")

        self.tx_encoder_train_mode = (self.tx_encoder is not None)
        self.ph_encoder_train_mode = (self.ph_encoder is not None)

        # Knn Classifiers
        self.knn_classifier = WeightedKNNClassifier()
        self.knn_classifier_input = WeightedKNNClassifier()
        
        # VicReg Loss
        self.loss = VicRegLoss(
            sim_loss_weight=self.hparams.sim_loss_weight,
            var_loss_weight=self.hparams.var_loss_weight,
            cov_loss_weight=self.hparams.cov_loss_weight,
            gather_distributed=self.hparams.gather_distributed,
            normalize=self.hparams.normalize,
        )

    def set_encoder_mode(self, encoder="ph", train_mode=True):
        encoder_name = encoder.lower()
        if encoder_name not in ["ph", "tx", "all"]:
            raise ValueError(f"Wrong encoder name. Please select a value among '{['ph','tx','all']}'")
        if encoder_name in ["ph", "all"]:
            if (self.ph_encoder is None) and train_mode:
                raise ValueError("Ph encoder cannot be set to train mode as it is disabled!")
            elif self.ph_encoder is not None:
                for p in self.ph_encoder.parameters():
                    p.requires_grad = train_mode
                self.ph_encoder_train_mode = train_mode
        if encoder_name in ["tx", "all"]:
            if (self.tx_encoder is None) and train_mode:
                raise ValueError("Tx encoder cannot be set to train mode as it is disabled!")
            elif self.tx_encoder is not None:
                for p in self.tx_encoder.parameters():
                    p.requires_grad = train_mode
                self.tx_encoder_train_mode = train_mode

    def forward(self, x):
        # Method takes only tx embeddings 
        return self.tx_encoder(x) if (self.tx_encoder is not None) else self.ph_encoder(x)

    def training_step(self, batch, batch_idx):
        x_tx, x_ph, labels = batch

        # Aligned transcriptomics embeddings
        z_tx_aligned = self.tx_encoder(x_tx) if self.tx_encoder is not None else x_tx

        # Phenomics embeddings
        z_ph = self.ph_encoder(x_ph) if self.ph_encoder is not None else x_ph

        # VicReg loss
        alignment_loss = self.loss(z_tx_aligned, z_ph)

        # Total loss
        loss = alignment_loss

        # Logging
        self.log('train_loss', loss)
        self.log('alignment_loss', alignment_loss)

        self.knn_classifier.update(train_features=z_tx_aligned.detach(), train_targets=labels)
        if self.current_epoch == 0:
            self.knn_classifier_input.update(train_features=x_tx.detach(), train_targets=labels)

        return loss

    def validation_step(self, batch, batch_idx):
        x_tx, x_ph, labels = batch

        # Aligned transcriptomics embeddings
        z_tx_aligned = self.tx_encoder(x_tx) if self.tx_encoder is not None else x_tx

        # Phenomics embeddings
        z_ph = self.ph_encoder(x_ph) if self.ph_encoder is not None else x_ph

        # VicReg loss
        alignment_loss = self.loss(z_tx_aligned, z_ph)

        # Total loss
        loss = alignment_loss

        # Logging
        self.log('val_loss', loss)
        self.log('val_alignment_loss', alignment_loss)

        self.knn_classifier.update(test_features=z_tx_aligned.detach(), test_targets=labels)
        if self.current_epoch == 0:
            self.knn_classifier_input.update(test_features=x_tx.detach(), test_targets=labels)

        return loss

    def on_validation_epoch_end(self):
        knn_top1, knn_top5 = self.knn_classifier.compute()
        self.log('knn_top1', knn_top1)
        self.log('knn_top5', knn_top5)
        if self.current_epoch == 0:
            knn_top1_input, knn_top5_input = self.knn_classifier_input.compute()
            self.log('original_knn_top1', knn_top1_input)
            self.log('original_knn_top5', knn_top5_input)

    def configure_optimizers(self):
        parameters = []
        if self.ph_encoder_train_mode:
            parameters.append({"params": self.ph_encoder.parameters(), "lr": self.hparams.ph_encoder_lr})
        if self.tx_encoder_train_mode:
            parameters.append({"params": self.tx_encoder.parameters(), "lr": self.hparams.tx_encoder_lr})

        optimizer = torch.optim.SGD(parameters, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)

        # Set up the Cosine Annealing scheduler
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.n_epochs,
            eta_min=self.hparams.min_lr
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }
