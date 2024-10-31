import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Dict, Any
from pytorch_lightning import LightningModule
from ..models.mlp import MLP
from ..tools.clip_losses import ClipLoss
from ..tools.dcc_loss import DCCLoss
from ..tools.ot_loss import MultiViewLoss

from ..eval.knn import WeightedKNNClassifier

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    )
logger = logging.getLogger(__name__)


class DCCA(LightningModule):
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
            raise ValueError(f"Both Tx and Img encoders are null. Please provide at least one model for either of them.")

        self.tx_encoder_train_mode = (self.tx_encoder is not None)
        self.ph_encoder_train_mode = (self.ph_encoder is not None)


        self.lambda_preserve_tx = self.hparams.lambda_preserve_tx
        self.lambda_preserve_ph = self.hparams.lambda_preserve_ph

        # Initialize the fixed projection matrix as a buffer
        aligned_dim = self.hparams.tx_output_size  # Dimension of z_tx_aligned (e.g., 768)
        original_dim = self.hparams.tx_input_size  # Dimension of z_tx_original (e.g., 256)

        # Create the projection matrix
        projection_matrix = torch.randn(aligned_dim, original_dim)
        projection_matrix = projection_matrix / torch.norm(projection_matrix, dim=0, keepdim=True)  # Normalize columns

        # Register the projection matrix as a buffer
        self.register_buffer('projection_matrix', projection_matrix)

        # Knn
        self.knn_classifier = WeightedKNNClassifier()
        self.knn_classifier_input = WeightedKNNClassifier()
        
        # DCCA loss
        self.loss = DCCLoss(
            outdim_size=30,
            use_all_singular_values=True,
            epsilon=1e-6,  # You can adjust this value as needed
        )

    def linear_CKA(self, z1, z2):
        """Compute the linear CKA between two sets of representations."""
        # Ensure the inputs are centered
        z1 = z1 - z1.mean(dim=0, keepdim=True)
        z2 = z2 - z2.mean(dim=0, keepdim=True)

        numerator = torch.norm(torch.mm(z1.T, z2), p='fro') ** 2
        denominator = torch.norm(torch.mm(z1.T, z1), p='fro') * torch.norm(torch.mm(z2.T, z2), p='fro')
        cka = numerator / denominator
        return cka
    
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
        x_tx, x_ph, labels = batch

        # Original transcriptomics embeddings (before alignment)
        z_tx_original = x_tx  # Assuming x_tx is already an embedding; if not, pass through a frozen encoder

        # Aligned transcriptomics embeddings
        z_tx_aligned = self.tx_encoder(x_tx) if self.tx_encoder is not None else x_tx

        # Phenomics embeddings (assuming ph_encoder is frozen)
        z_ph = self.ph_encoder(x_ph) if self.ph_encoder is not None else x_ph

        # Alignment loss
        alignment_loss = self.loss(z_tx_aligned, z_ph)

        # Total loss
        loss = alignment_loss 

        # Logging
        self.log('train_loss', loss)
        self.log('alignment_loss', alignment_loss)

        self.knn_classifier.update(train_features=z_tx_aligned, train_targets=labels)

        if self.current_epoch == 0 :
            self.knn_classifier_input.update(train_features=z_tx_original, train_targets=labels)

        return loss

    def validation_step(self, batch, batch_idx):
        x_tx, x_ph, labels = batch

        # Original transcriptomics embeddings (before alignment)
        z_tx_original = x_tx  # Assuming x_tx is already an embedding; if not, pass through a frozen encoder

        # Aligned transcriptomics embeddings
        z_tx_aligned = self.tx_encoder(x_tx) if self.tx_encoder is not None else x_tx

        # Phenomics embeddings (assuming ph_encoder is frozen)
        z_ph = self.ph_encoder(x_ph) if self.ph_encoder is not None else x_ph

        # Alignment loss
        alignment_loss = self.loss(z_tx_aligned, z_ph)


        # Total loss
        loss = alignment_loss #+ self.lambda_preserve_tx * preservation_loss

        # Logging
        self.log('val_loss', loss)
        self.log('val_alignment_loss', alignment_loss)

        self.knn_classifier.update(test_features=z_tx_aligned, test_targets=labels)

        if self.current_epoch == 0 :
            self.knn_classifier_input.update(test_features=z_tx_original, test_targets=labels)

        return loss

    def on_validation_epoch_end(self):
        knn_top1, knn_top5 = self.knn_classifier.compute()
        self.log('knn_top1', knn_top1)
        self.log('knn_top5', knn_top5)
        if self.current_epoch == 0 :
            knn_top1, knn_top5 = self.knn_classifier_input.compute()
            self.log('original_knn_top1', knn_top1)
            self.log('original_knn_top5', knn_top5)
    
    def predition_step(self, batch, batch_idx):
        return self.tx_encoder(batch) if self.tx_encoder is not None else batch
    
    def configure_optimizers(self):
        parameters = []
        if self.ph_encoder_train_mode == True:
            parameters.append({"params": self.ph_encoder.parameters(), "lr": self.hparams.ph_encoder_lr})
        if self.tx_encoder_train_mode == True:
            parameters.append({"params": self.tx_encoder.parameters(), "lr": self.hparams.tx_encoder_lr})

        optimizer = torch.optim.SGD(parameters, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)

        # Set up the Cosine Annealing scheduler
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.n_epochs,                  # Total number of iterations or epochs
            eta_min=self.hparams.min_lr                # Minimum learning rate
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }
    
