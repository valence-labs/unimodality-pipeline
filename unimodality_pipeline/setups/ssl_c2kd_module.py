import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from typing import Dict, Any
from ..models.mlp import MLP
from ..eval.knn import WeightedKNNClassifier

class SslC2KDModule(LightningModule):
    def __init__(self, hparams: Dict[str, Any]):
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

        self.tx_encoder_train_mode = self.tx_encoder is not None
        self.ph_encoder_train_mode = self.ph_encoder is not None  # Both encoders are trainable

        # Projection heads
        self.proj_size = self.hparams.proj_size  # Size of the projection head outputs
        self.tx_proj = nn.Linear(self.hparams.tx_output_size, self.proj_size)
        self.ph_proj = nn.Linear(self.hparams.ph_output_size, self.proj_size)

        # Loss weights 
        self.lambda_preserve_tx = self.hparams.lambda_preserve_tx
        self.lambda_kl_tx = self.hparams.lambda_kl_tx
        self.lambda_kl_ph = self.hparams.lambda_kl_ph


        # Loss function
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')

        # Temperature for knowledge distillation
        self.T = self.hparams.temperature_KD  # Temperature for softening probabilities

        # KNN classifiers (optional)
        self.knn_classifier = WeightedKNNClassifier()
        self.knn_classifier_input = WeightedKNNClassifier()

    def linear_CKA(self, z1, z2):
        """Compute the linear CKA between two sets of representations."""
        # Ensure the inputs are centered
        z1 = z1 - z1.mean(dim=0, keepdim=True)
        z2 = z2 - z2.mean(dim=0, keepdim=True)

        numerator = torch.norm(torch.mm(z1.T, z2), p='fro') ** 2
        denominator = torch.norm(torch.mm(z1.T, z1), p='fro') * torch.norm(torch.mm(z2.T, z2), p='fro')
        cka = numerator / denominator
        return cka

    def forward(self, x):
        # Forward method
        return self.tx_encoder(x) if self.tx_encoder is not None else self.ph_encoder(x)

    def training_step(self, batch, batch_idx):
        x_tx, x_ph, labels = batch  # Remove labels
        x_tx = x_tx.to(self.device)
        x_ph = x_ph.to(self.device)
        labels = labels.to(self.device)

        # Student outputs
        z_tx = self.tx_encoder(x_tx) if self.tx_encoder is not None else x_tx
        stu_logits = self.tx_proj(z_tx)  # Student projection

        # Teacher outputs
        z_ph = self.ph_encoder(x_ph) if self.ph_encoder is not None else x_ph
        tea_logits = self.ph_proj(z_ph)  # Teacher projection

        # Bidirectional KL Divergence Losses
        kl_loss_stu_to_tea = self.criterion_kl(
            F.log_softmax(stu_logits / self.T, dim=-1),
            F.softmax(tea_logits.detach() / self.T, dim=-1)
        ) * (self.T ** 2)

        kl_loss_tea_to_stu = self.criterion_kl(
            F.log_softmax(tea_logits / self.T, dim=-1),
            F.softmax(stu_logits.detach() / self.T, dim=-1)
        ) * (self.T ** 2)

        # Self-preservation loss using CKA
        cka_value = self.linear_CKA(z_tx, x_tx)
        preservation_loss = -cka_value  # Negative because we want to maximize CKA

        # Total Loss
        loss = self.lambda_kl_tx * kl_loss_stu_to_tea + self.lambda_kl_ph * kl_loss_tea_to_stu + self.lambda_preserve_tx * preservation_loss

        # Logging
        self.log('train_loss', loss)
        self.log('kl_loss_stu_to_tea', kl_loss_stu_to_tea)
        self.log('kl_loss_tea_to_stu', kl_loss_tea_to_stu)
        self.log('preservation_loss', preservation_loss)

        # KNN classifier updates (optional)
        self.knn_classifier.update(train_features=z_tx, train_targets=labels)
        if self.current_epoch == 0:
            self.knn_classifier_input.update(train_features=x_tx, train_targets=labels)

        return loss

    def validation_step(self, batch, batch_idx):
        x_tx, x_ph, labels = batch  # Remove labels
        x_tx = x_tx.to(self.device)
        x_ph = x_ph.to(self.device)
        labels = labels.to(self.device)

        # Student outputs
        z_tx = self.tx_encoder(x_tx) if self.tx_encoder is not None else x_tx
        stu_logits = self.tx_proj(z_tx)  # Student projection

        # Teacher outputs
        z_ph = self.ph_encoder(x_ph) if self.ph_encoder is not None else x_ph
        tea_logits = self.ph_proj(z_ph)  # Teacher projection

        # Bidirectional KL Divergence Losses
        kl_loss_stu_to_tea = self.criterion_kl(
            F.log_softmax(stu_logits / self.T, dim=-1),
            F.softmax(tea_logits.detach() / self.T, dim=-1)
        ) * (self.T ** 2)

        kl_loss_tea_to_stu = self.criterion_kl(
            F.log_softmax(tea_logits / self.T, dim=-1),
            F.softmax(stu_logits.detach() / self.T, dim=-1)
        ) * (self.T ** 2)

        # Self-preservation loss using CKA
        cka_value = self.linear_CKA(z_tx, x_tx)
        preservation_loss = -cka_value  # Negative because we want to maximize CKA

        # Total Loss
        loss = self.lambda_kl_tx * kl_loss_stu_to_tea + self.lambda_kl_ph * kl_loss_tea_to_stu + self.lambda_preserve_tx * preservation_loss

        # Logging
        self.log('val_loss', loss)
        self.log('val_kl_loss_stu_to_tea', kl_loss_stu_to_tea)
        self.log('val_kl_loss_tea_to_stu', kl_loss_tea_to_stu)
        self.log('val_preservation_loss', preservation_loss)

        # KNN classifier updates (optional)
        self.knn_classifier.update(test_features=z_tx, test_targets=labels)
        if self.current_epoch == 0:
            self.knn_classifier_input.update(test_features=x_tx, test_targets=labels)

        return loss

    def on_validation_epoch_end(self):
        # KNN evaluation (optional)
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
            parameters.extend([
                {"params": self.ph_encoder.parameters(), "lr": self.hparams.ph_encoder_lr},
                {"params": self.ph_proj.parameters(), "lr": self.hparams.ph_classifier_lr},
            ])
        if self.tx_encoder_train_mode:
            parameters.extend([
                {"params": self.tx_encoder.parameters(), "lr": self.hparams.tx_encoder_lr},
                {"params": self.tx_proj.parameters(), "lr": self.hparams.tx_classifier_lr},
            ])

        optimizer = torch.optim.SGD(parameters, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)

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
