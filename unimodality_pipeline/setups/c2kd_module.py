import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from typing import Dict, Any
from scipy.stats import kendalltau
import numpy as np
from copy import deepcopy
from ..models.mlp import MLP
from ..eval.knn import WeightedKNNClassifier

class C2KDModule(LightningModule):
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
        self.ph_encoder_train_mode = self.ph_encoder is not None

        # Classifiers
        self.num_classes = 4056 #self.hparams.num_classes
        self.tx_classifier = nn.Linear(self.hparams.tx_output_size, self.num_classes)
        self.ph_classifier = nn.Linear(self.hparams.ph_output_size, self.num_classes)

        # Projection heads for OFSD
        self.proj_size = self.num_classes #self.hparams.proj_size  # Size of the projection head outputs
        self.tx_proj = nn.Linear(self.hparams.tx_output_size, self.proj_size)
        self.ph_proj = nn.Linear(self.hparams.ph_output_size, self.proj_size)

        # OFSD parameters
        self.krc_threshold = self.hparams.krc_threshold  # Threshold for Kendall Rank Correlation

        # Loss functions
        self.criterion_ce = nn.CrossEntropyLoss(reduction='none')  # Reduction set to 'none' to compute per-sample losses
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')
        self.criterion_kl_none = nn.KLDivLoss(reduction='none')

        # Temperature for knowledge distillation
        self.T = self.hparams.temperature_KD  # Temperature for softening probabilities

        # KNN classifiers (optional)
        self.knn_classifier = WeightedKNNClassifier()
        self.knn_classifier_input = WeightedKNNClassifier()

    def forward(self, x):
        # Forward method to align with your existing code
        return self.tx_encoder(x) if self.tx_encoder is not None else self.ph_encoder(x)

    def nt_kl_divergence(self, logits_s, logits_t, labels, mask):
        """Compute KL divergence over non-target classes for selected samples."""
        # logits_s and logits_t: [batch_size, num_classes]
        # labels: [batch_size]
        # mask: [batch_size], values are 0 or 1

        batch_size, num_classes = logits_s.size()
        device = logits_s.device

        # Create mask to zero out the target class
        target_mask = F.one_hot(labels, num_classes).float().to(device)  # [batch_size, num_classes]

        # Zero out the target class in logits
        logits_s_no_target = logits_s * (1 - target_mask)
        logits_t_no_target = logits_t * (1 - target_mask)

        # Apply mask to select samples
        mask = mask.view(-1, 1)  # [batch_size, 1]
        logits_s_selected = logits_s_no_target[mask.squeeze() == 1]
        logits_t_selected = logits_t_no_target[mask.squeeze() == 1]

        if logits_s_selected.size(0) == 0:
            return torch.tensor(0.0, device=device)

        # Compute softmax over non-target classes
        log_probs_s = F.log_softmax(logits_s_selected / self.T, dim=1)
        probs_t = F.softmax(logits_t_selected.detach() / self.T, dim=1)

        # Compute KL divergence
        kl_div = self.criterion_kl(log_probs_s, probs_t) * (self.T ** 2)

        return kl_div

    def training_step(self, batch, batch_idx):
        x_tx, x_ph, labels = batch
        labels = labels.to(self.device)

        # Student outputs
        z_tx = self.tx_encoder(x_tx) if self.tx_encoder is not None else x_tx
        logits_tx = self.tx_classifier(z_tx)
        stu_fit = self.tx_proj(z_tx)  # Student projection
        stu_logits = stu_fit

        # Teacher outputs
        z_ph = self.ph_encoder(x_ph) if self.ph_encoder is not None else x_ph
        logits_ph = self.ph_classifier(z_ph)
        tea_fit = self.ph_proj(z_ph)  # Teacher projection
        tea_logits = tea_fit

        # Cross-Entropy Losses
        ce_loss_tx = self.criterion_ce(logits_tx, labels).mean()
        ce_loss_ph = self.criterion_ce(logits_ph, labels).mean()

        # KL Divergence Losses (Bidirectional)
        kl_loss_t_t_im = self.criterion_kl(
            F.log_softmax(tea_logits / self.T, dim=-1),
            F.softmax(logits_ph.detach() / self.T, dim=-1)
        ) * (self.T ** 2)
        kl_loss_t_im_t = self.criterion_kl(
            F.log_softmax(logits_ph / self.T, dim=-1),
            F.softmax(tea_logits.detach() / self.T, dim=-1)
        ) * (self.T ** 2)
        kl_loss_s_s_im = self.criterion_kl(
            F.log_softmax(stu_logits / self.T, dim=-1),
            F.softmax(logits_tx.detach() / self.T, dim=-1)
        ) * (self.T ** 2)
        kl_loss_s_im_s = self.criterion_kl(
            F.log_softmax(logits_tx / self.T, dim=-1),
            F.softmax(stu_logits.detach() / self.T, dim=-1)
        ) * (self.T ** 2)

        # Compute Kendall Rank Correlation (KRC) per sample
        batch_size = tea_logits.size(0)
        krc_values = np.empty(batch_size)
        for i in range(batch_size):
            tea_logits_i = tea_logits[i].detach().cpu().numpy()
            stu_logits_i = stu_logits[i].detach().cpu().numpy()
            krc_values[i] = kendalltau(tea_logits_i, stu_logits_i)[0]
        krc_tensor = torch.from_numpy(krc_values).to(self.device)
        krc_mean = krc_tensor.mean()
        mask = (krc_tensor > self.krc_threshold).float()
        mask_sum = mask.sum()

        # OFSD KL Divergence Losses
        if mask_sum > 0:
            kl_loss_st_s_t = self.nt_kl_divergence(stu_logits, tea_logits.detach(), labels, mask)
            kl_loss_st_t_s = self.nt_kl_divergence(tea_logits, stu_logits.detach(), labels, mask)
        else:
            kl_loss_st_s_t = torch.tensor(0.0, device=self.device)
            kl_loss_st_t_s = torch.tensor(0.0, device=self.device)

        # Total Losses
        tmp1 = ce_loss_ph + kl_loss_t_t_im + kl_loss_st_t_s + kl_loss_t_im_t
        tmp2 = ce_loss_tx + kl_loss_s_s_im + kl_loss_st_s_t + kl_loss_s_im_s
        loss = tmp1 + tmp2

        # Logging
        self.log('train_loss', loss)
        self.log('ce_loss_tx', ce_loss_tx)
        self.log('ce_loss_ph', ce_loss_ph)
        self.log('kl_loss_t_t_im', kl_loss_t_t_im)
        self.log('kl_loss_t_im_t', kl_loss_t_im_t)
        self.log('kl_loss_s_s_im', kl_loss_s_s_im)
        self.log('kl_loss_s_im_s', kl_loss_s_im_s)
        self.log('kl_loss_st_s_t', kl_loss_st_s_t)
        self.log('kl_loss_st_t_s', kl_loss_st_t_s)
        self.log('krc_mean', krc_mean)
        self.log('mask_ratio', mask_sum / batch_size)

        # KNN classifier updates (optional)
        self.knn_classifier.update(train_features=z_tx, train_targets=labels)
        if self.current_epoch == 0:
            self.knn_classifier_input.update(train_features=x_tx, train_targets=labels)

        return loss

    def validation_step(self, batch, batch_idx):
        x_tx, x_ph, labels = batch
        labels = labels.to(self.device)

        # Student outputs
        z_tx = self.tx_encoder(x_tx) if self.tx_encoder is not None else x_tx
        logits_tx = self.tx_classifier(z_tx)
        stu_fit = self.tx_proj(z_tx)  # Student projection
        stu_logits = stu_fit

        # Teacher outputs
        z_ph = self.ph_encoder(x_ph) if self.ph_encoder is not None else x_ph
        logits_ph = self.ph_classifier(z_ph)
        tea_fit = self.ph_proj(z_ph)  # Teacher projection
        tea_logits = tea_fit

        # Cross-Entropy Losses
        ce_loss_tx = self.criterion_ce(logits_tx, labels).mean()
        ce_loss_ph = self.criterion_ce(logits_ph, labels).mean()

        # KL Divergence Losses (Bidirectional)
        kl_loss_t_t_im = self.criterion_kl(
            F.log_softmax(tea_logits / self.T, dim=-1),
            F.softmax(logits_ph.detach() / self.T, dim=-1)
        ) * (self.T ** 2)
        kl_loss_t_im_t = self.criterion_kl(
            F.log_softmax(logits_ph / self.T, dim=-1),
            F.softmax(tea_logits.detach() / self.T, dim=-1)
        ) * (self.T ** 2)
        kl_loss_s_s_im = self.criterion_kl(
            F.log_softmax(stu_logits / self.T, dim=-1),
            F.softmax(logits_tx.detach() / self.T, dim=-1)
        ) * (self.T ** 2)
        kl_loss_s_im_s = self.criterion_kl(
            F.log_softmax(logits_tx / self.T, dim=-1),
            F.softmax(stu_logits.detach() / self.T, dim=-1)
        ) * (self.T ** 2)

        # Compute Kendall Rank Correlation (KRC) per sample
        batch_size = tea_logits.size(0)
        krc_values = np.empty(batch_size)
        for i in range(batch_size):
            tea_logits_i = tea_logits[i].detach().cpu().numpy()
            stu_logits_i = stu_logits[i].detach().cpu().numpy()
            krc_values[i] = kendalltau(tea_logits_i, stu_logits_i)[0]
        krc_tensor = torch.from_numpy(krc_values).to(self.device)
        krc_mean = krc_tensor.mean()
        mask = (krc_tensor > self.krc_threshold).float()
        mask_sum = mask.sum()

        # OFSD KL Divergence Losses
        if mask_sum > 0:
            kl_loss_st_s_t = self.nt_kl_divergence(stu_logits, tea_logits.detach(), labels, mask)
            kl_loss_st_t_s = self.nt_kl_divergence(tea_logits, stu_logits.detach(), labels, mask)
        else:
            kl_loss_st_s_t = torch.tensor(0.0, device=self.device)
            kl_loss_st_t_s = torch.tensor(0.0, device=self.device)

        # Total Losses
        tmp1 = ce_loss_ph + kl_loss_t_t_im + kl_loss_st_t_s + kl_loss_t_im_t
        tmp2 = ce_loss_tx + kl_loss_s_s_im + kl_loss_st_s_t + kl_loss_s_im_s
        loss = tmp1 + tmp2

        # Logging
        self.log('val_loss', loss)
        self.log('val_ce_loss_tx', ce_loss_tx)
        self.log('val_ce_loss_ph', ce_loss_ph)
        self.log('val_kl_loss_t_t_im', kl_loss_t_t_im)
        self.log('val_kl_loss_t_im_t', kl_loss_t_im_t)
        self.log('val_kl_loss_s_s_im', kl_loss_s_s_im)
        self.log('val_kl_loss_s_im_s', kl_loss_s_im_s)
        self.log('val_kl_loss_st_s_t', kl_loss_st_s_t)
        self.log('val_kl_loss_st_t_s', kl_loss_st_t_s)
        self.log('val_krc_mean', krc_mean)
        self.log('val_mask_ratio', mask_sum / batch_size)

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
                {"params": self.ph_classifier.parameters(), "lr": self.hparams.ph_classifier_lr},
                {"params": self.ph_proj.parameters(), "lr": self.hparams.ph_classifier_lr},
            ])
        if self.tx_encoder_train_mode:
            parameters.extend([
                {"params": self.tx_encoder.parameters(), "lr": self.hparams.tx_encoder_lr},
                {"params": self.tx_classifier.parameters(), "lr": self.hparams.tx_classifier_lr},
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
