import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Dict, Any
from pytorch_lightning import LightningModule
from ..models.mlp import MLP
from ..eval.knn import WeightedKNNClassifier

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class ShakeModule(LightningModule):
    def __init__(
        self, 
        hparams: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        # Student Encoder (Transcriptomics)
        self.tx_encoder = None if self.hparams.tx_disabled else MLP(
            self.hparams.tx_input_size, 
            self.hparams.tx_hidden_dims,
            self.hparams.tx_activations, 
            self.hparams.tx_output_size,
            self.hparams.tx_output_activation
        )

        # Teacher Encoder Backbone (Phenomics)
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

        # Classifiers / Heads
        self.num_classes = 4056 #self.hparams.num_classes  # Number of classes in your dataset

        # Student Classifier Head
        self.tx_classifier = nn.Linear(self.hparams.tx_output_size, self.num_classes)
        self.ph_classifier = nn.Linear(self.hparams.ph_output_size, self.num_classes)

        # Teacher Shadow Head (Proxy Teacher)
        self.ph_shadow_head = nn.Linear(self.hparams.ph_output_size, self.num_classes)

        # Teacher Classifier Head (Fixed, Pretrained Teacher)
        # Assuming the teacher classifier is pre-trained and fixed
        # If you have a pre-trained classifier, you can load it here
        #pretrained_ph_ckpt_path = './experiments/save_ph_encoder/save_ph_encoder/epoch=03-val_loss=5.42-val_acc=0.00.ckpt'
        pretrained_ph_ckpt_path = self.hparams.pretrained_weights

        ckpt = torch.load(pretrained_ph_ckpt_path, map_location=self.device)
        # Extract the state dict
        state_dict = ckpt['state_dict']
        # Load ph_encoder weights
        ph_encoder_state_dict = {k.replace('ph_encoder.', ''): v for k, v in state_dict.items() if k.startswith('ph_encoder.')}
        self.ph_encoder.load_state_dict(ph_encoder_state_dict)
        # Load ph_classifier weights
        ph_classifier_state_dict = {k.replace('ph_classifier.', ''): v for k, v in state_dict.items() if k.startswith('ph_classifier.')}
        self.ph_classifier.load_state_dict(ph_classifier_state_dict)
        # Freeze the pre-trained teacher classifier
        for param in self.ph_classifier.parameters():
            param.requires_grad = False
        for param in self.ph_encoder.parameters():
            param.requires_grad = False
        


        # Loss Functions
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')

        # Temperature for knowledge distillation
        self.T = self.hparams.temperature_KD  # Temperature for softening probabilities

        # Hyperparameters for loss weights
        self.alpha = self.hparams.alpha  # Weight for KL divergence loss
        self.beta = self.hparams.beta   # Weight for reversed distillation loss

        # KNN Classifiers (Optional)
        self.knn_classifier = WeightedKNNClassifier()
        self.knn_classifier_input = WeightedKNNClassifier()

    def forward(self, x):
        # Forward method to align with your existing code
        return self.tx_encoder(x) if self.tx_encoder is not None else self.ph_encoder(x)

    def training_step(self, batch, batch_idx):
        x_tx, x_ph, labels = batch
        labels = labels.to(self.device)

        # Student Forward Pass
        z_tx = self.tx_encoder(x_tx) if self.tx_encoder is not None else x_tx
        logits_tx = self.tx_classifier(z_tx)

        # Teacher Forward Pass (Backbone and Shadow Head)
        with torch.no_grad():
            z_ph = self.ph_encoder(x_ph) if self.ph_encoder is not None else x_ph
            logits_ph_teacher = self.ph_classifier(z_ph)  # Fixed teacher classifier output

        logits_ph_shadow = self.ph_shadow_head(z_ph)  # Proxy teacher output (shadow head)

        # Cross-Entropy Loss for Student
        loss_ce_student = self.criterion_ce(logits_tx, labels)

        # Distillation Loss: From Pretrained Teacher to Proxy Teacher
        loss_distill_teacher_to_proxy = self.criterion_kl(
            F.log_softmax(logits_ph_shadow / self.T, dim=1),
            F.softmax(logits_ph_teacher / self.T, dim=1)
        ) * (self.T ** 2)

        # Distillation Loss: From Proxy Teacher to Student
        loss_distill_proxy_to_student = self.criterion_kl(
            F.log_softmax(logits_tx / self.T, dim=1),
            F.softmax(logits_ph_shadow.detach() / self.T, dim=1)
        ) * (self.T ** 2)

        # Reversed Distillation Loss: From Student to Proxy Teacher
        loss_reversed_distill_student_to_proxy = self.criterion_kl(
            F.log_softmax(logits_ph_shadow / self.T, dim=1),
            F.softmax(logits_tx.detach() / self.T, dim=1)
        ) * (self.T ** 2)

        # Total Loss
        loss = loss_ce_student \
            + self.alpha * loss_distill_proxy_to_student \
            + self.beta * loss_reversed_distill_student_to_proxy \
            + loss_distill_teacher_to_proxy  # Ensure the proxy teacher learns from the fixed teacher

        # Logging
        self.log('train_loss', loss)
        self.log('loss_ce_student', loss_ce_student)
        self.log('loss_distill_proxy_to_student', loss_distill_proxy_to_student)
        self.log('loss_reversed_distill_student_to_proxy', loss_reversed_distill_student_to_proxy)
        self.log('loss_distill_teacher_to_proxy', loss_distill_teacher_to_proxy)

        # KNN Classifier Updates (Optional)
        self.knn_classifier.update(train_features=z_tx.detach(), train_targets=labels)
        if self.current_epoch == 0:
            self.knn_classifier_input.update(train_features=x_tx.detach(), train_targets=labels)

        return loss

    def validation_step(self, batch, batch_idx):
        x_tx, x_ph, labels = batch
        labels = labels.to(self.device)

        # Student Forward Pass
        z_tx = self.tx_encoder(x_tx) if self.tx_encoder is not None else x_tx
        logits_tx = self.tx_classifier(z_tx)

        # Teacher Forward Pass (Backbone and Shadow Head)
        with torch.no_grad():
            z_ph = self.ph_encoder(x_ph) if self.ph_encoder is not None else x_ph
            logits_ph_teacher = self.ph_classifier(z_ph)  # Fixed teacher classifier output

        logits_ph_shadow = self.ph_shadow_head(z_ph)  # Proxy teacher output (shadow head)

        # Cross-Entropy Loss for Student
        loss_ce_student = self.criterion_ce(logits_tx, labels)

        # Distillation Loss: From Pretrained Teacher to Proxy Teacher
        loss_distill_teacher_to_proxy = self.criterion_kl(
            F.log_softmax(logits_ph_shadow / self.T, dim=1),
            F.softmax(logits_ph_teacher / self.T, dim=1)
        ) * (self.T ** 2)

        # Distillation Loss: From Proxy Teacher to Student
        loss_distill_proxy_to_student = self.criterion_kl(
            F.log_softmax(logits_tx / self.T, dim=1),
            F.softmax(logits_ph_shadow.detach() / self.T, dim=1)
        ) * (self.T ** 2)

        # Reversed Distillation Loss: From Student to Proxy Teacher
        loss_reversed_distill_student_to_proxy = self.criterion_kl(
            F.log_softmax(logits_ph_shadow / self.T, dim=1),
            F.softmax(logits_tx.detach() / self.T, dim=1)
        ) * (self.T ** 2)

        # Total Loss
        loss = loss_ce_student \
            + self.alpha * loss_distill_proxy_to_student \
            + self.beta * loss_reversed_distill_student_to_proxy \
            + loss_distill_teacher_to_proxy

        # Logging
        self.log('val_loss', loss)
        self.log('val_loss_ce_student', loss_ce_student)
        self.log('val_loss_distill_proxy_to_student', loss_distill_proxy_to_student)
        self.log('val_loss_reversed_distill_student_to_proxy', loss_reversed_distill_student_to_proxy)
        self.log('val_loss_distill_teacher_to_proxy', loss_distill_teacher_to_proxy)

        # KNN Classifier Updates (Optional)
        self.knn_classifier.update(test_features=z_tx.detach(), test_targets=labels)
        if self.current_epoch == 0:
            self.knn_classifier_input.update(test_features=x_tx.detach(), test_targets=labels)

        return loss

    def on_validation_epoch_end(self):
        # KNN Evaluation (Optional)
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
                {"params": self.ph_shadow_head.parameters(), "lr": self.hparams.ph_classifier_lr},
            ])
        if self.tx_encoder_train_mode:
            parameters.extend([
                {"params": self.tx_encoder.parameters(), "lr": self.hparams.tx_encoder_lr},
                {"params": self.tx_classifier.parameters(), "lr": self.hparams.tx_classifier_lr},
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
