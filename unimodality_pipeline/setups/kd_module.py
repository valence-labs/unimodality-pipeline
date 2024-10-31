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

class KDModule(LightningModule):
    def __init__(
        self, 
        hparams: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        # Student Encoder (Transcriptomics)
        if self.hparams.tx_disabled:
            raise ValueError("Tx encoder cannot be disabled in KDModule.")
        self.tx_encoder = MLP(
            self.hparams.tx_input_size, 
            self.hparams.tx_hidden_dims,
            self.hparams.tx_activations, 
            self.hparams.tx_output_size,
            self.hparams.tx_output_activation
        )

        # Teacher Encoder is disabled
        self.ph_encoder = None  # We won't use ph_encoder
        self.ph_encoder_train_mode = False

        # Classifiers
        self.num_classes = 4056 #self.hparams.num_classes  # Number of classes in your dataset

        # Student Classifier Head
        self.tx_classifier = nn.Linear(self.hparams.tx_output_size, self.num_classes)

        # Teacher Classifier Head (Fixed, Pretrained Teacher)
        self.ph_classifier = nn.Linear(self.hparams.ph_input_size, self.num_classes)
        pretrained_ph_ckpt_path = './experiments/save_ph_encoder/save_ph_encoder/epoch=03-val_loss=5.42-val_acc=0.00.ckpt'


        # Load the pretrained ph_classifier
        ckpt = torch.load(pretrained_ph_ckpt_path, map_location=self.device)
        # Extract the state dict
        state_dict = ckpt['state_dict']
        # Load ph_classifier weights
        ph_classifier_state_dict = {k.replace('ph_classifier.', ''): v for k, v in state_dict.items() if k.startswith('ph_classifier.')}
        self.ph_classifier.load_state_dict(ph_classifier_state_dict)
        # Freeze the pre-trained teacher classifier
        for param in self.ph_classifier.parameters():
            param.requires_grad = False


        # Loss Functions
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')

        # Temperature for knowledge distillation
        self.T = self.hparams.temperature_KD  # Temperature for softening probabilities

        # Weight for KL divergence loss
        self.alpha = self.hparams.alpha  # Hyperparameter to weight the KD loss

        # KNN Classifiers (Optional)
        self.knn_classifier = WeightedKNNClassifier()
        self.knn_classifier_input = WeightedKNNClassifier()

    def forward(self, x):
        # Forward method for student
        z_tx = self.tx_encoder(x)
        return z_tx

    def training_step(self, batch, batch_idx):
        x_tx, x_ph, labels = batch
        labels = labels.to(self.device)

        # Student Forward Pass
        z_tx = self.tx_encoder(x_tx)
        logits_tx = self.tx_classifier(z_tx)

        # Teacher Forward Pass (No encoder)
        with torch.no_grad():
            logits_ph_teacher = self.ph_classifier(x_ph)  # x_ph is directly used as input to ph_classifier

        # Cross-Entropy Loss for Student
        loss_ce_student = self.criterion_ce(logits_tx, labels)

        # KL Divergence Loss between student and teacher logits (teacher logits detached)
        loss_kd = self.criterion_kl(
            F.log_softmax(logits_tx / self.T, dim=1),
            F.softmax(logits_ph_teacher.detach() / self.T, dim=1)
        ) * (self.T ** 2)

        # Total Loss
        loss = loss_ce_student + self.alpha * loss_kd

        # Logging
        self.log('train_loss', loss)
        self.log('loss_ce_student', loss_ce_student)
        self.log('loss_kd', loss_kd)

        # KNN Classifier Updates (Optional)
        self.knn_classifier.update(train_features=z_tx.detach(), train_targets=labels)
        if self.current_epoch == 0:
            self.knn_classifier_input.update(train_features=x_tx.detach(), train_targets=labels)

        return loss

    def validation_step(self, batch, batch_idx):
        x_tx, x_ph, labels = batch
        labels = labels.to(self.device)

        # Student Forward Pass
        z_tx = self.tx_encoder(x_tx)
        logits_tx = self.tx_classifier(z_tx)

        # Teacher Forward Pass (No encoder)
        with torch.no_grad():
            logits_ph_teacher = self.ph_classifier(x_ph)  # x_ph is directly used as input to ph_classifier

        # Cross-Entropy Loss for Student
        loss_ce_student = self.criterion_ce(logits_tx, labels)

        # KL Divergence Loss between student and teacher logits (teacher logits detached)
        loss_kd = self.criterion_kl(
            F.log_softmax(logits_tx / self.T, dim=1),
            F.softmax(logits_ph_teacher.detach() / self.T, dim=1)
        ) * (self.T ** 2)

        # Total Loss
        loss = loss_ce_student + self.alpha * loss_kd

        # Logging
        self.log('val_loss', loss)
        self.log('val_loss_ce_student', loss_ce_student)
        self.log('val_loss_kd', loss_kd)

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
        parameters = [
            {"params": self.tx_encoder.parameters(), "lr": self.hparams.tx_encoder_lr},
            {"params": self.tx_classifier.parameters(), "lr": self.hparams.tx_classifier_lr},
        ]

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
