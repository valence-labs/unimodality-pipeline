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

class PhSupervisedModule(LightningModule):
    def __init__(
        self, 
        hparams: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        # Phenomics Encoder
        self.ph_encoder = MLP(
            self.hparams.ph_input_size, 
            self.hparams.ph_hidden_dims,
            self.hparams.ph_activations, 
            self.hparams.ph_output_size,
            self.hparams.ph_output_activation
        )
        
        # Phenomics Classifier
        self.num_classes = 4056 #self.hparams.num_classes  # Number of classes in your dataset
        self.ph_classifier = nn.Linear(self.hparams.ph_output_size, self.num_classes)
        
        # Loss Function
        self.criterion_ce = nn.CrossEntropyLoss()
        
        # KNN Classifiers (Optional)
        self.knn_classifier = WeightedKNNClassifier()
        
    def forward(self, x):
        # Forward method
        z_ph = self.ph_encoder(x)
        logits_ph = self.ph_classifier(z_ph)
        return logits_ph
    
    def training_step(self, batch, batch_idx):
        # Since your data module provides (x_tx, x_ph, labels), we can ignore x_tx
        x_tx, x_ph, labels = batch
        labels = labels.to(self.device)
        
        # Forward Pass
        z_ph = self.ph_encoder(x_ph)
        logits_ph = self.ph_classifier(z_ph)
        
        # Loss Computation
        loss = self.criterion_ce(logits_ph, labels)
        
        # Logging
        self.log('train_loss', loss)
        
        # KNN Classifier Updates (Optional)
        self.knn_classifier.update(train_features=z_ph.detach(), train_targets=labels)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x_tx, x_ph, labels = batch
        labels = labels.to(self.device)
        
        # Forward Pass
        z_ph = self.ph_encoder(x_ph)
        logits_ph = self.ph_classifier(z_ph)
        
        # Loss Computation
        loss = self.criterion_ce(logits_ph, labels)
        
        # Logging
        self.log('val_loss', loss)
        
        # KNN Classifier Updates (Optional)
        self.knn_classifier.update(test_features=z_ph.detach(), test_targets=labels)
        
        return loss
    
    def on_validation_epoch_end(self):
        # KNN Evaluation (Optional)
        knn_top1, knn_top5 = self.knn_classifier.compute()
        self.log('knn_top1', knn_top1)
        self.log('knn_top5', knn_top5)
        
    def configure_optimizers(self):
        parameters = [
            {"params": self.ph_encoder.parameters(), "lr": self.hparams.ph_encoder_lr},
            {"params": self.ph_classifier.parameters(), "lr": self.hparams.ph_classifier_lr},
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
