"""Stores contrastive loss functions."""

import math
from typing import Union, Dict

import torch
import torch.nn as nn
from torch import distributed as torch_dist
import torch.nn.functional as F
import numpy as np
from einops import reduce

from torchmetrics import Accuracy

from .cloom_utils import cloob, infoLOOB_loss, clip, hopfield_clip
from .util import all_gather, world_size, rank, eye_rank, compute_similarity_matrix


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, final_teacher_temp,
                 warmup_teacher_temp_epochs, nepochs,
                 student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, final_teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * final_teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        student_output: (B*ncrops, out_dim)
        teacher_output: (B*2, out_dim)
        """
        student_out = student_output/self.student_temp
        student_out = student_out.chunk(self.ncrops) # global views + local views

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output-self.center)/temp, dim=-1)
        teacher_out = teacher_out.chunk(2) # global views

        total_loss = n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq: # skip cases where student and teacher operate on the same view
                    continue
                loss = reduce(-q*F.log_softmax(student_out[v], dim=-1), 'b o -> b', 'sum')
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = reduce(teacher_output, 'b o -> 1 o', 'mean')

        self.center = self.center * self.center_momentum + \
                      batch_center * (1 - self.center_momentum)
    

class InfoLOOBLoss(torch.nn.Module):
    """Implementation of the InfoLOOB loss."""

    def __init__(
            self,
            gather_distributed: bool = False,
            normalize: bool = True,
            inv_tau: float = 4.6052
    ):
        super().__init__()
        self.normalize = normalize
        self.inv_tau = inv_tau
        if gather_distributed and not torch_dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

    def forward(
            self,
            img_rep: torch.Tensor,
            mol_rep: torch.Tensor,
    ) -> (torch.Tensor, Dict):
        """Forward pass of the InfoLOOB loss."""
        if self.normalize:
            img_rep = nn.functional.normalize(img_rep, dim=1)
            mol_rep = nn.functional.normalize(mol_rep, dim=1)

        if self.gather_distributed and world_size() > 1:
            img_rep_all = all_gather(img_rep, 0, None)
            mol_rep_all = all_gather(mol_rep, 0, None)
        else:
            img_rep_all = img_rep
            mol_rep_all = mol_rep
        loss0, metrics_1 = self._loss(img_rep, mol_rep, img_rep_all, mol_rep_all, "mol_from_img")
        loss1, metrics_2 = self._loss(mol_rep, img_rep, mol_rep_all, img_rep_all, 'img_from_mol')
        all_metrics = dict()
        all_metrics.update(metrics_1)
        all_metrics.update(metrics_2)
        return 0.5 * (loss0 + loss1), all_metrics

    def _loss(self, out0, out1, out0_all, out1_all, type):
        """Calculates InfoLOOB loss for out0 with respect to its positives in out1
        and the negatives in out1, out0_all, and out1_all.
        """
        batch_size = out0.shape[0]
        num_classes = world_size() * batch_size

        acc = Accuracy(task='multiclass', num_classes=num_classes).to(out0.device)
        acc_10 = Accuracy(task='multiclass', num_classes=num_classes, top_k=10).to(out0.device)
        acc_5 = Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(out0.device)
        top_k = int(math.ceil(num_classes * .001))
        acc_point_1_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(out0.device)
        top_k = int(math.ceil(num_classes * .01))
        acc_1_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(out0.device)
        top_k = int(math.ceil(num_classes * .05))
        acc_5_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(out0.device)
        top_k = int(math.ceil(num_classes * .1))
        acc_10_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(out0.device)

        accuracy_dict = {
            'top_1_accuracy': acc,
            "top_10_accuracy": acc_10,
            "top_5_accuracy": acc_5,
            "top_0.1_percent_accuracy": acc_point_1_percent,
            "top_1_percent_accuracy": acc_1_percent,
            "top_5_percent_accuracy": acc_5_percent,
            "top_10_percent_accuracy": acc_10_percent,
        }
        metrics_dict = dict()
        labels = torch.arange(batch_size, device=out0.device, dtype=torch.long)
        if self.gather_distributed and world_size() > 1:
            diag_mask = eye_rank(batch_size, device=out0.device)
        else:
            diag_mask = torch.eye(batch_size, device=out0.device, dtype=torch.bool)
        identity = torch.eye(out0.shape[0]) > 0.5
        i = identity.to(out0.device)
        loss_mol = infoLOOB_loss(out0, out1, i, inv_tau=self.inv_tau)
        loss_dose = infoLOOB_loss(out1, out0, i, inv_tau=self.inv_tau)
        loss = loss_mol + loss_dose
        logits = torch.einsum("nc,mc->nm", out0, out1_all)
        if type == 'img_from_mol':
            for accuracy_name, accuracy in accuracy_dict.items():
                metrics_dict[f"{accuracy_name}_mol"] = accuracy(logits, labels).cpu()
        if type == 'mol_from_img':
            for accuracy_name, accuracy in accuracy_dict.items():
                metrics_dict[f"{accuracy_name}_img"] = accuracy(logits, labels).cpu()
        metrics_dict[f"loss_{type}"] = loss
        return loss, metrics_dict


class CloomLoss(torch.nn.Module):
    """Implementation of the CLOOM loss using hopfield retrieval."""

    def __init__(
            self,
            hopfield_layer: any,
            gather_distributed: bool = False,
            normalize: bool = True,
            inv_tau: float = 4.6052
    ):
        super().__init__()
        self.hopfield_layer = hopfield_layer
        self.normalize = normalize
        self.inv_tau = inv_tau
        self.gather_distributed = gather_distributed
        if gather_distributed and not torch_dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

    def forward(
            self,
            img_rep: torch.Tensor,
            mol_rep: torch.Tensor,
    ) -> (torch.Tensor, Dict):
        """Forward pass of the CLOOM loss."""
        if self.normalize:
            img_rep = nn.functional.normalize(img_rep, dim=1)
            mol_rep = nn.functional.normalize(mol_rep, dim=1)

        if self.gather_distributed and world_size() > 1:
            img_rep_all = all_gather(img_rep, 0, None)
            mol_rep_all = all_gather(mol_rep, 0, None)
        else:
            img_rep_all = img_rep
            mol_rep_all = mol_rep
        loss0, metrics_1 = self._loss(img_rep, mol_rep, img_rep_all, mol_rep_all, "mol_from_img")
        loss1, metrics_2 = self._loss(mol_rep, img_rep, mol_rep_all, img_rep_all, 'img_from_mol')
        all_metrics = dict()
        all_metrics.update(metrics_1)
        all_metrics.update(metrics_2)
        return 0.5 * (loss0 + loss1), all_metrics

    def _loss(self, out0, out1, out0_all, out1_all, type):
        """Calculates CLOOM loss for out0 with respect to its positives in out1
        and the negatives in out1, out0_all, and out1_all.
        """
        batch_size = out0.shape[0]
        num_classes = world_size() * batch_size

        acc = Accuracy(task='multiclass', num_classes=num_classes).to(out0.device)
        acc_10 = Accuracy(task='multiclass', num_classes=num_classes, top_k=10).to(out0.device)
        acc_5 = Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(out0.device)
        top_k = int(math.ceil(num_classes * .001))
        acc_point_1_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(out0.device)
        top_k = int(math.ceil(num_classes * .01))
        acc_1_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(out0.device)
        top_k = int(math.ceil(num_classes * .05))
        acc_5_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(out0.device)
        top_k = int(math.ceil(num_classes * .1))
        acc_10_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(out0.device)

        accuracy_dict = {
            'top_1_accuracy': acc,
            "top_10_accuracy": acc_10,
            "top_5_accuracy": acc_5,
            "top_0.1_percent_accuracy": acc_point_1_percent,
            "top_1_percent_accuracy": acc_1_percent,
            "top_5_percent_accuracy": acc_5_percent,
            "top_10_percent_accuracy": acc_10_percent,
        }
        metrics_dict = dict()
        labels = torch.arange(batch_size, device=out0.device, dtype=torch.long)
        if self.gather_distributed and world_size() > 1:
            diag_mask = eye_rank(batch_size, device=out0.device)
        else:
            diag_mask = torch.eye(batch_size, device=out0.device, dtype=torch.bool)
        loss, p_xx, p_xy, p_yx, p_yy = cloob(out0, out1, self.hopfield_layer, inv_tau=self.inv_tau)
        if type == 'img_from_mol':
            for accuracy_name, accuracy in accuracy_dict.items():
                assert len(p_xx) == len(labels)
                logits = torch.einsum("nc,mc->nm", p_xx, p_xy)
                metrics_dict[f"{accuracy_name}_mol"] = accuracy(logits, labels).cpu()
        if type == 'mol_from_img':
            for accuracy_name, accuracy in accuracy_dict.items():
                assert len(p_yy) == len(labels)
                logits = torch.einsum("nc,mc->nm", p_yy, p_yx)
                metrics_dict[f"{accuracy_name}_img"] = accuracy(logits, labels).cpu()
        metrics_dict[f"loss_{type}"] = loss
        return loss, metrics_dict


class ClipLoss(torch.nn.Module):
    """Implementation of the CLIP loss."""

    def __init__(
            self,
            gather_distributed: bool = False,
            normalize: bool = True,
            inv_tau: float = 4.6052
    ):
        super().__init__()
        self.normalize = normalize
        self.inv_tau = inv_tau
        self.gather_distributed = gather_distributed
        self.loss_fnc_mol = nn.CrossEntropyLoss()
        self.loss_fnc_dose = nn.CrossEntropyLoss()
        if gather_distributed and not torch_dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

    def forward(
            self,
            img_rep: torch.Tensor,
            mol_rep: torch.Tensor,
    ) -> (torch.Tensor, Dict):
        """Forward pass of the CLIP loss."""
        if self.normalize:
            img_rep = nn.functional.normalize(img_rep, dim=1)
            mol_rep = nn.functional.normalize(mol_rep, dim=1)

        if self.gather_distributed and world_size() > 1:
            img_rep_all = all_gather(img_rep, 0, None)
            mol_rep_all = all_gather(mol_rep, 0, None)
        else:
            img_rep_all = img_rep
            mol_rep_all = mol_rep
        loss0, metrics_1 = self._loss(img_rep, mol_rep, img_rep_all, mol_rep_all, "mol_from_img")
        loss1, metrics_2 = self._loss(mol_rep, img_rep, mol_rep_all, img_rep_all, 'img_from_mol')
        all_metrics = dict()
        all_metrics.update(metrics_1)
        all_metrics.update(metrics_2)
        return 0.5 * (loss0 + loss1), all_metrics

    def _loss(self, out0, out1, out0_all, out1_all, type):
        """Calculates CLIP loss for out0 with respect to its positives in out1
        and the negatives in out1, out0_all, and out1_all.
        """
        batch_size = out0.shape[0]
        num_classes = world_size() * batch_size

        acc = Accuracy(task='multiclass', num_classes=num_classes).to(out0.device)
        acc_10 = Accuracy(task='multiclass', num_classes=num_classes, top_k=10).to(out0.device)
        acc_5 = Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(out0.device)
        top_k = int(math.ceil(num_classes * .001))
        acc_point_1_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(out0.device)
        top_k = int(math.ceil(num_classes * .01))
        acc_1_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(out0.device)
        top_k = int(math.ceil(num_classes * .05))
        acc_5_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(out0.device)
        top_k = int(math.ceil(num_classes * .1))
        acc_10_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(out0.device)

        accuracy_dict = {
            'top_1_accuracy': acc,
            "top_10_accuracy": acc_10,
            "top_5_accuracy": acc_5,
            "top_0.1_percent_accuracy": acc_point_1_percent,
            "top_1_percent_accuracy": acc_1_percent,
            "top_5_percent_accuracy": acc_5_percent,
            "top_10_percent_accuracy": acc_10_percent,
        }
        metrics_dict = dict()
        loss, logits_per_mol, logits_per_dose, labels = clip(out0, out1, self.loss_fnc_mol, self.loss_fnc_dose,
                                                             inv_tau=self.inv_tau)
        if type == 'img_from_mol':
            for accuracy_name, accuracy in accuracy_dict.items():
                metrics_dict[f"{accuracy_name}_mol"] = accuracy(logits_per_mol, labels).cpu()
        if type == 'mol_from_img':
            for accuracy_name, accuracy in accuracy_dict.items():
                metrics_dict[f"{accuracy_name}_img"] = accuracy(logits_per_dose, labels).cpu()
        metrics_dict[f"loss_{type}"] = loss
        return loss, metrics_dict


class HopfieldClipLoss(torch.nn.Module):
    """Implementation of the HopfieldCLIP loss."""

    def __init__(
            self,
            hopfield_layer: any,
            gather_distributed: bool = False,
            normalize: bool = True,
            inv_tau: float = 4.6052
    ):
        super().__init__()
        self.hopfield_layer = hopfield_layer
        self.normalize = normalize
        self.inv_tau = inv_tau
        self.gather_distributed = gather_distributed
        self.loss_fnc_mol = nn.CrossEntropyLoss()
        self.loss_fnc_dose = nn.CrossEntropyLoss()
        if gather_distributed and not torch_dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

    def forward(
            self,
            img_rep: torch.Tensor,
            mol_rep: torch.Tensor,
    ) -> (torch.Tensor, Dict):
        """Forward pass of the HopfieldClip loss."""
        if self.normalize:
            img_rep = nn.functional.normalize(img_rep, dim=1)
            mol_rep = nn.functional.normalize(mol_rep, dim=1)

        if self.gather_distributed and world_size() > 1:
            img_rep_all = all_gather(img_rep, 0, None)
            mol_rep_all = all_gather(mol_rep, 0, None)
        else:
            img_rep_all = img_rep
            mol_rep_all = mol_rep
        loss0, metrics_1 = self._loss(img_rep, mol_rep, img_rep_all, mol_rep_all, "mol_from_img")
        loss1, metrics_2 = self._loss(mol_rep, img_rep, mol_rep_all, img_rep_all, 'img_from_mol')
        all_metrics = dict()
        all_metrics.update(metrics_1)
        all_metrics.update(metrics_2)
        return 0.5 * (loss0 + loss1), all_metrics

    def _loss(self, out0, out1, out0_all, out1_all, type):
        """Calculates HopfieldCLIP loss for out0 with respect to its positives in out1
        and the negatives in out1, out0_all, and out1_all.
        """
        batch_size = out0.shape[0]
        num_classes = world_size() * batch_size

        acc = Accuracy(task='multiclass', num_classes=num_classes).to(out0.device)
        acc_10 = Accuracy(task='multiclass', num_classes=num_classes, top_k=10).to(out0.device)
        acc_5 = Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(out0.device)
        top_k = int(math.ceil(num_classes * .001))
        acc_point_1_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(out0.device)
        top_k = int(math.ceil(num_classes * .01))
        acc_1_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(out0.device)
        top_k = int(math.ceil(num_classes * .05))
        acc_5_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(out0.device)
        top_k = int(math.ceil(num_classes * .1))
        acc_10_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(out0.device)

        accuracy_dict = {
            'top_1_accuracy': acc,
            "top_10_accuracy": acc_10,
            "top_5_accuracy": acc_5,
            "top_0.1_percent_accuracy": acc_point_1_percent,
            "top_1_percent_accuracy": acc_1_percent,
            "top_5_percent_accuracy": acc_5_percent,
            "top_10_percent_accuracy": acc_10_percent,
        }
        metrics_dict = dict()
        loss, logits_per_mol, logits_per_dose, labels = hopfield_clip(out0, out1, self.loss_fnc_mol,
                                                                      self.loss_fnc_dose, self.hopfield_layer,
                                                                      inv_tau=self.inv_tau)
        if type == 'img_from_mol':
            for accuracy_name, accuracy in accuracy_dict.items():
                metrics_dict[f"{accuracy_name}_mol"] = accuracy(logits_per_mol, labels).cpu()
        if type == 'mol_from_img':
            for accuracy_name, accuracy in accuracy_dict.items():
                metrics_dict[f"{accuracy_name}_img"] = accuracy(logits_per_dose, labels).cpu()
        metrics_dict[f"loss_{type}"] = loss
        return loss, metrics_dict


class DCLLoss(torch.nn.Module):
    """Implementation of the Decoupled Contrastive Learning Loss from
    Decoupled Contrastive Learning [0].

    This code implements Equation 6 in [0], including the sum over all images `i`
    and views `k`. The loss is reduced to a mean loss over the mini-batch.
    The implementation was inspired by [1].

    - [0] Chun-Hsiao Y. et. al., 2021, Decoupled Contrastive Learning https://arxiv.org/abs/2110.06848
    - [1] https://github.com/raminnakhli/Decoupled-Contrastive-Learning

    Attributes:
        temperature:
            Similarities are scaled by inverse temperature.
        weight_fn:
            Weighting function `w` from the paper. Scales the loss between the
            positive views (views from the same image). No weighting is performed
            if weight_fn is None. The function must take the two input tensors
            passed to the forward call as input and return a weight tensor. The
            returned weight tensor must have the same length as the input tensors.
        gather_distributed:
            If True then negatives from all gpus are gathered before the
            loss calculation.

    Examples:
        >>> loss_fn = DCLLoss(temperature=0.07)
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # embed images using some model, for example SimCLR
        >>> out0 = model(t0)
        >>> out1 = model(t1)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(out0, out1)
        >>>
        >>> # you can also add a custom weighting function
        >>> weight_fn = lambda out0, out1: torch.sum((out0 - out1) ** 2, dim=1)
        >>> loss_fn = DCLLoss(weight_fn=weight_fn)
    """

    def __init__(
            self,
            use_self_similarity_mol: bool = True,
            use_self_similarity_img: bool = True,
            temperature: float = 0.1,
            gather_distributed: bool = False,
            normalize: bool = True,
            pval_sample_weight: bool = False,
    ):
        super().__init__()
        self.use_self_similarity_mol = use_self_similarity_mol
        self.use_self_similarity_img = use_self_similarity_img
        self.temperature = temperature
        self.gather_distributed = gather_distributed
        self.normalize = normalize
        self.pval_sample_weight = pval_sample_weight

        if gather_distributed and not torch_dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

    def forward(
            self,
            img_rep: torch.Tensor,
            mol_rep: torch.Tensor,
            sample_weight: torch.tensor = torch.tensor(1.),
    ) -> (torch.Tensor, Dict):
        """Forward pass of the DCL loss.

        Args:
            out0:
                Output projections of the first set of transformed images.
                Shape: (batch_size, embedding_size)
            out1:
                Output projections of the second set of transformed molecules.
                Shape: (batch_size, embedding_size)

        Returns:
            Mean loss over the mini-batch.
        """
        if self.normalize:
            img_rep = nn.functional.normalize(img_rep, dim=1)
            mol_rep = nn.functional.normalize(mol_rep, dim=1)

        if self.gather_distributed and world_size() > 1:
            img_rep_all = all_gather(img_rep, 0, None)
            mol_rep_all = all_gather(mol_rep, 0, None)
        else:
            img_rep_all = img_rep
            mol_rep_all = mol_rep
        if self.pval_sample_weight:
            loss0, metrics_1 = self._loss(img_rep, mol_rep, img_rep_all, mol_rep_all, "mol_from_img", sample_weight)
            loss1, metrics_2 = self._loss(mol_rep, img_rep, mol_rep_all, img_rep_all, 'img_from_mol', sample_weight)
        else:
            loss0, metrics_1 = self._loss(img_rep, mol_rep, img_rep_all, mol_rep_all, "mol_from_img")
            loss1, metrics_2 = self._loss(mol_rep, img_rep, mol_rep_all, img_rep_all, 'img_from_mol')
        all_metrics = dict()
        all_metrics.update(metrics_1)
        all_metrics.update(metrics_2)
        return 0.5 * (loss0 + loss1), all_metrics

    def _loss(self, out0, out1, out0_all, out1_all, type, sample_weight=None):
        """Calculates DCL loss for out0 with respect to its positives in out1
        and the negatives in out1, out0_all, and out1_all.

        This code implements Equation 6 in [0], including the sum over all images `i`
        but with `k` fixed at 0.

        Args:
            out0:
                Output projections of the first set of transformed images.
                Shape: (batch_size, embedding_size)
            out1:
                Output projections of the second set of transformed images.
                Shape: (batch_size, embedding_size)
            out0_all:
                Output projections of the first set of transformed images from
                all distributed processes/gpus. Should be equal to out0 in an
                undistributed setting.
                Shape (batch_size * world_size, embedding_size)
            out1_all:
                Output projections of the second set of transformed images from
                all distributed processes/gpus. Should be equal to out1 in an
                undistributed setting.
                Shape (batch_size * world_size, embedding_size)
            type:
                Whether out0 is img or mol

        Returns:
            Mean loss over the mini-batch.
        """
        batch_size = out0.shape[0]
        num_classes = world_size() * batch_size

        acc = Accuracy(task='multiclass', num_classes=num_classes).to(out0.device)
        acc_10 = Accuracy(task='multiclass', num_classes=num_classes, top_k=10).to(out0.device)
        acc_5 = Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(out0.device)
        top_k = int(math.ceil(num_classes * .001))
        acc_point_1_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(out0.device)
        top_k = int(math.ceil(num_classes * .01))
        acc_1_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(out0.device)
        top_k = int(math.ceil(num_classes * .05))
        acc_5_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(out0.device)
        top_k = int(math.ceil(num_classes * .1))
        acc_10_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(out0.device)

        accuracy_dict = {
            'top_1_accuracy': acc,
            "top_10_accuracy": acc_10,
            "top_5_accuracy": acc_5,
            "top_0.1_percent_accuracy": acc_point_1_percent,
            "top_1_percent_accuracy": acc_1_percent,
            "top_5_percent_accuracy": acc_5_percent,
            "top_10_percent_accuracy": acc_10_percent,
        }
        metrics_dict = dict()
        labels_idx = (torch.arange(batch_size) + rank() * batch_size).to(out0.device)
        if self.gather_distributed and world_size() > 1:
            diag_mask = eye_rank(batch_size, device=out0.device)
        else:
            diag_mask = torch.eye(batch_size, device=out0.device, dtype=torch.bool)
        if self.use_self_similarity_mol and type == 'img_from_mol':
            sim_00 = torch.einsum("nc,mc->nm", out0, out0_all) / self.temperature
        if self.use_self_similarity_img and type == 'mol_from_img':
            sim_00 = torch.einsum("nc,mc->nm", out0, out0_all) / self.temperature
        sim_01 = torch.einsum("nc,mc->nm", out0, out1_all) / self.temperature
        positive_loss = -sim_01[diag_mask]
        if type == 'mol_from_img':
            metrics_dict['positive_img_loss'] = torch.mean(positive_loss.cpu())
            for accuracy_name, accuracy in accuracy_dict.items():
                metrics_dict[f"{accuracy_name}_img"] = accuracy(sim_01, labels_idx).cpu()
        if type == 'img_from_mol':
            metrics_dict['positive_mol_loss'] = torch.mean(positive_loss.cpu())
            for accuracy_name, accuracy in accuracy_dict.items():
                metrics_dict[f"{accuracy_name}_mol"] = accuracy(sim_01, labels_idx).cpu()
        if self.use_self_similarity_mol and type == 'img_from_mol':
            sim_00 = sim_00[~diag_mask].view(batch_size, -1)
            negative_loss_00 = torch.logsumexp(sim_00, dim=1)
        elif self.use_self_similarity_img and type == 'mol_from_img':
            sim_00 = sim_00[~diag_mask].view(batch_size, -1)
            negative_loss_00 = torch.logsumexp(sim_00, dim=1)
        else:
            negative_loss_00 = torch.tensor(0)
        sim_01 = sim_01[~diag_mask].view(batch_size, -1)
        negative_loss_01 = torch.logsumexp(sim_01, dim=1)
        if type == 'mol_from_img':
            metrics_dict['negative_img_loss'] = torch.mean(negative_loss_01.cpu())
            if self.use_self_similarity_img:
                metrics_dict['self_negative_img_loss'] = torch.mean(negative_loss_00.cpu())
        if type == 'img_from_mol':
            metrics_dict['negative_mol_loss'] = torch.mean(negative_loss_01.cpu())
            if self.use_self_similarity_mol:
                metrics_dict['self_negative_mol_loss'] = torch.mean(negative_loss_00.cpu())
        if sample_weight:
            positive_loss = positive_loss.mean() * sample_weight.squeeze()
        else:
            positive_loss = positive_loss.mean()
        negative_loss = (negative_loss_01 + negative_loss_00).mean()
        loss = positive_loss + negative_loss
        metrics_dict[f"loss_{type}"] = loss
        return loss, metrics_dict


class NTXentLoss(torch.nn.Module):
    """Implementation of the Contrastive Cross Entropy Loss.
    This implementation follows the SimCLR[0] paper. If you enable the memory
    bank by setting the `memory_bank_size` value > 0 the loss behaves like
    the one described in the MoCo[1] paper.

    - [0] SimCLR, 2020, https://arxiv.org/abs/2002.05709
    - [1] MoCo, 2020, https://arxiv.org/abs/1911.05722

    Attributes:
        temperature:
            Scale logits by the inverse of the temperature.
        memory_bank_size:
            Number of negative samples to store in the memory bank.
            Use 0 for SimCLR. For MoCo we typically use numbers like 4096 or 65536.
        gather_distributed:
            If True then negatives from all gpus are gathered before the
            loss calculation. This flag has no effect if memory_bank_size > 0.

    Raises:
        ValueError: If abs(temperature) < 1e-8 to prevent divide by zero.

    Examples:

        >>> # initialize loss function without memory bank
        >>> loss_fn = NTXentLoss(memory_bank_size=0)
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through SimCLR or MoCo model
        >>> batch = torch.cat((t0, t1), dim=0)
        >>> output = model(batch)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(output)
    """

    def __init__(
            self,
            temperature: float = 0.1,
            gather_distributed: bool = False,
            use_self_similarity_mol: bool = True,
            use_self_similarity_img: bool = True,
            normalize: bool = True,
            img_from_mol: bool = True,
            mol_from_img: bool = True,
            p_val_sample_weight: bool = False,
    ):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.gather_distributed = gather_distributed
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.eps = 1e-8
        self.use_self_similarity_mol = use_self_similarity_mol
        self.use_self_similarity_img = use_self_similarity_img
        self.normalize = normalize
        self.img_from_mol = img_from_mol
        self.mol_from_img = mol_from_img
        self.pval_sample_weight = p_val_sample_weight
        if self.pval_sample_weight:
            raise NotImplementedError("P value sample weight for NTXent is not implemented yet")
        if abs(self.temperature) < self.eps:
            raise ValueError(
                "Illegal temperature: abs({}) < 1e-8".format(self.temperature)
            )
        if gather_distributed and not torch_dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

    def forward(
            self,
            img_rep: torch.Tensor,
            mol_rep: torch.Tensor,
            sample_weight: torch.tensor = torch.tensor(1.),
    ):
        """Forward pass through Contrastive Cross-Entropy Loss.

        If used with a memory bank, the samples from the memory bank are used
        as negative examples. Otherwise, within-batch samples are used as
        negative samples.

        Args:
            out0:
                Output projections of the first set of transformed images.
                Shape: (batch_size, embedding_size)
            out1:
                Output projections of the second set of transformed images.
                Shape: (batch_size, embedding_size)

        Returns:
            Contrastive Cross Entropy Loss value.
        """
        if self.normalize:
            img_rep = nn.functional.normalize(img_rep, dim=1)
            mol_rep = nn.functional.normalize(mol_rep, dim=1)
        if self.gather_distributed and world_size() > 1:
            img_rep_all = all_gather(img_rep, 0, None)
            mol_rep_all = all_gather(mol_rep, 0, None)
        else:
            img_rep_all = img_rep
            mol_rep_all = mol_rep
        if self.mol_from_img:
            loss0, metrics_1 = self._loss(img_rep, mol_rep, img_rep_all, mol_rep_all, "mol_from_img")
        else:
            loss0 = torch.tensor(0)
            metrics_1 = dict()
        if self.img_from_mol:
            loss1, metrics_2 = self._loss(mol_rep, img_rep, mol_rep_all, img_rep_all, 'img_from_mol')
        else:
            loss1 = torch.tensor(0)
            metrics_2 = dict()
        all_metrics = dict()
        all_metrics.update(metrics_1)
        all_metrics.update(metrics_2)
        return 0.5 * (loss0 + loss1), all_metrics

    def _loss(self, out0, out1, out0_all, out1_all, type):
        """Calculates InfoNCE loss for out0 with respect to its positives in out1
        and the negatives in out1, out0_all, and out1_all.

        Args:
            out0:
                Output projections of the first set of transformed images.
                Shape: (batch_size, embedding_size)
            out1:
                Output projections of the second set of transformed images.
                Shape: (batch_size, embedding_size)
            out0_all:
                Output projections of the first set of transformed images from
                all distributed processes/gpus. Should be equal to out0 in an
                undistributed setting.
                Shape (batch_size * world_size, embedding_size)
            out1_all:
                Output projections of the second set of transformed images from
                all distributed processes/gpus. Should be equal to out1 in an
                undistributed setting.
                Shape (batch_size * world_size, embedding_size)
            type:
                Whether out0 is img or mol

        Returns:
            Mean loss over the mini-batch.
        """
        batch_size = out0.shape[0]
        num_classes = world_size() * batch_size
        device = out0.device
        acc = Accuracy(task='multiclass', num_classes=num_classes).to(device)
        acc_10 = Accuracy(task='multiclass', num_classes=num_classes, top_k=10).to(device)
        acc_5 = Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(device)
        top_k = int(math.ceil(num_classes * .001))
        acc_point_1_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(device)
        top_k = int(math.ceil(num_classes * .01))
        acc_1_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(device)
        top_k = int(math.ceil(num_classes * .05))
        acc_5_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(device)
        top_k = int(math.ceil(num_classes * .1))
        acc_10_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(device)
        accuracy_dict = {
            'top_1_accuracy': acc,
            "top_10_accuracy": acc_10,
            "top_5_accuracy": acc_5,
            "top_0.1_percent_accuracy": acc_point_1_percent,
            "top_1_percent_accuracy": acc_1_percent,
            "top_5_percent_accuracy": acc_5_percent,
            "top_10_percent_accuracy": acc_10_percent,
        }
        metrics_dict = dict()
        labels = torch.arange(batch_size, device=device, dtype=torch.long)
        if self.gather_distributed and world_size() > 1:
            diag_mask = eye_rank(batch_size, device=device)
        else:
            diag_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        logits = torch.einsum("nc,mc->nm", out0, out1_all) / self.temperature
        if type == 'img_from_mol':
            for accuracy_name, accuracy in accuracy_dict.items():
                metrics_dict[f"{accuracy_name}_mol"] = accuracy(logits, labels).cpu()
            if self.use_self_similarity_mol:
                logits_00 = torch.einsum("nc,mc->nm", out0, out0_all) / self.temperature
                logits_00 = logits_00[~diag_mask].view(batch_size, -1)
                logits = torch.cat([logits, logits_00], dim=1)
        if type == 'mol_from_img':
            for accuracy_name, accuracy in accuracy_dict.items():
                metrics_dict[f"{accuracy_name}_img"] = accuracy(logits, labels).cpu()
            if self.use_self_similarity_img:
                logits_00 = torch.einsum("nc,mc->nm", out0, out0_all) / self.temperature
                logits_00 = logits_00[~diag_mask].view(batch_size, -1)
                logits = torch.cat([logits, logits_00], dim=1)
        loss = self.cross_entropy(logits, labels)
        metrics_dict[f"loss_{type}"] = loss
        return loss, metrics_dict


class CWCL(torch.nn.Module):
    """Implementation of the Contrastive Cross Entropy Loss.
    This implementation follows the SimCLR[0] paper. If you enable the memory
    bank by setting the `memory_bank_size` value > 0 the loss behaves like
    the one described in the MoCo[1] paper.

    - [0] SimCLR, 2020, https://arxiv.org/abs/2002.05709
    - [1] MoCo, 2020, https://arxiv.org/abs/1911.05722

    Attributes:
        temperature:
            Scale logits by the inverse of the temperature.
        memory_bank_size:
            Number of negative samples to store in the memory bank.
            Use 0 for SimCLR. For MoCo we typically use numbers like 4096 or 65536.
        gather_distributed:
            If True then negatives from all gpus are gathered before the
            loss calculation. This flag has no effect if memory_bank_size > 0.

    Raises:
        ValueError: If abs(temperature) < 1e-8 to prevent divide by zero.

    Examples:

        >>> # initialize loss function without memory bank
        >>> loss_fn = NTXentLoss(memory_bank_size=0)
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through SimCLR or MoCo model
        >>> batch = torch.cat((t0, t1), dim=0)
        >>> output = model(batch)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(output)
    """

    def __init__(
            self,
            temperature: float = 0.1,
            gather_distributed: bool = False,
            use_self_similarity_mol: bool = True,
            use_self_similarity_img: bool = True,
            normalize: bool = True,
            self_sim_temp: float = 1.0,
            self_sim_bias: float = 1.0,
            self_sim_clip_val: float = -1.0,
            similarity_fn_name: str = 'arctan',
            p_val_sample_weight=False,

    ):
        super(CWCL, self).__init__()
        self.temperature = temperature
        self.gather_distributed = gather_distributed
        self.use_self_similarity_mol = use_self_similarity_mol
        self.use_self_similarity_img = use_self_similarity_img
        self.normalize = normalize
        self.self_sim_bias = self_sim_bias
        self.self_sim_temp = self_sim_temp
        self.similarity_fn_name = similarity_fn_name
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.eps = 1e-8
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.self_sim_clip_val = self_sim_clip_val
        self.p_val_sample_weight = p_val_sample_weight
        if abs(self.temperature) < self.eps:
            raise ValueError(
                "Illegal temperature: abs({}) < 1e-8".format(self.temperature)
            )
        if gather_distributed and not torch_dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

    def get_matrix_for_loss(
            self,
            similarity_vector,
            device,
            batch_size,
            use_self_similarity=False,
            similarity='cosine',
            temp=1,
            bias=1,
            lower_similarity_clip_value=-1,
    ):
        if self.gather_distributed and world_size() > 1:
            similarity_all = all_gather(similarity_vector, 0, None)
        else:
            similarity_all = similarity_vector
        if similarity == 'cosine':
            sim_matrix = compute_similarity_matrix(
                similarity_vector,
                similarity_all,
                similarity=similarity,
            )
            sim_matrix[sim_matrix < lower_similarity_clip_value] = -1

            # https://www.desmos.com/calculator/e0tnru8xai
            sim_matrix = ((sim_matrix + bias) / (1 + bias)) ** temp
        elif similarity == 'arctan':
            sim_matrix = compute_similarity_matrix(
                similarity_vector,
                similarity_all,
                similarity=similarity,
            )
            sim_matrix[sim_matrix < lower_similarity_clip_value] = -1
            sim_matrix = ((sim_matrix + bias) / (1 + bias)) ** temp
        elif similarity == 'arctan_cosine':
            sim_cosine = compute_similarity_matrix(
                similarity_vector,
                similarity_all,
                similarity='arctan',
            )
            sim_arctan = compute_similarity_matrix(
                similarity_vector,
                similarity_all,
                similarity='cosine',
            )
            arctan_exp_factor = 0.5
            lambda_val = 0.65
            scale_factor = 2.5
            # exponentiate sim arctan by actan exp factor
            exponentiate_with_neg = lambda x, factor: torch.sign(x) * (torch.abs(x) ** factor)
            sim_arctan = exponentiate_with_neg(sim_arctan, arctan_exp_factor)

            sim_matrix = (sim_cosine * lambda_val + (1 - lambda_val) * sim_arctan) * scale_factor
            sim_matrix[sim_matrix < lower_similarity_clip_value] = -1
            sim_matrix = ((sim_matrix + bias) / (1 + bias)) ** temp

        if use_self_similarity:
            sim_matrix = torch.concat((sim_matrix, torch.zeros((batch_size, batch_size - 1), device=device)), dim=1)
        return sim_matrix

    def forward(
            self,
            img_rep: torch.Tensor,
            mol_rep: torch.Tensor,
            sim_vec_img: Union[torch.tensor, None],
            sim_vec_mol: Union[torch.tensor, None],
            sample_weight: torch.tensor = None,
    ):
        """Forward pass through Contrastive Cross-Entropy Loss.

        If used with a memory bank, the samples from the memory bank are used
        as negative examples. Otherwise, within-batch samples are used as
        negative samples.

        Args:
            out0:
                Output projections of the first set of transformed images.
                Shape: (batch_size, embedding_size)
            out1:
                Output projections of the second set of transformed images.
                Shape: (batch_size, embedding_size)

        Returns:
            Contrastive Cross Entropy Loss value.

        """
        if self.normalize:
            img_rep = nn.functional.normalize(img_rep, dim=1)
            mol_rep = nn.functional.normalize(mol_rep, dim=1)
        if self.gather_distributed and world_size() > 1:
            img_rep_all = all_gather(img_rep, 0, None)
            mol_rep_all = all_gather(mol_rep, 0, None)
        else:
            img_rep_all = img_rep
            mol_rep_all = mol_rep
        batch_size = img_rep.shape[0]
        device = img_rep.device
        if self.p_val_sample_weight:
            loss0, metrics_1 = self._loss(
                img_rep,
                mol_rep,
                img_rep_all,
                mol_rep_all,
                "mol_from_img",
                torch.tensor(0),
                sample_weight,
            )
        else:
            loss0, metrics_1 = self._loss(
                img_rep,
                mol_rep,
                img_rep_all,
                mol_rep_all,
                "mol_from_img",
                torch.tensor(0),
            )
        sim_matrix_img = self.get_matrix_for_loss(
            similarity_vector=sim_vec_img,
            device=device,
            batch_size=batch_size,
            use_self_similarity=self.use_self_similarity_img,
            similarity=self.similarity_fn_name,
            temp=self.self_sim_temp,
            bias=self.self_sim_bias,
            lower_similarity_clip_value=self.self_sim_clip_val
        )
        if self.p_val_sample_weight:
            loss1, metrics_2 = self._loss(
                mol_rep,
                img_rep,
                mol_rep_all,
                img_rep_all,
                'img_from_mol',
                sim_matrix_img,
                sample_weight,
            )
        else:
            loss1, metrics_2 = self._loss(
                mol_rep,
                img_rep,
                mol_rep_all,
                img_rep_all,
                'img_from_mol',
                sim_matrix_img,
            )
        all_metrics = dict()
        all_metrics.update(metrics_1)
        all_metrics.update(metrics_2)
        return 0.5 * (loss0 + loss1), all_metrics

    def _loss(self, out0, out1, out0_all, out1_all, type, sim_matrix, sample_weight: torch.tensor = None, ):
        """Calculates InfoNCE loss for out0 with respect to its positives in out1
        and the negatives in out1, out0_all, and out1_all.

        Args:
            out0:
                Output projections of the first set of transformed images.
                Shape: (batch_size, embedding_size)
            out1:
                Output projections of the second set of transformed images.
                Shape: (batch_size, embedding_size)
            out0_all:
                Output projections of the first set of transformed images from
                all distributed processes/gpus. Should be equal to out0 in an
                undistributed setting.
                Shape (batch_size * world_size, embedding_size)
            out1_all:
                Output projections of the second set of transformed images from
                all distributed processes/gpus. Should be equal to out1 in an
                undistributed setting.
                Shape (batch_size * world_size, embedding_size)
            type:
                Whether out0 is img or mol

        Returns:
            Mean loss over the mini-batch.
        """
        batch_size = out0.shape[0]
        num_classes = world_size() * batch_size
        device = out0.device
        acc = Accuracy(task='multiclass', num_classes=num_classes).to(device)
        acc_10 = Accuracy(task='multiclass', num_classes=num_classes, top_k=10).to(device)
        acc_5 = Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(device)
        top_k = int(math.ceil(num_classes * .001))
        acc_point_1_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(device)
        top_k = int(math.ceil(num_classes * .01))
        acc_1_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(device)
        top_k = int(math.ceil(num_classes * .05))
        acc_5_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(device)
        top_k = int(math.ceil(num_classes * .1))
        acc_10_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(device)
        accuracy_dict = {
            'top_1_accuracy': acc,
            "top_10_accuracy": acc_10,
            "top_5_accuracy": acc_5,
            "top_0.1_percent_accuracy": acc_point_1_percent,
            "top_1_percent_accuracy": acc_1_percent,
            "top_5_percent_accuracy": acc_5_percent,
            "top_10_percent_accuracy": acc_10_percent,
        }
        metrics_dict = dict()
        labels = torch.arange(batch_size, device=device, dtype=torch.long)
        if self.gather_distributed and world_size() > 1:
            diag_mask = eye_rank(batch_size, device=device)
        else:
            diag_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        logits = torch.einsum("nc,mc->nm", out0, out1_all) / self.temperature
        if type == 'img_from_mol':
            for accuracy_name, accuracy in accuracy_dict.items():
                metrics_dict[f"{accuracy_name}_mol"] = accuracy(logits, labels).cpu()
            if self.use_self_similarity_mol:
                logits_00 = torch.einsum("nc,mc->nm", out0, out0_all) / self.temperature
                logits_00 = logits_00[~diag_mask].view(batch_size, -1)
                logits = torch.cat([logits, logits_00], dim=1)
        if type == 'mol_from_img':
            for accuracy_name, accuracy in accuracy_dict.items():
                metrics_dict[f"{accuracy_name}_img"] = accuracy(logits, labels).cpu()
            if self.use_self_similarity_img:
                logits_00 = torch.einsum("nc,mc->nm", out0, out0_all) / self.temperature
                logits_00 = logits_00[~diag_mask].view(batch_size, -1)
                logits = torch.cat([logits, logits_00], dim=1)
        if torch.any(sim_matrix):
            log_prob = self.log_softmax(logits)
            scaled_prob_log = torch.multiply(log_prob, sim_matrix)
            per_point_scaling_factor = 1 / sim_matrix.sum(axis=-1)
            if self.p_val_sample_weight:
                per_point_scaling_factor = per_point_scaling_factor * sample_weight
            per_sample_loss = scaled_prob_log.sum(axis=-1)
            loss = torch.multiply(
                per_point_scaling_factor,
                per_sample_loss
            )
            loss = loss.mean()
            loss = torch.multiply(loss, -1)
        else:
            loss = self.cross_entropy(logits, labels)
        metrics_dict[f"loss_{type}"] = loss
        return loss, metrics_dict


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """

    def __init__(
            self,
            gather_distributed=True,
            rank=0,
            world_size=1,
            loss_with_sim_matrix=False,
            use_self_similarity_mol: bool = False,
            use_self_similarity_img: bool = False,
            self_sim_temp: float = 1.0,
            self_sim_bias: float = 1.0,
            self_sim_clip_val: float = -1,
            similarity_fn_name: str = 'arctan',
            soft_self_similarity: bool = False,
            p_val_sample_weight: bool = False,
            experimental: str = "",
            alpha: float = 1.0,
            beta: float = 0.0,
    ):
        super().__init__()
        self.gather_distributed = gather_distributed
        self.rank = rank
        self.world_size = world_size
        self.loss_with_sim_matrix = loss_with_sim_matrix
        self.use_self_similarity_mol = use_self_similarity_mol
        self.use_self_similarity_img = use_self_similarity_img
        self.self_sim_temp = self_sim_temp
        self.self_sim_bias = self_sim_bias
        self.self_sim_clip_val = self_sim_clip_val
        self.similarity_fn_name = similarity_fn_name
        self.soft_self_similarity = soft_self_similarity
        self.experimental = experimental
        self.alpha = alpha
        self.beta = beta

        # whether to weight the samples by p_values
        self.p_val_sample_weight = p_val_sample_weight
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, mol_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ mol_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def get_self_logits(self, modality_features, logit_scale, logit_bias=None):
        logits = logit_scale * modality_features @ modality_features.T
        diag_mask = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
        logits = logits[~diag_mask].view(logits.shape[0], -1)
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def get_self_similarity_matrix(self, sim_matrix):
        diag_mask = torch.eye(sim_matrix.shape[0], device=sim_matrix.device, dtype=torch.bool)
        new_sim_matrix = sim_matrix[~diag_mask].view(sim_matrix.shape[0], -1)
        return new_sim_matrix

    def _loss(
            self,
            image_features,
            mol_features,
            logit_scale,
            logit_bias=None,
            negative_only=False,
            sim_matrix=torch.tensor(0.),
            sample_weight: torch.tensor = None,
    ):

        batch_size = image_features.shape[0]
        num_classes = world_size() * batch_size
        device = image_features.device
        acc = Accuracy(task='multiclass', num_classes=num_classes).to(device)
        acc_10 = Accuracy(task='multiclass', num_classes=num_classes, top_k=10).to(device)
        acc_5 = Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(device)
        top_k = int(math.ceil(num_classes * .001))
        acc_point_1_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(device)
        top_k = int(math.ceil(num_classes * .0071))
        acc_point_7_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(device)
        top_k = int(math.ceil(num_classes * .01))
        acc_1_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(device)
        top_k = int(math.ceil(num_classes * .05))
        acc_5_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(device)
        top_k = int(math.ceil(num_classes * .1))
        acc_10_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(device)
        accuracy_dict = {
            'top_1_accuracy': acc,
            "top_10_accuracy": acc_10,
            "top_5_accuracy": acc_5,
            "top_0.1_percent_accuracy": acc_point_1_percent,
            "top_0.7_percent_accuracy": acc_point_7_percent,
            "top_1_percent_accuracy": acc_1_percent,
            "top_5_percent_accuracy": acc_5_percent,
            "top_10_percent_accuracy": acc_10_percent,
        }
        metrics_dict = dict()
        logits = self.get_logits(image_features, mol_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        metrics_labels = torch.arange(batch_size, device=device, dtype=torch.long)
        for accuracy_name, accuracy in accuracy_dict.items():
            metrics_dict[f"{accuracy_name}_mol"] = accuracy(logits, metrics_labels).cpu()
        if torch.any(sim_matrix):
            positives = -nn.functional.logsigmoid(1. * logits)
            negatives = -nn.functional.logsigmoid(-1. * logits)
            # loss = (sim_matrix * positives + (2 - sim_matrix) * negatives)
            loss = (sim_matrix * positives + self.alpha * ((1 + self.beta) - sim_matrix) * negatives)
            if sample_weight:
                weighted_loss = torch.diagonal(loss) * sample_weight.squeeze()
                for i in range(weighted_loss.shape[0]):
                    loss[i, i] = weighted_loss[i]
            flatten_term = sim_matrix * (1 - sim_matrix) * 2.7723
            loss = loss - flatten_term
            loss = loss.sum(axis=-1)
            loss = loss.mean()
        else:
            loss = -nn.functional.logsigmoid(labels * logits)
            if sample_weight:
                weighted_loss = torch.diagonal(loss) * sample_weight.squeeze()
                for i in range(weighted_loss.shape[0]):
                    loss[i, i] = weighted_loss[i]
            loss = loss.sum(axis=-1)
            loss = loss.mean()
        if self.use_self_similarity_img:
            self_logits_image = self.get_self_logits(image_features, logit_scale, logit_bias)
            # only provide the option for soft self similarity for the image
            if torch.any(sim_matrix) and self.soft_self_similarity:
                self_sim_matrix = self.get_self_similarity_matrix(sim_matrix)
                positives = -nn.functional.logsigmoid(1. * self_logits_image) / image_features.shape[0]
                negatives = -nn.functional.logsigmoid(-1. * self_logits_image) / image_features.shape[0]
                negatives_loss = ((self_sim_matrix * positives + (1 - self_sim_matrix) * negatives))
                loss = loss - flatten_term
                loss = loss.sum(axis=-1)
                loss += negatives_loss.mean()
            else:
                negative_labels = torch.zeros(self_logits_image.shape, device=self_logits_image.device) - 1
                loss += -nn.functional.logsigmoid(negative_labels * self_logits_image).sum(axis=-1).mean()
        if self.use_self_similarity_mol:
            self_logits_mol = self.get_self_logits(mol_features, logit_scale, logit_bias)

            negative_labels = torch.zeros(self_logits_mol.shape, device=self_logits_mol.device) - 1
            loss += -nn.functional.logsigmoid(negative_labels * self_logits_mol).sum(axis=-1).mean()

        metrics_dict[f"loss_mol"] = loss.item()
        return loss, metrics_dict

    def forward(
            self,
            image_features,
            mol_features,
            logit_scale,
            logit_bias,
            sim_vec_img: Union[torch.tensor, None],
            sample_weight: torch.tensor = None,
    ):
        if self.world_size > 1:
            raise NotImplementedError("Distributed SigLipLoss not yet implemented")
        if self.loss_with_sim_matrix:
            batch_size = image_features.shape[0]
            device = image_features.device
            sim_matrix = self.get_matrix_for_loss(
                similarity_vector=sim_vec_img,
                device=device,
                batch_size=batch_size,
                use_self_similarity=self.use_self_similarity_img,
                similarity=self.similarity_fn_name,
                temp=self.self_sim_temp,
                bias=self.self_sim_bias,
                lower_similarity_clip_value=self.self_sim_clip_val,
            )
        else:
            sim_matrix = torch.tensor(0.)
        image_features = nn.functional.normalize(image_features, dim=1)
        mol_features = nn.functional.normalize(mol_features, dim=1)
        if self.p_val_sample_weight:
            loss, metrics_dict = self._loss(
                image_features,
                mol_features,
                logit_scale=logit_scale,
                logit_bias=logit_bias,
                sim_matrix=sim_matrix,
                sample_weight=sample_weight,
            )
        else:
            if not self.experimental:
                loss, metrics_dict = self._loss(
                    image_features,
                    mol_features,
                    logit_scale=logit_scale,
                    logit_bias=logit_bias,
                    sim_matrix=sim_matrix,
                )
            elif self.experimental == 'mse':
                loss, metrics_dict = self._mse_loss(
                    image_features=image_features,
                    mol_features=mol_features,
                    logit_scale=logit_scale,
                    logit_bias=logit_bias,
                    sim_matrix=sim_matrix
                )
            elif self.experimental == 'continuous_class_sigmoid':
                loss, metrics_dict = self._continuous_class_sigmoid_loss(
                    image_features=image_features,
                    mol_features=mol_features,
                    logit_scale=logit_scale,
                    logit_bias=logit_bias,
                    sim_matrix=sim_matrix
                )
            elif self.experimental == 'logit_bce_loss':
                loss, metrics_dict = self._logit_bce_loss(
                    image_features=image_features,
                    mol_features=mol_features,
                    logit_scale=logit_scale,
                    logit_bias=logit_bias,
                    sim_matrix=sim_matrix
                )
            else:
                raise ValueError(f"Invalid value for experimental: {self.experimental}")

        if self.world_size > 1:
            raise NotImplementedError("Distributed SigLipLoss not yet implemented")
        metrics_dict['siglip_temperature'] = logit_scale
        metrics_dict['siglip_bias'] = logit_bias
        return loss, metrics_dict

    def get_matrix_for_loss(
            self,
            similarity_vector,
            device,
            batch_size,
            use_self_similarity=False,
            similarity='cosine',
            temp=1,
            bias=1,
            lower_similarity_clip_value=-1,
    ):
        if self.gather_distributed and world_size() > 1:
            similarity_all = all_gather(similarity_vector, 0, None)
        else:
            similarity_all = similarity_vector
        if similarity == 'cosine':
            sim_matrix = compute_similarity_matrix(
                similarity_vector,
                similarity_all,
                similarity=similarity,
            )
            # set values below lower_similarity_clip_value to -1
            sim_matrix[sim_matrix < lower_similarity_clip_value] = -1

            # https://www.desmos.com/calculator/e0tnru8xai
            sim_matrix = ((sim_matrix + bias) / (1 + bias)) ** temp
        elif similarity == 'arctan':
            sim_matrix = compute_similarity_matrix(
                similarity_vector,
                similarity_all,
                similarity=similarity,
            )
            sim_matrix[sim_matrix < lower_similarity_clip_value] = -1

            sim_matrix = ((sim_matrix + bias) / (1 + bias)) ** temp
        elif similarity == 'arctan_cosine':
            sim_cosine = compute_similarity_matrix(
                similarity_vector,
                similarity_all,
                similarity='arctan',
            )
            sim_arctan = compute_similarity_matrix(
                similarity_vector,
                similarity_all,
                similarity='cosine',
            )
            arctan_exp_factor = 0.5
            lambda_val = 0.65
            scale_factor = 2.5
            # exponentiate sim arctan by actan exp factor
            exponentiate_with_neg = lambda x, factor: torch.sign(x) * (torch.abs(x) ** factor)
            sim_arctan = exponentiate_with_neg(sim_arctan, arctan_exp_factor)

            sim_matrix = (sim_cosine * lambda_val + (1 - lambda_val) * sim_arctan) * scale_factor

            sim_matrix[sim_matrix < lower_similarity_clip_value] = -1
            sim_matrix = ((sim_matrix + bias) / (1 + bias)) ** temp

        return sim_matrix

    def _mse_loss(
            self,
            image_features,
            mol_features,
            logit_scale,
            logit_bias=None,
            sim_matrix=torch.tensor(0.),
    ):

        batch_size = image_features.shape[0]
        assert batch_size > 1
        num_classes = world_size() * batch_size
        device = image_features.device

        acc = Accuracy(task='multiclass', num_classes=num_classes).to(device)
        acc_10 = Accuracy(task='multiclass', num_classes=num_classes, top_k=10).to(device)
        acc_5 = Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(device)

        top_k = int(math.ceil(num_classes * .001))
        assert top_k > 0
        acc_point_1_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(device)
        top_k = int(math.ceil(num_classes * .0071))
        assert top_k > 0
        acc_point_7_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(device)
        top_k = int(math.ceil(num_classes * .01))
        assert top_k > 0
        acc_1_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(device)
        top_k = int(math.ceil(num_classes * .05))
        assert top_k > 0
        acc_5_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(device)
        top_k = int(math.ceil(num_classes * .1))
        assert top_k > 0
        acc_10_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(device)

        accuracy_dict = {
            'top_1_accuracy': acc,
            "top_10_accuracy": acc_10,
            "top_5_accuracy": acc_5,
            "top_0.1_percent_accuracy": acc_point_1_percent,
            "top_0.7_percent_accuracy": acc_point_7_percent,
            "top_1_percent_accuracy": acc_1_percent,
            "top_5_percent_accuracy": acc_5_percent,
            "top_10_percent_accuracy": acc_10_percent,
        }
        metrics_dict = dict()

        logits = self.get_logits(image_features, mol_features, logit_scale=1, logit_bias=0.)
        metrics_labels = torch.arange(batch_size, device=device, dtype=torch.long)

        for accuracy_name, accuracy in accuracy_dict.items():
            metrics_dict[f"{accuracy_name}_mol"] = accuracy(logits, metrics_labels).cpu()

        assert torch.any(sim_matrix)
        loss = F.mse_loss(logits, sim_matrix)

        assert not self.use_self_similarity_img
        assert not self.use_self_similarity_mol

        metrics_dict[f"loss_mol"] = loss.item()
        return loss, metrics_dict

    def _continuous_class_sigmoid_loss(
            self,
            image_features,
            mol_features,
            logit_scale,
            logit_bias=None,
            sim_matrix=torch.tensor(0.),

    ):

        batch_size = image_features.shape[0]
        assert batch_size > 1
        num_classes = world_size() * batch_size
        device = image_features.device

        acc = Accuracy(task='multiclass', num_classes=num_classes).to(device)
        acc_10 = Accuracy(task='multiclass', num_classes=num_classes, top_k=10).to(device)
        acc_5 = Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(device)
        top_k = int(math.ceil(num_classes * .001))
        assert top_k > 0
        acc_point_1_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(device)
        top_k = int(math.ceil(num_classes * .01))
        assert top_k > 0
        acc_1_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(device)
        top_k = int(math.ceil(num_classes * .05))
        assert top_k > 0
        acc_5_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(device)
        top_k = int(math.ceil(num_classes * .1))
        assert top_k > 0
        acc_10_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(device)

        accuracy_dict = {
            'top_1_accuracy': acc,
            "top_10_accuracy": acc_10,
            "top_5_accuracy": acc_5,
            "top_0.1_percent_accuracy": acc_point_1_percent,
            "top_1_percent_accuracy": acc_1_percent,
            "top_5_percent_accuracy": acc_5_percent,
            "top_10_percent_accuracy": acc_10_percent,
        }
        metrics_dict = dict()

        logits = self.get_logits(image_features, mol_features, logit_scale=logit_scale, logit_bias=logit_bias)
        metrics_labels = torch.arange(batch_size, device=device, dtype=torch.long)

        for accuracy_name, accuracy in accuracy_dict.items():
            metrics_dict[f"{accuracy_name}_mol"] = accuracy(logits, metrics_labels).cpu()

        assert torch.any(sim_matrix)
        loss = -nn.functional.logsigmoid((sim_matrix * 2 - 1) * logits)

        loss = loss.sum(axis=-1)
        loss = loss.mean()

        assert not self.use_self_similarity_img
        assert not self.use_self_similarity_mol

        metrics_dict[f"loss_mol"] = loss.item()
        return loss, metrics_dict

    def _logit_bce_loss(
            self,
            image_features,
            mol_features,
            logit_scale,
            logit_bias=None,
            sim_matrix=torch.tensor(0.),
    ):

        batch_size = image_features.shape[0]
        assert batch_size > 1
        num_classes = world_size() * batch_size
        device = image_features.device

        acc = Accuracy(task='multiclass', num_classes=num_classes).to(device)
        acc_10 = Accuracy(task='multiclass', num_classes=num_classes, top_k=10).to(device)
        acc_5 = Accuracy(task='multiclass', num_classes=num_classes, top_k=5).to(device)
        top_k = int(math.ceil(num_classes * .001))
        assert top_k > 0
        acc_point_1_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(device)
        top_k = int(math.ceil(num_classes * .01))
        assert top_k > 0
        acc_1_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(device)
        top_k = int(math.ceil(num_classes * .05))
        assert top_k > 0
        acc_5_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(device)
        top_k = int(math.ceil(num_classes * .1))
        assert top_k > 0
        acc_10_percent = Accuracy(task='multiclass', num_classes=num_classes, top_k=top_k).to(device)

        accuracy_dict = {
            'top_1_accuracy': acc,
            "top_10_accuracy": acc_10,
            "top_5_accuracy": acc_5,
            "top_0.1_percent_accuracy": acc_point_1_percent,
            "top_1_percent_accuracy": acc_1_percent,
            "top_5_percent_accuracy": acc_5_percent,
            "top_10_percent_accuracy": acc_10_percent,
        }
        metrics_dict = dict()

        logits = self.get_logits(image_features, mol_features, logit_scale, logit_bias)
        metrics_labels = torch.arange(batch_size, device=device, dtype=torch.long)

        for accuracy_name, accuracy in accuracy_dict.items():
            metrics_dict[f"{accuracy_name}_mol"] = accuracy(logits, metrics_labels).cpu()

        assert torch.any(sim_matrix)
        # https://www.desmos.com/calculator/oyzpdy4tuh
        loss = F.binary_cross_entropy_with_logits(logits, sim_matrix)

        assert not self.use_self_similarity_img
        assert not self.use_self_similarity_mol

        metrics_dict[f"loss_mol"] = loss.item()
        return loss, metrics_dict
