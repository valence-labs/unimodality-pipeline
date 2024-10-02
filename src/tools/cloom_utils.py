"""Stores CLOOM loss functions and utilities."""

import torch


def infoLOOB_loss(x, y, i, inv_tau):
    tau = 1 / inv_tau
    k = x @ y.T / tau
    positives = -torch.mean(torch.sum(k * i, dim=1))
    large_neg = -1000.0
    arg_lse = k * torch.logical_not(i) + i * large_neg
    negatives = torch.mean(torch.logsumexp(arg_lse, dim=1))
    return tau * (positives + negatives)


def cloob(mol_features, dose_features, hopfield_layer, inv_tau=4.6052):
    p_xx, p_yy, p_xy, p_yx = hopfield_retrieval(mol_features, dose_features, hopfield_layer)
    identity = torch.eye(p_xx.shape[0]) > 0.5
    i = identity.to(p_xx.device)
    loss_mol = infoLOOB_loss(p_xx, p_xy, i, inv_tau=inv_tau)
    loss_dose = infoLOOB_loss(p_yy, p_yx, i, inv_tau=inv_tau)
    return loss_mol + loss_dose, p_xx, p_xy, p_yx, p_yy


def clip(mol_features, dose_features, loss_fnc_mol, loss_fnc_dose, inv_tau=4.6052):
    logits_per_mol = inv_tau * mol_features @ dose_features.t()
    logits_per_dose = logits_per_mol.t()
    ground_truth = torch.arange(len(logits_per_mol)).long()
    loss_mol = loss_fnc_mol(logits_per_mol, ground_truth) / 2
    loss_dose = loss_fnc_dose(logits_per_dose, ground_truth) / 2
    return loss_mol + loss_dose, logits_per_mol, logits_per_dose, ground_truth


def hopfield_clip(mol_features, dose_features, loss_fnc_mol, loss_fnc_dose, hopfield_layer, inv_tau=4.6052):
    mol_features, dose_features, p_xy, p_yx = hopfield_retrieval(mol_features, dose_features, hopfield_layer)
    logits_per_mol = inv_tau * mol_features @ dose_features.t()
    logits_per_dose = logits_per_mol.t()
    ground_truth = torch.arange(len(logits_per_mol)).long()
    loss_mol = loss_fnc_mol(logits_per_mol, ground_truth) / 2
    loss_dose = loss_fnc_dose(logits_per_dose, ground_truth) / 2
    return loss_mol + loss_dose, logits_per_mol, logits_per_dose, ground_truth


def hopfield_retrieval(mol_features, dose_features, hopfield_layer):
    patterns_xx = hopfield(state_patterns=mol_features, stored_patterns=mol_features, hopfield_layer=hopfield_layer)
    patterns_yy = hopfield(state_patterns=dose_features, stored_patterns=dose_features, hopfield_layer=hopfield_layer)
    patterns_xy = hopfield(state_patterns=dose_features, stored_patterns=mol_features, hopfield_layer=hopfield_layer)
    patterns_yx = hopfield(state_patterns=mol_features, stored_patterns=dose_features, hopfield_layer=hopfield_layer)
    return patterns_xx, patterns_yy, patterns_xy, patterns_yx


def hopfield(state_patterns, stored_patterns, hopfield_layer):
    retrieved_patterns = hopfield_layer.forward(
        (stored_patterns.unsqueeze(0), state_patterns.unsqueeze(0), stored_patterns.unsqueeze(0))).squeeze()
    retrieved_patterns = retrieved_patterns / retrieved_patterns.norm(dim=1, keepdim=True)
    return retrieved_patterns
