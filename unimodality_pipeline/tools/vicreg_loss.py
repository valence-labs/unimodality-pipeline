import torch
import torch.nn as nn
import torch.nn.functional as F

class VicRegLoss(nn.Module):
    """Implementation of the VICReg loss."""
    def __init__(
            self,
            sim_loss_weight: float = 25.0,
            var_loss_weight: float = 25.0,
            cov_loss_weight: float = 1.0,
            gather_distributed: bool = False,
            normalize: bool = False,
    ):
        super(VicRegLoss, self).__init__()
        self.sim_loss_weight = sim_loss_weight
        self.var_loss_weight = var_loss_weight
        self.cov_loss_weight = cov_loss_weight
        self.gather_distributed = gather_distributed
        self.normalize = normalize

    def forward(
            self,
            z1: torch.Tensor,
            z2: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the VICReg loss."""
        if self.normalize:
            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)

        sim_loss = F.mse_loss(z1, z2)

        # Optionally gather representations from all GPUs
        if self.gather_distributed and torch.distributed.is_initialized():
            z1 = self.gather(z1)
            z2 = self.gather(z2)

        var_loss = self.variance_loss(z1) + self.variance_loss(z2)
        cov_loss = self.covariance_loss(z1) + self.covariance_loss(z2)

        loss = (
            self.sim_loss_weight * sim_loss +
            self.var_loss_weight * var_loss +
            self.cov_loss_weight * cov_loss
        )
        return loss

    def variance_loss(self, z):
        eps = 1e-4
        std_z = torch.sqrt(z.var(dim=0) + eps)
        std_loss = torch.mean(F.relu(1 - std_z))
        return std_loss

    def covariance_loss(self, z):
        N, D = z.size()
        z = z - z.mean(dim=0)
        cov_z = (z.T @ z) / (N - 1)
        diag = torch.eye(D, device=z.device)
        cov_loss = cov_z[~diag.bool()].pow(2).sum() / D
        return cov_loss

    def gather(self, z):
        """Gathers tensors from all GPUs."""
        tensors_gather = [torch.zeros_like(z) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, z)
        output = torch.cat(tensors_gather, dim=0)
        return output
