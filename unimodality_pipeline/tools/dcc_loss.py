import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    )
logger = logging.getLogger(__name__)



class DCCLoss(nn.Module):
    """Implementation of the Deep Canonical Correlation (DCC) loss using SVD."""
    def __init__(
            self,
            outdim_size: int,
            use_all_singular_values: bool = False,
            epsilon: float = 1e-5,
    ):
        super(DCCLoss, self).__init__()
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values
        self.epsilon = epsilon

    def forward(
            self,
            tx_rep: torch.Tensor,
            ph_rep: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the DCCA loss."""
        o1 = tx_rep.size(1)
        o2 = ph_rep.size(1)
        batch_size = tx_rep.size(0)

        # Center the representations
        H1 = tx_rep - tx_rep.mean(dim=0)
        H2 = ph_rep - ph_rep.mean(dim=0)

        N = batch_size - 1  # Adjust for unbiased covariance estimation

        # Compute covariance matrices with regularization
        S11 = (H1.t() @ H1) / N + self.epsilon * torch.eye(o1).to(tx_rep.device)
        S22 = (H2.t() @ H2) / N + self.epsilon * torch.eye(o2).to(tx_rep.device)
        S12 = (H1.t() @ H2) / N

        # Compute the inverse square root of S11 and S22 via SVD
        U1, S1, _ = torch.svd(S11)
        U2, S2, _ = torch.svd(S22)

        # Regularize singular values and compute inverse square roots
        S1_inv_sqrt = torch.diag(1.0 / torch.sqrt(S1 + self.epsilon))
        S2_inv_sqrt = torch.diag(1.0 / torch.sqrt(S2 + self.epsilon))

        # Compute inverse square root matrices
        inv_S11_sqrt = U1 @ S1_inv_sqrt @ U1.t()
        inv_S22_sqrt = U2 @ S2_inv_sqrt @ U2.t()

        # Compute the T matrix
        T = inv_S11_sqrt @ S12 @ inv_S22_sqrt

        # Compute SVD of T
        U, S, V = torch.svd(T)

        # Select top singular values if required
        if self.use_all_singular_values:
            corr = S
        else:
            corr = S[:self.outdim_size]

        # Sum up the correlations (negative for minimization)
        loss = -torch.sum(corr)

        return loss