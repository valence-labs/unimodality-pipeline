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

class SigClipLoss(torch.nn.Module):
    """Implementation of the SigCLIP loss."""
    def __init__(
            self,
            temperature: float = 1.0,
            normalize: bool = True,
            bias: float = 0.0,
            learnable_bias: bool = True,
            no_tx_head: bool = False,
            no_ph_head: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        self.no_tx_head = no_tx_head
        self.no_ph_head = no_ph_head

        if learnable_bias:
            self.bias = nn.Parameter(torch.tensor(bias))
        else:
            self.register_buffer('bias', torch.tensor(bias))

    def forward(
            self,
            tx_rep: torch.Tensor,
            ph_rep: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the SigCLIP loss."""
        if self.normalize:
            tx_rep = F.normalize(tx_rep, dim=-1)
            ph_rep = F.normalize(ph_rep, dim=-1)

        # Compute similarities
        if self.no_ph_head:
            # Ph encoder is disabled; use tx_rep directly
            # Compute similarities between tx_rep and tx_rep
            logits = self.temperature * (tx_rep @ tx_rep.T) - self.bias
        elif self.no_tx_head:
            # Tx encoder is disabled; use ph_rep directly
            # Compute similarities between ph_rep and ph_rep
            logits = self.temperature * (ph_rep @ ph_rep.T) - self.bias
        else:
            # Both encoders are active
            logits = self.temperature * (tx_rep @ ph_rep.T) - self.bias

        # Create labels z_{ij}
        batch_size = tx_rep.size(0)
        device = tx_rep.device
        z = torch.ones_like(logits, device=device)
        z.fill_(-1)
        z.fill_diagonal_(1)

        # Compute loss
        loss = F.softplus(-z * logits).mean()

        return loss