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



class ClipLoss(torch.nn.Module):
    """Implementation of the CLIP loss."""
    def __init__(
            self,
            gather_distributed: bool = False,
            normalize: bool = True,
            temperature: float = 4.6052
    ):
        super().__init__()
        self.normalize = normalize
        self.temperature = temperature
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(
            self,
            tx_rep: torch.Tensor,
            ph_rep: torch.Tensor,
    ) -> (torch.Tensor):
        """Forward pass of the CLIP loss."""
        if self.normalize:
            tx_rep = F.normalize(tx_rep, dim=-1)
            ph_rep = F.normalize(ph_rep, dim=-1)

        logits = (tx_rep @ ph_rep.T) / self.temperature
        ph_similarity = ph_rep @ ph_rep.T
        tx_similarity = tx_rep @ tx_rep.T
        targets = F.softmax(
            (ph_similarity + tx_similarity) / 2 * self.temperature, dim=-1
        )
        ph_loss = (-targets.T * self.log_softmax(logits.T)).sum(-1)
        tx_loss = (-targets * self.log_softmax(logits)).sum(-1)
        return (tx_loss + ph_loss) / 2.0






