import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import mse_loss

def my_rbf_kernel(data, gamma=20):
    """Compute the RBF (Gaussian) kernel matrix for the input data."""
    gamma = gamma / data.shape[1]
    pairwise_sq_dists = torch.cdist(data, data, p=2) ** 2
    K = torch.exp(-gamma * pairwise_sq_dists)
    return K

def gromov_wasserstein(C1, C2, p, q, loss_fun='square_loss', **kwargs):
    """
    Compute the Gromov-Wasserstein transport between two metric spaces.
    """
    # Import the POT library's Gromov-Wasserstein function
    import ot
    # Convert tensors to NumPy arrays
    C1_np = C1.detach().cpu().numpy()
    C2_np = C2.detach().cpu().numpy()
    p_np = p.squeeze().detach().cpu().numpy()
    q_np = q.squeeze().detach().cpu().numpy()
    # Compute the transport plan using the POT library
    T_np = ot.gromov.gromov_wasserstein(C1_np, C2_np, p_np, q_np, loss_fun=loss_fun, **kwargs)
    # Convert the transport plan back to a tensor
    T = torch.from_numpy(T_np).to(C1.device).type(C1.dtype)
    return T

class MultiViewLoss(nn.Module):
    """Implementation of the loss function for multi-view representations."""
    def __init__(
        self,
        Wlambda: float = 0.0,
        iters: int = 1,
        gamma: float = 1000.0,
    ):
        super().__init__()
        self.Wlambda = Wlambda
        self.iters = iters
        self.gamma = gamma

        # Initialize learnable weights for each view (modality)
        # Weights will be automatically moved to the correct device by PyTorch Lightning
        self.weights = nn.Parameter(torch.ones(2))  # Assuming two views: transcriptomics and phenomics

    def forward(
        self,
        tx_rep: torch.Tensor,
        ph_rep: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the MultiViewLoss.

        Args:
            tx_rep (torch.Tensor): Transcriptomics representations (batch_size, feature_dim).
            ph_rep (torch.Tensor): Phenomics representations (batch_size, feature_dim).

        Returns:
            torch.Tensor: Computed loss value.
        """
        batch_size = tx_rep.size(0)
        views = 2  # Number of views/modalities

        # Apply softmax to the weights to get normalized view weights
        weights_views = F.softmax(self.weights, dim=-1)

        # Compute similarity matrices using the RBF kernel
        Cs = []
        Cs.append(my_rbf_kernel(tx_rep, gamma=self.gamma))
        Cs.append(my_rbf_kernel(ph_rep, gamma=self.gamma))

        # Create uniform distributions over the samples for each view
        # Use the device and dtype of the input tensors
        ps = [torch.ones(batch_size, 1, device=tx_rep.device, dtype=tx_rep.dtype) / batch_size for _ in range(views)]

        # Initialize matrices C and E, and distributions p and p_e based on the batch size
        C = torch.eye(batch_size, device=tx_rep.device, dtype=tx_rep.dtype)
        E = torch.eye(batch_size, device=tx_rep.device, dtype=tx_rep.dtype)
        p = torch.ones(batch_size, 1, device=tx_rep.device, dtype=tx_rep.dtype) / batch_size
        p_e = torch.ones(batch_size, 1, device=tx_rep.device, dtype=tx_rep.dtype) / batch_size

        # Iteratively update C using Gromov-Wasserstein barycenter computation
        for iteration in range(self.iters):
            B = torch.zeros_like(C)
            for i in range(views):
                # Compute the Gromov-Wasserstein transport plan
                T = gromov_wasserstein(Cs[i], C, ps[i], p)
                # Update B with weighted transported similarities
                B += weights_views[i] * T.T @ Cs[i] @ T
            # Update C as the normalized barycenter
            C = B / (p @ p.T)

        # Compute the entropy of the weights as a regularization term
        weight_entropy = torch.dot(weights_views, torch.log(weights_views + 1e-10))  # Adding epsilon for numerical stability

        # Compute the loss as Mean Squared Error between C and E, plus the weighted entropy
        loss = mse_loss(C, E) + self.Wlambda * weight_entropy

        return loss
