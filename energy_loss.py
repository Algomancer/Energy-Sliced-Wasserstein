import torch
import torch.nn as nn

class EnergySlicedWasserstein(nn.Module):
    """
    Computes the Energy Sliced Wasserstein Loss between two distributions.
    """
    def __init__(self, num_projections=100, p=2):
        """
        Initializes the loss function.

        Parameters:
        - num_projections: Number of random projections (L).
        - p: Power parameter (p).
        """
        super(EnergySlicedWasserstein, self).__init__()
        self.num_projections = num_projections
        self.p = p

    def forward(self, X, Y):
        """
        Computes the loss between distributions X and Y.

        Parameters:
        - X, Y: Input tensors of shape (batch_size, features).

        Returns:
        - loss: Scalar tensor representing the loss.
        """
        device = X.device
        dtype = X.dtype
        dim = X.size(-1)
        N = X.size(0)

        # Generate random projections
        theta = torch.randn(self.num_projections, dim, device=device, dtype=dtype)
        theta = theta / theta.norm(dim=1, keepdim=True)  # Normalize to unit vectors

        # Project the samples
        X_proj = X @ theta.T  # Shape: (N, num_projections)
        Y_proj = Y @ theta.T  # Shape: (N, num_projections)

        # Sort the projections
        X_proj_sorted, _ = X_proj.sort(dim=0)
        Y_proj_sorted, _ = Y_proj.sort(dim=0)

        # Compute the p-th power of the absolute differences
        wasserstein_distance = torch.abs(X_proj_sorted - Y_proj_sorted) ** self.p
        wasserstein_distance = wasserstein_distance.sum(dim=0)  # Shape: (num_projections,)

        # Compute softmax weights
        weights = torch.softmax(wasserstein_distance, dim=0)

        # Compute the weighted sum of the distances
        loss = (weights * wasserstein_distance).sum() / N  # Normalize by the number of samples

        # Finalize the loss
        return loss.pow(1.0 / self.p)
