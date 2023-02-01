import torch
import torch.nn as nn


class VAELoss(nn.Module):
    def __init__(self, kld_weight: float = 1e-3) -> None:
        super(VAELoss, self).__init__()
        self.kld_weight = kld_weight

    def forward(
        self,
        label: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        recons_loss = nn.functional.cross_entropy(
            x_hat, label, reduction="mean"
        )
        kld_loss = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())
        vae_loss = recons_loss + (self.kld_weight * kld_loss)
        return vae_loss
