from typing import Dict

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from config import Config


def get_optimizer(cfg: Config, model: nn.Module) -> Optimizer:
    """Instantiate optimizer."""
    optimizer_class = getattr(optim, cfg.training.optimizer.name)
    optimizer: Optimizer = optimizer_class(
        model.parameters(), **cfg.training.optimizer.params
    )
    return optimizer


def get_lr_scheduler(cfg: Config, optimizer: Optimizer) -> _LRScheduler:
    """Instantiate scheduler."""
    lr_scheduler_class = getattr(
        optim.lr_scheduler, cfg.training.lr_scheduler.name
    )
    lr_scheduler: _LRScheduler = lr_scheduler_class(
        optimizer, **cfg.training.lr_scheduler.params  # type: ignore
    )
    return lr_scheduler


class CustomLoss(nn.Module):
    """Custom loss."""

    def __init__(self, cfg: Config, model: nn.Module) -> None:
        """Initialize class."""
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.kl_weight = cfg.model.vae.kl_weight

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss."""
        vae = {
            "reconst": torch.empty(0),
            "mean": torch.empty(0),
            "logvar": torch.empty(0),
        }
        loss = {"xent": 0.0, "kl": 0.0}
        loss_func = {"xent": nn.CrossEntropyLoss(), "mse": nn.MSELoss()}

        source = batch["source"]  # melody + chord
        target = batch["target"]  # melody

        _, encoder_state = self.model.encoder(source)  # type: ignore
        hiddens = torch.squeeze(encoder_state[0])  # [32, 1024]
        vae["reconst"], vae["mean"], vae["logvar"] = self.model.vae(hiddens)  # type: ignore
        inputs = vae["reconst"].unsqueeze(1)
        inputs = inputs.repeat(1, target.shape[1], 1)
        outputs, _ = self.model.decoder(inputs)  # type: ignore
        outputs = torch.permute(outputs, (0, 2, 1))

        loss["xent"] = loss_func["xent"](outputs, target)
        loss["kl"] = (-0.5) * torch.sum(  # type: ignore
            1 + vae["logvar"] - vae["mean"].pow(2) - vae["logvar"].exp()
        )
        output: torch.Tensor = loss["xent"] + self.kl_weight * loss["kl"]
        return output


def get_loss(cfg: Config, model: nn.Module) -> CustomLoss:
    """Instantiate customized loss."""
    custom_loss = CustomLoss(cfg, model)
    return custom_loss
