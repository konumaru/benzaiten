from abc import ABCMeta, abstractmethod
from typing import Tuple, Union

import torch
import torch.nn as nn


class BaseComposerModel(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = torch.empty(1)
        logvar = torch.empty(1)
        return mean, logvar

    @abstractmethod
    def decode(
        self, latent: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        x_hat = torch.empty(1)
        return x_hat

    @abstractmethod
    def reparameterization(
        self, mean: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        latent = torch.empty(1)
        return latent

    def generate(
        self, condition: torch.Tensor, x: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        x = torch.empty(1)
        return x
