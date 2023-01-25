from typing import Any, Dict, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from .base import BaseComposerModel


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        num_lstm_layers: int = 1,
        num_fc_layers: int = 4,
        bidirectional: bool = False,
    ) -> None:
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc_layers = nn.Sequential()
        for i in range(num_fc_layers):
            self.fc_layers.add_module(
                f"fc{i}", nn.Linear(hidden_dim, hidden_dim)
            )
            self.fc_layers.add_module(f"relu{i}", nn.ReLU())

        self._to_mean = nn.Linear(hidden_dim, latent_dim)
        self._to_lnvar = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self, sequence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, (h, c) = self.lstm(sequence)

        if self.bidirectional:
            z = h[:2, :, :].mean(dim=0)
        else:
            z = h[0]

        z = self.fc_layers(z)
        mean = self._to_mean(z)
        logvar = self._to_lnvar(z)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_lstm_layers: int = 1,
        num_fc_layers: int = 4,
        bidirectional: bool = False,
    ) -> None:
        super(Decoder, self).__init__()
        self.reverse_latent = nn.Linear(latent_dim, hidden_dim)
        self.fc_layers = nn.Sequential()
        for i in range(num_fc_layers):
            self.fc_layers.add_module(
                f"fc{i}", nn.Linear(hidden_dim, hidden_dim)
            )
            self.fc_layers.add_module(f"relu{i}", nn.ReLU())
        lstm_input_dim = hidden_dim + condition_dim
        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(
        self, latent: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        z = self.reverse_latent(latent)
        z = self.fc_layers(z)

        seq_len = condition.shape[1]
        z = z.unsqueeze(1)
        z = z.repeat(1, seq_len, 1)

        z_seq = torch.cat([z, condition], dim=2)
        output, _ = self.lstm(z_seq)
        x_hat = self.output_layer(output)
        return x_hat  # type: ignore


class CVAEModel(BaseComposerModel):
    """CVAE model to generate melody from chord progression."""

    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_lstm_layers: int = 2,
        num_fc_layers: int = 2,
        bidirectional: bool = False,
    ) -> None:
        super(CVAEModel, self).__init__()
        self.encoder = Encoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_lstm_layers=num_lstm_layers,
            num_fc_layers=num_fc_layers,
            bidirectional=bidirectional,
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            num_lstm_layers=num_lstm_layers,
            num_fc_layers=num_fc_layers,
            bidirectional=bidirectional,
        )

    def reparameterization(
        self, mean: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent = mean + eps * std
        return latent

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = self.encoder(x)
        return mean, logvar

    def decode(
        self, latent: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        x_hat = self.decoder(latent, condition)
        return x_hat  # type: ignore

    def forward(
        self, melody: torch.Tensor, chord_prog: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encode(melody)
        latent = self.reparameterization(mean, logvar)
        x_hat = self.decode(latent, condition=chord_prog)
        x_hat = x_hat.permute(0, 2, 1)
        return x_hat, mean, logvar

    def generate(
        self, x: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        z = torch.empty(1)
        return z


class CVAELoss(nn.Module):
    def __init__(self, kld_weight: float = 1e-3) -> None:
        super(CVAELoss, self).__init__()
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


class CVAEPLModule(pl.LightningModule):
    def __init__(
        self,
        input_dim: int = 49,
        condition_dim: int = 12,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_lstm_layers: int = 2,
        num_fc_layers: int = 2,
        bidirectional: bool = False,
    ) -> None:
        super(CVAEPLModule, self).__init__()

        self.model = CVAEModel(
            input_dim=input_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_lstm_layers=num_lstm_layers,
            num_fc_layers=num_fc_layers,
            bidirectional=bidirectional,
        )
        self.criterion = CVAELoss()

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=3e-4,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[1000, 1500]
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x, condition, label = batch
        x_hat, mean, logvar = self.model(x, condition)  # type: ignore

        loss = self.criterion(label, x_hat, mean, logvar)
        self.log("train_loss", loss, prog_bar=True)
        return loss  # type: ignore
