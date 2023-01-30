from typing import Any, Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        num_lstm_layers: int = 1,
        num_fc_layers: int = 4,
        bidirectional: bool = False,
    ) -> None:
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc_layers = nn.ModuleList([])
        self.fc_layers += [
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_fc_layers)
        ]

        self._to_mean = nn.Linear(hidden_dim, latent_dim)
        self._to_lnvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (h, c) = self.lstm(x)

        if self.bidirectional:
            z = h[:2, :, :].mean(dim=0)
        else:
            z = h[0]

        for fc in self.fc_layers:
            z = fc(z)
            z = nn.functional.relu(z)

        mean = self._to_mean(z)
        logvar = self._to_lnvar(z)
        return mean, logvar


class LSTMDecoder(nn.Module):
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
        super(LSTMDecoder, self).__init__()
        self.reverse_latent = nn.Linear(latent_dim, hidden_dim)
        self.fc_layers = nn.ModuleList([])
        self.fc_layers += [
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_fc_layers)
        ]
        lstm_input_dim = hidden_dim + condition_dim
        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, latent, condition) -> torch.Tensor:
        z = self.reverse_latent(latent)
        for fc in self.fc_layers:
            z = fc(z)
            z = nn.functional.relu(z)

        seq_len = condition.shape[1]
        z = z.unsqueeze(1)
        z = z.repeat(1, seq_len, 1)

        z_seq = torch.cat([z, condition], dim=2)
        output, _ = self.lstm(z_seq)
        x_hat = self.output_layer(output)
        return x_hat


class Chord2Melody(nn.Module):
    """CVAE model to generate melody from chord progression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        condition_dim: int,
        num_lstm_layers: int = 2,
        num_fc_layers: int = 2,
        bidirectional: bool = False,
    ) -> None:
        super(Chord2Melody, self).__init__()
        self.encoder = LSTMEncoder(
            input_dim,
            latent_dim,
            hidden_dim,
            num_lstm_layers,
            num_fc_layers,
            bidirectional=bidirectional,
        )
        self.decoder = LSTMDecoder(
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

    def encode(self, x):
        mean, logvar = self.encoder(x)
        return mean, logvar

    def decode(self, latent, condition):
        x_hat = self.decoder(latent, condition)
        return x_hat

    def forward(self, melody, chord_prog):
        mean, logvar = self.encode(melody)
        latent = self.reparameterization(mean, logvar)
        x_hat = self.decode(latent, condition=chord_prog)
        x_hat = x_hat.permute(0, 2, 1)
        return x_hat, mean, logvar


class VAELoss(nn.Module):
    def __init__(self, kld_weight: float = 1e-3) -> None:
        super(VAELoss, self).__init__()
        self.kld_weight = kld_weight

    def forward(self, label, x_hat, mu, logvar):
        recons_loss = nn.functional.cross_entropy(
            x_hat, label, reduction="mean"
        )
        kld_loss = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())
        vae_loss = recons_loss + (self.kld_weight * kld_loss)
        return vae_loss


class PLModule(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = Chord2Melody(
            input_dim=49, latent_dim=128, hidden_dim=1024, condition_dim=12
        )
        self.criterion = VAELoss()

        self.save_hyperparameters()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, condition, label = batch
        x_hat, mean, logvar = self.model(x, condition)

        loss: torch.Tensor = self.criterion(label, x_hat, mean, logvar)
        self.log("train_loss", loss, prog_bar=True)
        return loss

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
