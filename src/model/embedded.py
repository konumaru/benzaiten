from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn

from .loss import VAELoss


class EncoderLSTM(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        latent_dim: int,
        hidden_dim: int,
        num_lstm_layers: int = 1,
        num_fc_layers: int = 4,
        bidirectional: bool = False,
    ) -> None:
        super(EncoderLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
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

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.embed(x)
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
        latent = self.reparameterization(mean, logvar)
        return latent, mean, logvar

    def reparameterization(
        self, mean: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent = mean + eps * std
        return latent


class DecoderLSTM(nn.Module):
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
        super(DecoderLSTM, self).__init__()
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
        lstm_hidden_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_layer = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(
        self, latent: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        z = self.reverse_latent(latent)
        for fc in self.fc_layers:
            z = fc(z)
            z = nn.functional.relu(z)

        seq_len = condition.shape[1]
        z = z.unsqueeze(1)
        z = z.repeat(1, seq_len, 1)

        z_seq = torch.cat([z, condition], dim=2)
        z, _ = self.lstm(z_seq)
        z = self.output_layer(z)
        return z  # type: ignore


class EmbeddedLstmVAE(pl.LightningModule):
    """CVAE model to generate melody from chord progression."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        latent_dim: int,
        condition_dim: int,
        num_lstm_layers: int = 2,
        num_fc_layers: int = 2,
        bidirectional: bool = False,
    ) -> None:
        super(EmbeddedLstmVAE, self).__init__()
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.criterion = VAELoss()
        self.encoder = EncoderLSTM(
            num_embeddings=input_dim,
            embedding_dim=embedding_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_lstm_layers=1,
            num_fc_layers=4,
            bidirectional=False,
        )
        self.decoder = DecoderLSTM(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            num_lstm_layers=num_lstm_layers,
            num_fc_layers=num_fc_layers,
            bidirectional=bidirectional,
        )

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, mean, logvar = self.encoder(x)
        return latent, mean, logvar

    def decode(
        self, latent: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        x_hat = self.decoder(latent, condition)
        return x_hat  # type: ignore

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, mean, logvar = self.encode(x)
        x_hat = self.decode(latent, condition)
        return x_hat, mean, logvar

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x, condition, label = batch
        x_hat, mean, logvar = self(x, condition)

        x_hat = x_hat.permute(0, 2, 1)
        loss: torch.Tensor = self.criterion(label, x_hat, mean, logvar)
        self.log("train_loss", value=loss, prog_bar=True)
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
