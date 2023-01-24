from typing import Any, Dict, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from .model.cvae import CVAELoss, CVAEModel


class CVAETrainer(pl.LightningModule):
    def __init__(
        self,
        ckpt_dirpath: str,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        condition_dim: int,
        num_lstm_layers: int = 2,
        num_fc_layers: int = 2,
        bidirectional: bool = False,
    ) -> None:
        super(CVAETrainer, self).__init__()

        self.ckpt_dirpath = ckpt_dirpath
        self.model = CVAEModel(
            input_dim,
            hidden_dim,
            latent_dim,
            condition_dim,
            num_lstm_layers,
            num_fc_layers,
            bidirectional,
        )
        self.criterion = CVAELoss()

    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.ckpt_dirpath,
            filename="{epoch}-{train_loss:.4f}",
            monitor="train_loss",
            save_top_k=1,
        )
        return [checkpoint_callback]

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
