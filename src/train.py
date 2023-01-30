from pathlib import Path
from typing import List

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from dataset import get_dataloader
from model.cvae import PLModule


def main() -> None:
    # ====================
    # Dataset & Dataloader
    # ====================
    dataloader = get_dataloader()

    # ====================
    # Model & Train
    # ====================
    output_dir = Path("tmp")
    csv_logger = CSVLogger(str(output_dir / "logs"))
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="{epoch}-{train_loss:.4f}",
        monitor="train_loss",
        save_top_k=1,
    )
    callbacks: List[Callback] = [checkpoint_callback]
    model = PLModule()
    trainer = pl.Trainer(
        logger=csv_logger,
        max_epochs=10,  # 2000,
        accelerator="gpu",
        callbacks=callbacks,
        gradient_clip_val=1.0,
    )
    trainer.fit(model=model, train_dataloaders=dataloader)
    model = PLModule.load_from_checkpoint(
        checkpoint_path=checkpoint_callback.best_model_path
    )
    torch.save(
        model.model.cpu().state_dict(), str(output_dir / "state_dict_v2.pt")
    )


if __name__ == "__main__":
    main()
