import os
import shutil
from pathlib import Path
from typing import List

import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from config import Config
from dataset import get_dataloader
from model.cvae import Chord2Melody


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    dataloader = get_dataloader(batch_size=cfg.train.batch_size)

    output_dir = Path(
        os.path.join(
            cfg.benzaiten.root_dir, cfg.benzaiten.train_dir, cfg.exp.name
        )
    )
    csv_logger = CSVLogger(str(output_dir / "logs"))
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="{epoch}-{train_loss:.4f}",
        monitor="train_loss",
        save_top_k=1,
    )
    callbacks: List[Callback] = [checkpoint_callback]
    # NOTE: model_nameごとに定義を分岐できる
    model = Chord2Melody(**dict(cfg.model))  # type: ignore
    trainer = pl.Trainer(
        logger=csv_logger,
        max_epochs=cfg.train.num_epoch,
        accelerator="gpu",
        callbacks=callbacks,
        gradient_clip_val=cfg.train.grad_clip_val,
        log_every_n_steps=10,
    )
    trainer.fit(model=model, train_dataloaders=dataloader)

    OmegaConf.save(dict(model.hparams), output_dir / "config.yaml")
    shutil.copyfile(
        checkpoint_callback.best_model_path,
        str(output_dir / cfg.benzaiten.model_filename),
    )


if __name__ == "__main__":
    main()
