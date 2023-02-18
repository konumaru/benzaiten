import os
import shutil
from pathlib import Path
from typing import List, Tuple, Union

import hydra
import numpy as np
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from config import Config
from dataset import get_dataloader
from model import EmbeddedLstmVAE, OnehotLstmVAE


def get_dataloader_and_model(
    cfg: Config,
) -> Tuple[DataLoader, Union[OnehotLstmVAE, EmbeddedLstmVAE]]:
    assert cfg.exp.name in ("working", "onehot", "embedded")

    seq_notenum = np.load(cfg.feature.notenum_filepath)
    seq_note_onehot = np.load(cfg.feature.note_onehot_filepath)
    seq_chord_chroma = np.load(cfg.feature.chord_chroma_filepath)

    mode = np.load(cfg.feature.mode_filepath)
    seq_mode = np.tile(mode, (1, 64, 1)).transpose((2, 1, 0))

    condition = np.concatenate((seq_chord_chroma, seq_mode), axis=2)

    if cfg.exp.name == "onehot":
        dataloader = get_dataloader(
            seq_note_onehot.astype(np.float32),
            condition.astype(np.float32),
            seq_notenum.astype(np.int64),
            batch_size=cfg.train.batch_size,
        )
        model = OnehotLstmVAE(**dict(cfg.onehot_model))  # type: ignore
    elif cfg.exp.name == "embedded":
        dataloader = get_dataloader(
            seq_notenum.astype(np.int64),
            condition.astype(np.float32),
            seq_notenum.astype(np.int64),
            batch_size=cfg.train.batch_size,
        )
        model = EmbeddedLstmVAE(**dict(cfg.embedded_model))  # type: ignore
    else:
        dataloader = get_dataloader(
            seq_note_onehot.astype(np.float32),
            condition.astype(np.float32),
            seq_notenum.astype(np.int64),
            batch_size=cfg.train.batch_size,
        )
        model = OnehotLstmVAE(**dict(cfg.onehot_model))  # type: ignore

    return dataloader, model


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    dataloader, model = get_dataloader_and_model(cfg)

    output_dir = Path(
        os.path.join(
            cfg.benzaiten.root_dir, cfg.benzaiten.train_dir, cfg.exp.name
        )
    )
    csv_logger = CSVLogger(str(output_dir / "logs"))
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="best_model",
        monitor="train_loss",
        save_top_k=1,
    )
    callbacks: List[Callback] = [checkpoint_callback]
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
    OmegaConf.save(cfg, output_dir / "config_hydra.yaml")
    # shutil.copyfile(
    #     checkpoint_callback.best_model_path,
    #     str(output_dir / cfg.benzaiten.model_filename),
    # )


if __name__ == "__main__":
    main()
