from pathlib import Path
from typing import List

import hydra
import joblib
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from config import Config
from dataset import get_dataloader
from model import CVAEPLModule

# TODO:
# - [x] modelのconfigをhydraで管理する
# - [ ] modelのstate_dict.pthを保存する
# - [ ] 出力ファイル等のリファクタリング


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg), flush=True)

    # TODO: preprocess.pyで複数モデルの入力形式に対応した特徴量の保存方法に変更する
    features = joblib.load("./data/feats/benzaiten_feats.pkl")
    data_all = features["data"]
    label_all = features["label"]
    note_seq = data_all[:, :, :49]
    chord_seq = data_all[:, :, -12:]

    dataloader = get_dataloader(note_seq, chord_seq, label_all)
    model_config = cfg.cvae_model
    model = CVAEPLModule(**model_config)  # type: ignore

    # output_dir = Path(f"./data/train/{cfg.exp.name}")
    output_dir = Path("./CVAE")
    csv_logger = CSVLogger(str(output_dir / "logs"))
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="{epoch}-{train_loss:.4f}",
        monitor="train_loss",
        save_top_k=1,
    )
    callbacks: List[Callback] = [checkpoint_callback]
    trainer = pl.Trainer(
        logger=csv_logger,
        max_epochs=10,
        accelerator="gpu",
        callbacks=callbacks,
        gradient_clip_val=1.0,
        amp_backend="native",
    )
    trainer.fit(model=model, train_dataloaders=dataloader)

    model = CVAEPLModule.load_from_checkpoint(
        checkpoint_path=checkpoint_callback.best_model_path,
        **model_config,  # type: ignore
    )
    torch.save(
        model.model.cpu().state_dict(), str(output_dir / "state_dict.pt")
    )


if __name__ == "__main__":
    main()
