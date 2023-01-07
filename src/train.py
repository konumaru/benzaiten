from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from config import Config
from dataset import get_dataloader
from factory import get_loss, get_lr_scheduler
from model import Seq2SeqMelodyComposer


class PLModule(pl.LightningModule):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = Seq2SeqMelodyComposer(cfg)
        self.loss_fn = get_loss(cfg, self.model)

        self.save_hyperparameters()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        source, target = batch
        preds = {
            "source": source.float(),
            "target": target.long(),
        }
        loss: torch.Tensor = self.loss_fn(preds)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer_class = getattr(
            torch.optim, self.cfg.training.optimizer.name
        )
        optimizer = optimizer_class(
            self.parameters(), **self.cfg.training.optimizer.params
        )

        scheduler = get_lr_scheduler(self.cfg, optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    """Perform model training."""
    print(OmegaConf.to_yaml(cfg), flush=True)  # dump configuration

    dataloader = get_dataloader(cfg)
    model = PLModule(cfg)

    output_dir = Path(f"./data/train/{cfg.exp.name}")
    csv_logger = CSVLogger(str(output_dir / "logs"), name="exp_00100")
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        filename="{epoch}-{train_loss:.4f}",
        monitor="train_loss",
        save_top_k=1,
    )
    callbacks: List[Callback] = [checkpoint_callback]
    trainer = pl.Trainer(
        logger=csv_logger,
        limit_predict_batches=100,
        max_epochs=cfg.training.n_epoch,
        accelerator="gpu",
        callbacks=callbacks,
        gradient_clip_val=1.0,
        amp_backend="native",
    )
    trainer.fit(model=model, train_dataloaders=dataloader)

    print("\n\n")
    print(checkpoint_callback.best_model_path)
    print(checkpoint_callback.best_model_score)

    model = PLModule.load_from_checkpoint(
        checkpoint_path=checkpoint_callback.best_model_path, cfg=cfg
    )
    torch.save(
        model.model.cpu().state_dict(), str(output_dir / "state_dict.pt")
    )


if __name__ == "__main__":
    main()
