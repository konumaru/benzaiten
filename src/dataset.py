import os
import random
from typing import Any, Tuple

import joblib
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


class BenzaitenDataset(Dataset):
    """Build Dataset for training."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize class."""
        super().__init__()

        self.cfg = cfg
        feats_dir = os.path.join(
            self.cfg.benzaiten.root_dir, self.cfg.benzaiten.feat_dir
        )
        feat_file = os.path.join(feats_dir, self.cfg.preprocess.feat_file)
        features = joblib.load(feat_file)
        self.data_all = features["data"]
        self.label_all = features["label"]

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data_all)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Fetch items.

        Args:
            index (int): item index.

        Returns:
            _type_: _description_
        """
        return self.data_all[index], self.label_all[index]


def _worker_init_fn(worker_id: int) -> None:
    random.seed(worker_id)


def get_dataloader(cfg: DictConfig) -> DataLoader:
    dataset = BenzaitenDataset(cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.n_batch,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    return dataloader
