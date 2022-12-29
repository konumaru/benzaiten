import os
import random
from typing import Any, Tuple

import joblib
from torch.utils.data import DataLoader, Dataset


class BenzaitenDataset(Dataset):
    """Build Dataset for training."""

    def __init__(self, cfg) -> None:
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
        return self.data_all.shape[0]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Fetch items.

        Args:
            index (int): item index.

        Returns:
            _type_: _description_
        """
        return self.data_all[index], self.label_all[index]
