from typing import Tuple

import joblib
import numpy as np
from torch.utils.data import DataLoader, Dataset


class BenzaitenDataset(Dataset):
    """Build Dataset for training."""

    def __init__(
        self, data: np.ndarray, condition: np.ndarray, label: np.ndarray
    ) -> None:
        super().__init__()
        assert len(data) == len(condition)

        self.data = data.astype(np.float32)
        self.condition = condition.astype(np.float32)
        self.label = label.astype(np.int64)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def __getitem__(
        self, index: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.data[index], self.condition[index], self.label[index]


def get_dataloader(batch_size: int) -> DataLoader:
    features = joblib.load("/workspace/data/feature/benzaiten_feats.pkl")
    data_all = features["data"]
    label_all = features["label"]

    note_seq = data_all[:, :, :49]
    chord_seq = data_all[:, :, -12:]
    dataset = BenzaitenDataset(note_seq, chord_seq, label_all)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )
    return dataloader
