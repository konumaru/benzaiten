from typing import Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset


class BenzaitenDataset(Dataset):
    """Build Dataset for training."""

    def __init__(
        self,
        data: np.ndarray,
        condition: np.ndarray,
        label: np.ndarray,
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


def get_dataloader(
    data_filepath: str,
    condition_filepath: str,
    label_filepath: str,
    batch_size: int,
) -> DataLoader:
    data = np.load(data_filepath)
    condition = np.load(condition_filepath)
    label = np.load(label_filepath)

    dataset = BenzaitenDataset(data, condition, label)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
    )
    return dataloader
