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

        self.data = data
        self.condition = condition
        self.label = label

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def __getitem__(
        self, index: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.data[index], self.condition[index], self.label[index]


def get_dataloader(
    data: np.ndarray,
    condition: np.ndarray,
    label: np.ndarray,
    batch_size: int,
) -> DataLoader:
    dataset = BenzaitenDataset(data, condition, label)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )
    return dataloader
