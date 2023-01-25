from typing import Any, Tuple

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

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        return self.data[index], self.condition[index], self.label[index]


def get_dataloader(
    data: np.ndarray,
    condition: np.ndarray,
    label: np.ndarray,
    batch_size: int = 32,
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
