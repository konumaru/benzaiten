from typing import List

import matplotlib.pyplot as plt
import numpy as np


def plot_pianoroll(save_filepath: str, pianoroll: np.ndarray) -> None:
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Generated Melody.")
    ax.matshow(pianoroll.T)
    ax.set_xlabel("time")
    ax.set_ylabel("pitch")
    ax.invert_yaxis()
    fig.savefig(save_filepath)


def plot_pianorolls(save_filepath: str, pianorolls: List[np.ndarray]) -> None:
    num_plot = len(pianorolls)
    fig, axes = plt.subplots(num_plot, 1, figsize=(6, num_plot * 3))

    for i, ax in enumerate(axes):
        ax.set_title(f"Molody {i}")
        ax.matshow(np.transpose(pianorolls[i]))
        ax.set_xlabel("time")
        ax.set_ylabel("pitch")
        ax.invert_yaxis()

    fig.suptitle("Generateed Melodies.")
    fig.tight_layout()
    plt.savefig(save_filepath)
