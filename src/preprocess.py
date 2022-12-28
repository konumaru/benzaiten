import glob
import os
import subprocess

import joblib
import music21
import numpy as np
from hydra import compose, initialize
from progressbar import progressbar as prg

from utils import (
    add_rest_nodes,
    chord_seq_to_chroma,
    divide_seq,
    make_note_and_chord_seq_from_musicxml,
    note_seq_to_onehot,
)


def main():
    print("hello world")


if __name__ == "__main__":
    main()
