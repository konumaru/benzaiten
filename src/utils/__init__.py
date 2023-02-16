import csv
import random
from typing import Any, List, Tuple

import music21
import numpy as np


def extract_seq(
    index: int,
    onehot_seq: np.ndarray,
    chroma_seq: np.ndarray,
    unit_measures: int,
    width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    onehot_vectors = onehot_seq[
        index * width : (index + unit_measures) * width, :
    ]
    chord_vectors = chroma_seq[
        index * width : (index + unit_measures) * width, :
    ]
    return onehot_vectors, chord_vectors


def chord_seq_to_chroma(chord_seq: List[Any]) -> np.ndarray:
    matrix = np.zeros((len(chord_seq), 12))
    for i, chord in enumerate(chord_seq):
        if chord is not None:
            for note in chord._notes:
                matrix[i, note.pitch.midi % 12] = 1
    return matrix


def read_chord_file(
    csv_file: str, melody_length: int, n_beats: int
) -> List[str]:
    chord_seq = [None] * int(melody_length * n_beats)

    with open(csv_file, "r", encoding="utf-8") as file_handler:
        reader = csv.reader(file_handler)
        for row in reader:
            measure_id = int(row[0])
            if measure_id < melody_length:
                beat_id = int(row[1])
                idx = measure_id * 4 + beat_id
                chord_seq[idx] = music21.harmony.ChordSymbol(  # type: ignore
                    root=row[2], kind=row[3], bass=row[4]
                )

    chord = None
    for i, _chord in enumerate(chord_seq):
        chord = _chord if _chord is not None else chord
        chord_seq[i] = chord

    return chord_seq  # type: ignore


def make_chord_seq(
    chord_prog: List, division: int, n_beats: int, beat_reso: int
) -> List[Any]:
    time_length = int(n_beats * beat_reso / division)
    seq = []
    for _, chord in enumerate(chord_prog):
        for _ in range(time_length):
            if isinstance(chord, music21.harmony.ChordSymbol):
                seq.append(chord)
            else:
                seq.append(music21.harmony.ChordSymbol(chord))
    return seq


def calc_notenums_from_pianoroll(
    pianoroll: np.ndarray, notenum_from: int
) -> List[Any]:
    note_nums = []
    for i in range(pianoroll.shape[0]):
        num = np.argmax(pianoroll[i, :])
        note_num = -1 if num == 0 else num + notenum_from - 1
        note_nums.append(note_num)
    return note_nums


def calc_durations(notenums: List[Any]) -> Tuple[List[Any], List[int]]:
    note_length = len(notenums)
    duration = [1] * note_length
    for i in range(note_length):
        k = 1
        while i + k < note_length:
            if notenums[i] > 0 and notenums[i] == notenums[i + k]:
                notenums[i + k] = 0
                duration[i] += 1
            else:
                break
            k += 1
    return notenums, duration


def calc_xy(
    onehot_vectors: np.ndarray, chord_vectors: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    data = np.concatenate(
        [onehot_vectors, chord_vectors],
        axis=1,
    )
    label = np.argmax(onehot_vectors, axis=1)
    return data, label


def make_sequence(
    data: np.ndarray,
    max_seq_len: int,
    drop_last: bool = True,
    pad_value: Any = 0,
) -> np.ndarray:
    _sequence = []
    for i in range(0, len(data), max_seq_len):
        row = list(data[i : (i + max_seq_len)])

        if len(row) == max_seq_len:
            _sequence.append(row)
        elif not drop_last:
            num_pad = max_seq_len - len(row)
            row = row + [pad_value] * num_pad
            _sequence.append(row)

    sequence = np.array(_sequence)
    return sequence


def postprocess_to_diatonic_melody(notenums: List[int]) -> List[int]:
    for i, note in enumerate(notenums):
        key = note % 12

        if (
            key not in {0, 2, 4, 5, 7, 9, 11} and note > 0
        ):  # Cキーとした時のダイアトニックに該当しない場合
            notenums[i] += int(random.random() < 0.5)

    return notenums
