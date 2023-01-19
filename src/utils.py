import csv
from typing import Any, List, Tuple

import mido
import music21
import numpy as np

from config import Config


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


def make_empty_pianoroll(
    length: int, notenum_thru: int, notenum_from: int
) -> np.ndarray:
    return np.zeros((length, notenum_thru - notenum_from + 1))


def calc_notenums_from_pianoroll(
    pianoroll: np.ndarray, notenum_from: int
) -> List[Any]:
    note_nums = []
    for i in range(pianoroll.shape[0]):
        num = np.argmax(pianoroll[i, :])
        note_num = -1 if num == pianoroll.shape[1] - 1 else num + notenum_from
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


def make_midi(
    cfg: Config, backing_file: str, notenums: List[Any], durations: List[int]
) -> mido.MidiFile:
    beat_reso = cfg.feature.beat_reso
    n_beats = cfg.feature.n_beats
    transpose = cfg.feature.transpose
    intro_blank_measures = cfg.feature.intro_blank_measures

    midi = mido.MidiFile(backing_file)
    track = mido.MidiTrack()
    midi.tracks.append(track)

    var = {
        "init_tick": intro_blank_measures * n_beats * midi.ticks_per_beat,
        "cur_tick": 0,
        "prev_tick": 0,
    }
    for i, notenum in enumerate(notenums):
        if notenum > 0:
            var["cur_tick"] = (
                int(i * midi.ticks_per_beat / beat_reso) + var["init_tick"]
            )
            track.append(
                mido.Message(
                    "note_on",
                    note=notenum + transpose,
                    velocity=100,
                    time=var["cur_tick"] - var["prev_tick"],
                )
            )
            var["prev_tick"] = var["cur_tick"]
            var["cur_tick"] = (
                int((i + durations[i]) * midi.ticks_per_beat / beat_reso)
                + var["init_tick"]
            )
            track.append(
                mido.Message(
                    "note_off",
                    note=notenum + transpose,
                    velocity=100,
                    time=var["cur_tick"] - var["prev_tick"],
                )
            )
            var["prev_tick"] = var["cur_tick"]

    return midi


def calc_xy(
    onehot_vectors: np.ndarray, chord_vectors: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    data = np.concatenate(
        [onehot_vectors, chord_vectors],
        axis=1,
    )
    label = np.argmax(onehot_vectors, axis=1)
    return data, label


def divide_seq(
    cfg: Config,
    onehot_seq: np.ndarray,
    chroma_seq: np.ndarray,
    data_all: List,
    label_all: List,
) -> None:
    total_measures = cfg.feature.total_measures
    unit_measures = cfg.feature.unit_measures
    beat_width = cfg.feature.n_beats * cfg.feature.beat_reso
    for i in range(0, total_measures, unit_measures):
        onehot_vector, chord_vector = extract_seq(
            i, onehot_seq, chroma_seq, unit_measures, beat_width
        )
        if np.any(onehot_vector[:, 0:-1] != 0):
            data, label = calc_xy(onehot_vector, chord_vector)
            data_all.append(data)
            label_all.append(label)
