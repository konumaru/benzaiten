import csv
import os

import mido
import music21
import numpy as np
from omegaconf import DictConfig


def make_note_and_chord_seq_from_musicxml(score, total_measures, n_beats, beat_reso):
    """Generate note column and chord symbol column from MusicXML.

    Args:
        score (_type_): _description_
        total_measures (_type_): _description_
        n_beats (_type_): _description_
        beat_reso (_type_): _description_

    Returns:
        _type_: _description_
    """

    note_seq = [None] * (total_measures * n_beats * beat_reso)
    chord_seq = [None] * (total_measures * n_beats * beat_reso)

    for element in score.parts[0].elements:
        if isinstance(element, music21.stream.Measure):  # type: ignore
            measure_offset = element.offset

            for note in element.notes:
                if isinstance(note, music21.note.Note):
                    onset = measure_offset + note._activeSiteStoredOffset  # type: ignore
                    offset = onset + note._duration.quarterLength  # type: ignore

                    for i in range(int(onset * beat_reso), int(offset * beat_reso + 1)):
                        note_seq[i] = note

                if isinstance(note, music21.harmony.ChordSymbol):
                    chord_offset = measure_offset + note.offset
                    for i in range(
                        int(chord_offset * beat_reso),
                        int((measure_offset + n_beats) * beat_reso + 1),
                    ):
                        chord_seq[i] = note

    return note_seq, chord_seq


def note_seq_to_onehot(note_seq, notenum_thru, notenum_from):
    """_summary_

    Args:
        note_seq (_type_): _description_
        notenum_thru (_type_): _description_
        notenum_from (_type_): _description_

    Returns:
        _type_: _description_
    """

    n_note_width = notenum_thru - notenum_from
    n_note_seq = len(note_seq)
    matrix = np.zeros((n_note_seq, n_note_width))
    for i in range(n_note_seq):
        if note_seq[i] is not None:
            matrix[i, note_seq[i].pitch.midi - notenum_from] = 1
    return matrix


def add_rest_nodes(onehot_seq):
    rest = 1 - np.sum(onehot_seq, axis=1)
    rest = np.expand_dims(rest, 1)
    return np.concatenate([onehot_seq, rest], axis=1)


def extract_seq(index, onehot_seq, chroma_seq, unit_measures, width):
    onehot_vectors = onehot_seq[index * width : (index + unit_measures) * width, :]
    chord_vectors = chroma_seq[index * width : (index + unit_measures) * width, :]
    return onehot_vectors, chord_vectors


def chord_seq_to_chroma(chord_seq):
    matrix = np.zeros((len(chord_seq), 12))
    for i, chord in enumerate(chord_seq):
        if chord is not None:
            for note in chord._notes:
                matrix[i, note.pitch.midi % 12] = 1
    return matrix


def read_chord_file(csv_file, melody_length, n_beats):
    chord_seq = [None] * (melody_length * n_beats)

    with open(csv_file, encoding="utf-8") as file_handler:
        reader = csv.reader(file_handler)
        for row in reader:
            measure_id = int(row[0])
            if measure_id < melody_length:
                beat_id = int(row[1])
                chord_seq[measure_id * 4 + beat_id] = music21.harmony.ChordSymbol(
                    root=row[2], kind=row[3], bass=row[4]
                )

    chord = None
    for i, _chord in enumerate(chord_seq):
        chord = _chord if _chord is not None else chord
        chord_seq[i] = chord

    return chord_seq


def make_chord_seq(chord_prog, division, n_beats, beat_reso):
    time_length = int(n_beats * beat_reso / division)
    seq = [] * (time_length * int(chord_prog))

    for i, chord in enumerate(chord_prog):
        for _t in range(time_length):
            idx = int(i * time_length) + _t
            if isinstance(chord, music21.harmony.ChordSymbol):
                seq[idx] = chord
            else:
                seq[idx] = music21.harmony.ChordSymbol(chord)
    return seq


def make_empty_pianoroll(length, notenum_thru, notenum_from):
    return np.zeros((length, notenum_thru - notenum_from + 1))


def calc_notenums_from_pianoroll(pianoroll, notenum_from):
    note_nums = []
    for i in range(pianoroll.shape[0]):
        num = np.argmax(pianoroll[i, :])
        note_num = -1 if num == pianoroll.shape[1] - 1 else num + notenum_from
        note_nums.append(note_num)
    return note_nums


def calc_durations(notenums):
    note_length = len(notenums)
    duration = [] * note_length
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


def make_midi(cfg: DictConfig, notenums, durations):
    beat_reso = cfg.feature.beat_reso
    n_beats = cfg.feature.n_beats
    transpose = cfg.feature.transpose
    intro_blank_measures = cfg.feature.intro_blank_measures

    backing_file = os.path.join(
        cfg.benzaiten.root_dir, cfg.benzaiten.adlib_dir, cfg.demo.backing_file
    )
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


def calc_xy(onehot_vectors, chord_vectors):

    data = np.concatenate(
        [onehot_vectors, chord_vectors],
        axis=1,
    )
    label = np.argmax(onehot_vectors, axis=1)
    return data, label


def divide_seq(cfg: DictConfig, onehot_seq, chroma_seq, data_all, label_all):
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
