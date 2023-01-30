from typing import Any, List

import matplotlib.pyplot as plt
import midi2audio
import mido
import numpy as np
import torch
import torch.nn as nn

from model.cvae import Chord2Melody
from utils import (
    calc_durations,
    calc_notenums_from_pianoroll,
    calc_xy,
    chord_seq_to_chroma,
    extract_seq,
    make_chord_seq,
    make_empty_pianoroll,
    read_chord_file,
)


@torch.no_grad()
def generate_melody(
    model: nn.Module, chroma_vec: np.ndarray, device: torch.device
) -> np.ndarray:
    piano_roll = make_empty_pianoroll(chroma_vec.shape[0], 84, 36)
    beat_width = 4 * 4
    for i in range(0, 8, 4):
        onehot_vectors, chord_vectors = extract_seq(
            i, piano_roll, chroma_vec, 4, beat_width
        )
        feature_np, _ = calc_xy(onehot_vectors, chord_vectors)
        feature = torch.from_numpy(feature_np).to(device).float()
        feature = feature.unsqueeze(0)

        # NOTE: CVAEで乱数を入力値にしたときの生成方法
        # rand_latent = torch.rand(1, 128).to("cuda")

        if i == 0:
            rand_latent = torch.rand(1, 128).to("cuda")
            chord_prog = feature[:, :, -12:]
            y_new = model.decode(rand_latent, chord_prog)

            y_new = y_new.softmax(dim=2).cpu().detach().numpy()
            y_new = y_new[0].T
        else:
            chord_prog = feature[:, :, -12:]
            gen_melody = (
                torch.from_numpy(piano_roll.astype(np.float32))
                .to(device)
                .unsqueeze(0)
            )
            y_new, _, _ = model(gen_melody, chord_prog)

            y_new = y_new.permute(0, 2, 1)
            y_new = y_new.softmax(dim=2).cpu().detach().numpy()
            y_new = y_new[0].T

        index_from = i * 4 * 4
        piano_roll[index_from : index_from + y_new.shape[1], :] = y_new.T

    plt.matshow(np.transpose(piano_roll))
    plt.savefig("piano_roll.png")
    return piano_roll


def make_midi(
    backing_file: str, notenums: List[Any], durations: List[int]
) -> mido.MidiFile:
    beat_reso = 4
    n_beats = 4
    transpose = 12
    intro_blank_measures = 4

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


def generate_midi(
    model: nn.Module,
    backing_file: str,
    chord_file: str,
    device: torch.device,
) -> mido.MidiFile:
    """Synthesize melody with a trained model.
    Args:
        chord_file: a file of chord sequence (csv)
    Returns:
        midi: generated midi data
    """
    # NOTE: Generate piano_roll with pretrained model.
    chord_prog = read_chord_file(chord_file, 8, 4)
    chord_seq = make_chord_seq(chord_prog, 4, 4, 4)
    chroma_vec = chord_seq_to_chroma(chord_seq)
    piano_roll = generate_melody(model, chroma_vec, device)

    notenums = calc_notenums_from_pianoroll(piano_roll, 36)
    notenums, durations = calc_durations(notenums)
    midi = make_midi(backing_file, notenums, durations)
    return midi


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Chord2Melody(
        input_dim=49, latent_dim=128, hidden_dim=1024, condition_dim=12
    )
    model.load_state_dict(
        torch.load("/workspace/notebook/tmp/state_dict_v2.pt")
    )
    model.to(device)
    midi = generate_midi(
        model,
        "/workspace/data/comp_inputs/sample1/sample1_backing.mid",
        "/workspace/data/comp_inputs/sample1/sample1_chord.csv",
        device,
    )
    midi_file = "output.midi"
    midi.save(midi_file)
    fluid_synth = midi2audio.FluidSynth(
        sound_font="/usr/share/sounds/sf2/FluidR3_GM.sf2"
    )
    fluid_synth.midi_to_audio("output.midi", "output.wav")


if __name__ == "__main__":
    main()
