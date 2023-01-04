import os

import hydra
import matplotlib.pyplot as plt
import midi2audio
import mido
import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax

from config import Config
from model import Seq2SeqMelodyComposer
from utils import (
    calc_durations,
    calc_notenums_from_pianoroll,
    calc_xy,
    chord_seq_to_chroma,
    extract_seq,
    make_chord_seq,
    make_empty_pianoroll,
    make_midi,
    read_chord_file,
)


@torch.no_grad()
def generate_melody(
    cfg: Config, model: nn.Module, chroma_vec: np.ndarray, device: torch.device
) -> np.ndarray:
    """Perform a inference step to generate melody (piano-roll) data.
    Args:
        chroma_vec : sequence of many-hot (chroma) vectors
    Returns:
        piano_roll (numpy.ndarray): generated melody
    """
    piano_roll = make_empty_pianoroll(
        chroma_vec.shape[0],
        cfg.feature.notenum_thru,
        cfg.feature.notenum_from,
    )
    beat_width = cfg.feature.n_beats * cfg.feature.beat_reso
    for i in range(0, cfg.feature.melody_length, cfg.feature.unit_measures):
        onehot_vectors, chord_vectors = extract_seq(
            i, piano_roll, chroma_vec, cfg.feature.unit_measures, beat_width
        )
        feature_np, _ = calc_xy(onehot_vectors, chord_vectors)
        feature = torch.from_numpy(feature_np).to(device).float()
        feature = feature.unsqueeze(0)
        y_new = model(feature)
        y_new = y_new.to("cpu").detach().numpy().copy()
        y_new = softmax(y_new, axis=-1)
        index_from = i * (cfg.feature.n_beats * cfg.feature.beat_reso)
        piano_roll[index_from : index_from + y_new[0].shape[0], :] = y_new[0]

    plt.matshow(np.transpose(piano_roll))
    png_file = os.path.join(
        cfg.benzaiten.root_dir,
        cfg.benzaiten.adlib_dir,
        cfg.demo.pianoroll_file,
    )
    plt.savefig(png_file)
    return piano_roll


def generate_midi(
    cfg: Config, model: nn.Module, chord_file: str, device: torch.device
) -> mido.MidiFile:
    """Synthesize melody with a trained model.
    Args:
        chord_file: a file of chord sequence (csv)
    Returns:
        midi: generated midi data
    """
    chord_prog = read_chord_file(
        chord_file, cfg.feature.melody_length, cfg.feature.n_beats
    )
    chord_seq = make_chord_seq(
        chord_prog,
        cfg.feature.n_beats,
        cfg.feature.n_beats,
        cfg.feature.beat_reso,
    )
    chroma_vec = chord_seq_to_chroma(chord_seq)
    piano_roll = generate_melody(cfg, model, chroma_vec, device)
    notenums = calc_notenums_from_pianoroll(
        piano_roll, cfg.feature.notenum_from
    )
    notenums, durations = calc_durations(notenums)
    midi = make_midi(cfg, notenums, durations)
    return midi


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    """Perform ad-lib melody synthesis."""

    # setup network and load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckkpt_dir = os.path.join(cfg.benzaiten.root_dir, cfg.demo.chkpt_dir)
    checkpoint = os.path.join(ckkpt_dir, cfg.demo.chkpt_file)
    model = Seq2SeqMelodyComposer(cfg, device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()  # turn on eval mode

    # generate ad-lib melody in midi format
    chord_file = os.path.join(
        cfg.benzaiten.root_dir, cfg.benzaiten.adlib_dir, cfg.demo.chord_file
    )
    midi_file = os.path.join(
        cfg.benzaiten.root_dir, cfg.benzaiten.adlib_dir, cfg.demo.midi_file
    )
    midi = generate_midi(cfg, model, chord_file, device)
    midi.save(midi_file)

    # export midi to wav
    wav_file = os.path.join(
        cfg.benzaiten.root_dir, cfg.benzaiten.adlib_dir, cfg.demo.wav_file
    )
    fluid_synth = midi2audio.FluidSynth(sound_font=cfg.demo.sound_font)
    fluid_synth.midi_to_audio(midi_file, wav_file)


if __name__ == "__main__":
    main()
