import os
from pathlib import Path
from typing import Any, List

import hydra
import matplotlib.pyplot as plt
import midi2audio
import mido
import numpy as np
import torch

from config import Config
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
    model: Chord2Melody, chroma_vec: np.ndarray, pianoroll_filepath: str
) -> np.ndarray:
    piano_roll = make_empty_pianoroll(chroma_vec.shape[0], 84, 36)
    beat_width = 4 * 4
    for i in range(0, 8, 4):
        onehot_vectors, chord_vectors = extract_seq(
            i, piano_roll, chroma_vec, 4, beat_width
        )
        feature_np, _ = calc_xy(onehot_vectors, chord_vectors)
        feature = torch.from_numpy(feature_np).float()
        feature = feature.unsqueeze(0)

        # NOTE: CVAEで乱数を入力値にしたときの生成方法
        # rand_latent = torch.rand(1, 128).to("cuda")

        if i == 0:
            rand_latent = torch.rand(1, 128)
            chord_prog = feature[:, :, -12:]
            y_new = model.decode(rand_latent, chord_prog)

        else:
            chord_prog = feature[:, :, -12:]
            gen_melody = torch.from_numpy(
                piano_roll.astype(np.float32)
            ).unsqueeze(0)
            y_new, _, _ = model(gen_melody, chord_prog)

        y_new = y_new.softmax(dim=2).cpu().detach().numpy()
        y_new = y_new[0].T

        index_from = i * 4 * 4
        piano_roll[index_from : index_from + y_new.shape[1], :] = y_new.T

    plt.matshow(np.transpose(piano_roll))
    plt.savefig(pianoroll_filepath)
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
    model: Chord2Melody,
    backing_file: str,
    chord_file: str,
    pianoroll_filepath: str,
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
    piano_roll = generate_melody(model, chroma_vec, pianoroll_filepath)

    notenums = calc_notenums_from_pianoroll(piano_roll, 36)
    notenums, durations = calc_durations(notenums)
    midi = make_midi(backing_file, notenums, durations)
    return midi


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    train_output_dir = Path(
        os.path.join(
            cfg.benzaiten.root_dir, cfg.benzaiten.train_dir, cfg.exp.name
        )
    )
    model = Chord2Melody.load_from_checkpoint(
        checkpoint_path=str(train_output_dir / cfg.benzaiten.model_filename),
        hparams_file=str(train_output_dir / "config.yaml"),
    )
    competition_dir = Path(
        os.path.join(cfg.benzaiten.root_dir, cfg.benzaiten.competition_dir)
    )
    smpl_name = cfg.benzaiten.sample_name
    output_dir = Path(
        os.path.join(
            cfg.benzaiten.root_dir,
            cfg.benzaiten.generated_dir,
            cfg.benzaiten.sample_name,
            cfg.exp.name,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    midi = generate_midi(
        model,
        str(competition_dir / smpl_name / f"{smpl_name}_backing.mid"),
        str(competition_dir / smpl_name / f"{smpl_name}_chord.csv"),
        str(output_dir / cfg.benzaiten.pianoroll_filename),
    )
    midi_filepath = str(output_dir / cfg.benzaiten.midi_filename)
    midi.save(midi_filepath)
    fluid_synth = midi2audio.FluidSynth(
        sound_font="/usr/share/sounds/sf2/FluidR3_GM.sf2"
    )
    wav_filepath = str(output_dir / cfg.benzaiten.wav_filename)
    fluid_synth.midi_to_audio(midi_filepath, wav_filepath)


if __name__ == "__main__":
    main()
