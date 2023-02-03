import os
from pathlib import Path
from typing import List

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
def generate_pianoroll(model: Chord2Melody, chord_filepath: str) -> np.ndarray:
    chord_prog = read_chord_file(chord_filepath, 8, 4)
    chord_seq = make_chord_seq(chord_prog, 4, 4, 4)
    chroma_vec = chord_seq_to_chroma(chord_seq)

    pianoroll = make_empty_pianoroll(chroma_vec.shape[0], 84, 36)
    beat_width = 4 * 4
    for i in range(0, 8, 4):
        onehot_vectors, chord_vectors = extract_seq(
            i, pianoroll, chroma_vec, 4, beat_width
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
                pianoroll.astype(np.float32)
            ).unsqueeze(0)
            y_new, _, _ = model(gen_melody, chord_prog)

        y_new = y_new.softmax(dim=2).cpu().detach().numpy()
        y_new = y_new[0].T

        index_from = i * 4 * 4
        pianoroll[index_from : index_from + y_new.shape[1], :] = y_new.T

    return pianoroll


def plot_pianoroll(save_filepath: str, pianoroll: np.ndarray) -> None:
    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Generated Melody.")
    ax.matshow(pianoroll.T)
    ax.set_xlabel("time")
    ax.set_ylabel("pitch")
    ax.invert_yaxis()
    fig.savefig(save_filepath)


def plot_all_pianoroll(
    save_filepath: str, pianorolls: List[np.ndarray]
) -> None:
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


def make_midi(
    save_filepath: str, backing_filepath: str, pianoroll: np.ndarray
) -> None:
    notenums = calc_notenums_from_pianoroll(pianoroll, 36)
    notenums, durations = calc_durations(notenums)

    # NOTE: make midi.
    beat_reso = 4
    n_beats = 4
    transpose = 12
    intro_blank_measures = 4

    midi = mido.MidiFile(backing_filepath)
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

    midi.save(save_filepath)


def make_wav(save_filepath: str, midi_filepath: str) -> None:
    fluid_synth = midi2audio.FluidSynth(
        sound_font="/usr/share/sounds/sf2/FluidR3_GM.sf2"
    )
    fluid_synth.midi_to_audio(midi_filepath, save_filepath)


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

    smpl_name = cfg.sample_name
    competition_dir = Path(
        os.path.join(cfg.benzaiten.root_dir, cfg.benzaiten.competition_dir)
    )
    output_dir = Path(
        os.path.join(
            cfg.benzaiten.root_dir,
            cfg.benzaiten.generated_dir,
            cfg.sample_name,
            cfg.exp.name,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pianoroll").mkdir(parents=True, exist_ok=True)
    (output_dir / "midi").mkdir(parents=True, exist_ok=True)
    (output_dir / "wav").mkdir(parents=True, exist_ok=True)

    pianorolls = []
    for i in range(cfg.generate.num_output):
        pianoroll = generate_pianoroll(
            model,
            str(competition_dir / smpl_name / f"{smpl_name}_chord.csv"),
        )
        pianorolls.append(pianoroll)
        plot_pianoroll(
            str(
                output_dir
                / f"pianoroll/{i}_{cfg.benzaiten.pianoroll_filename}"
            ),
            pianoroll,
        )

        midi_filepath = str(
            output_dir / f"midi/{i}_{cfg.benzaiten.midi_filename}"
        )
        make_midi(
            midi_filepath,
            str(competition_dir / smpl_name / f"{smpl_name}_backing.mid"),
            pianoroll,
        )
        make_wav(
            save_filepath=str(
                output_dir / f"wav/{i}_{cfg.benzaiten.wav_filename}"
            ),
            midi_filepath=midi_filepath,
        )

    plot_all_pianoroll(str(output_dir / "all_pianoroll.png"), pianorolls)


if __name__ == "__main__":
    main()
