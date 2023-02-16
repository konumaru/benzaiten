import os
from pathlib import Path
from typing import Union

import hydra
import midi2audio
import mido
import numpy as np
import torch

from config import Config
from model import EmbeddedLstmVAE, OnehotLstmVAE
from utils import (
    calc_durations,
    calc_notenums_from_pianoroll,
    chord_seq_to_chroma,
    make_chord_seq,
    postprocess_to_diatonic_melody,
    read_chord_file,
)
from utils.visualize import plot_pianoroll, plot_pianorolls


def get_mode(key_filepath: str) -> str:
    with open(key_filepath, "r") as file:
        key = file.read()
    mode = key.split(" ")[1]
    return mode


@torch.no_grad()
def generate_pianoroll(
    chord_filepath: str,
    mode: str,
    model: Union[EmbeddedLstmVAE, OnehotLstmVAE],
) -> np.ndarray:
    chord_prog = read_chord_file(chord_filepath, 8, 4)
    seq_chord = make_chord_seq(chord_prog, 4, 4, 4)
    seq_chord_chroma = chord_seq_to_chroma(seq_chord)

    seq_mode = np.zeros((128, 1))  # major
    if mode == "minor":
        seq_mode = np.ones((128, 1))  # minor

    condition = np.concatenate((seq_chord_chroma, seq_mode), axis=1)
    inputs = torch.from_numpy(condition.astype(np.float32)).unsqueeze(0)

    melody_length = inputs.shape[1]  # 128
    batch_size = int(melody_length / 2)  # 64
    pianoroll = np.zeros((melody_length, 49))

    for i in range(0, melody_length, batch_size):
        latent_rand = torch.rand(1, model.latent_dim)
        y_new = model.decode(latent_rand, inputs[:, i : i + batch_size])

        y_new = y_new.softmax(dim=2).cpu().detach().numpy()
        pianoroll[i : i + batch_size, :] = y_new[0]

    return pianoroll


def make_midi(
    save_filepath: str, backing_filepath: str, pianoroll: np.ndarray
) -> None:
    notenums = calc_notenums_from_pianoroll(pianoroll, 36)
    notenums, durations = calc_durations(notenums)
    notenums = postprocess_to_diatonic_melody(notenums)

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
                    velocity=110,
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


def get_model(cfg: Config) -> Union[OnehotLstmVAE, EmbeddedLstmVAE]:
    assert cfg.exp.name in ("working", "onehot", "embedded")
    train_output_dir = Path(
        os.path.join(
            cfg.benzaiten.root_dir, cfg.benzaiten.train_dir, cfg.exp.name
        )
    )

    if cfg.exp.name == "onehot":
        return OnehotLstmVAE.load_from_checkpoint(
            checkpoint_path=str(
                train_output_dir / cfg.benzaiten.model_filename
            ),
            hparams_file=str(train_output_dir / "config.yaml"),
        )
    elif cfg.exp.name == "embedded":
        return EmbeddedLstmVAE.load_from_checkpoint(
            checkpoint_path=str(
                train_output_dir / cfg.benzaiten.model_filename
            ),
            hparams_file=str(train_output_dir / "config.yaml"),
        )
    else:
        return OnehotLstmVAE.load_from_checkpoint(
            checkpoint_path=str(
                train_output_dir / cfg.benzaiten.model_filename
            ),
            hparams_file=str(train_output_dir / "config.yaml"),
        )


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:

    model = get_model(cfg)

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

    mode = get_mode(str(competition_dir / smpl_name / f"{smpl_name}_key.txt"))

    pianorolls = []
    for i in range(cfg.generate.num_output):
        pianoroll = generate_pianoroll(
            str(competition_dir / smpl_name / f"{smpl_name}_chord.csv"),
            mode,
            model,
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

    plot_pianorolls(str(output_dir / "all_pianoroll.png"), pianorolls)


if __name__ == "__main__":
    main()
