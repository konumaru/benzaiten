import glob
import os
import subprocess
from typing import List, Tuple

import hydra
import joblib
import music21
import numpy as np
from hydra import compose, initialize
from omegaconf import DictConfig
from rich.progress import track

from config import Config
from utils import (
    add_rest_nodes,
    chord_seq_to_chroma,
    divide_seq,
    make_note_and_chord_seq_from_musicxml,
    note_seq_to_onehot,
)


def get_music_xml(cfg: Config) -> None:
    subprocess.run(
        "echo -n Download Omnibook MusicXML ...", text=True, shell=True
    )

    xml_url = cfg.preprocess.xml_url
    command = "wget " + xml_url
    subprocess.run(command, text=True, shell=True, capture_output=True)

    zip_file = os.path.basename(xml_url)
    command = "unzip -q " + zip_file
    subprocess.run(command, text=True, shell=True)

    command = str("rm " + zip_file)
    subprocess.run(command, text=True, shell=True)

    xml_dir = os.path.join(cfg.benzaiten.root_dir, cfg.benzaiten.xml_dir)
    os.makedirs(xml_dir, exist_ok=True)
    command = str("mv " + "Omnibook\\ xml/*.xml " + xml_dir)
    subprocess.run(command, text=True, shell=True)

    command = str("rm -rf Omnibook\\ xml")
    subprocess.run(command, text=True, shell=True)

    command = str("rm -rf __MACOSX")
    subprocess.run(command, text=True, shell=True)

    print(" done.")


def extract_features(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    data_all: List[float] = []
    label_all: List[float] = []

    xml_dir = os.path.join(cfg.benzaiten.root_dir, cfg.benzaiten.xml_dir)
    os.makedirs(xml_dir, exist_ok=True)

    for xml_file in track(
        glob.glob(xml_dir + "/*.xml"),
        description="Extract features from MusixXML: ",
    ):
        score = music21.converter.parse(xml_file)
        key = score.analyze("key")
        if key.mode == cfg.feature.key_mode:  # type: ignore
            inter = music21.interval.Interval(
                key.tonic,  # type: ignore
                music21.pitch.Pitch(cfg.feature.key_root),
            )
            score = score.transpose(inter)
            note_seq, chord_seq = make_note_and_chord_seq_from_musicxml(
                score,
                cfg.feature.total_measures,
                cfg.feature.n_beats,
                cfg.feature.beat_reso,
            )

            onehot_seq = note_seq_to_onehot(
                note_seq,
                cfg.feature.notenum_thru,
                cfg.feature.notenum_from,
            )
            onehot_seq = add_rest_nodes(onehot_seq)
            chroma_seq = chord_seq_to_chroma(chord_seq)
            divide_seq(cfg, onehot_seq, chroma_seq, data_all, label_all)

    return np.array(data_all), np.array(label_all)


def save_features(
    cfg: Config, data_all: np.ndarray, label_all: np.ndarray
) -> None:
    feat_dir = os.path.join(cfg.benzaiten.root_dir, cfg.benzaiten.feat_dir)
    os.makedirs(feat_dir, exist_ok=True)

    feat_file = os.path.join(feat_dir, cfg.preprocess.feat_file)
    joblib.dump({"data": data_all, "label": label_all}, feat_file)

    print("Save extracted features to " + feat_file)


def get_backing_chord(cfg: Config) -> None:
    """Download backing file (midi) and chord file (csv)."""
    g_drive_url = '"https://drive.google.com/uc?export=download&id="'
    adlib_dir = os.path.join(cfg.benzaiten.root_dir, cfg.benzaiten.adlib_dir)
    os.makedirs(adlib_dir, exist_ok=True)

    backing_url = g_drive_url + cfg.demo.backing_fid
    backing_file = os.path.join(adlib_dir, cfg.demo.backing_file)
    chord_url = g_drive_url + cfg.demo.chord_fid
    chord_file = os.path.join(adlib_dir, cfg.demo.chord_file)

    subprocess.run(
        "echo -n Download backing file for demo ... ", text=True, shell=True
    )
    command = "wget " + backing_url + " -O " + backing_file
    subprocess.run(command, text=True, shell=True, capture_output=True)
    print(" done.")

    subprocess.run(
        "echo -n Download chord file for demo ... ", text=True, shell=True
    )
    command = "wget " + chord_url + " -O " + chord_file
    subprocess.run(command, text=True, shell=True, capture_output=True)
    print(" done.")


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    """Perform preprocess."""
    # Download Omnibook MusicXML
    get_music_xml(cfg)

    # Extract features from MusicXM.
    data_all, label_all = extract_features(cfg)

    # Save extracted features.
    save_features(cfg, data_all, label_all)

    # Download backing file (midi) and chord file (csv)
    get_backing_chord(cfg)


if __name__ == "__main__":
    main()
