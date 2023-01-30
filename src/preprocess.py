import glob
import os
import subprocess
from typing import List, Tuple, Union

import hydra
import joblib
import music21
import numpy as np
from music21.chord import Chord
from music21.note import Note
from music21.stream.base import Score
from rich.progress import track

from config import Config
from utils import divide_seq


class MusicXMLFeature(object):
    def __init__(
        self,
        xml_file: str,
        key_root: str = "C",
        num_beats: int = 4,
        num_parts_of_beat: int = 4,
        max_measure_num: int = 240,
        min_note_num: int = 36,
        max_note_num: int = 84,
    ) -> None:
        assert key_root in ["C", "D", "E", "F", "G", "A", "B"]

        self.score = self._get_score(xml_file, key_root)
        self.num_beats = num_beats
        self.num_parts_of_beat = num_parts_of_beat
        self.max_measure_num = max_measure_num
        self.min_note_num = min_note_num
        self.max_note_num = max_note_num

    def _get_score(self, xml_file: str, root: str) -> Score:
        score: Score = music21.converter.parse(
            xml_file, format="musicxml"
        )  # type: ignore
        key = score.analyze("key")
        interval = music21.interval.Interval(
            key.tonic, music21.pitch.Pitch(root)  # type: ignore
        )
        score.transpose(interval, inPlace=True)
        return score

    def get_mode(self) -> str:
        key = self.score.analyze("key")
        mode = "None" if key is None else str(key.mode)
        return mode

    def get_note_seq(self) -> List[Union[None, Note]]:
        note_seq: List[Union[None, Note]] = [None] * int(
            self.max_measure_num * self.num_beats * self.num_parts_of_beat
        )

        for measure in self.score.parts[0].getElementsByClass("Measure"):
            for note in measure.getElementsByClass("Note"):
                onset = measure.offset + note._activeSiteStoredOffset
                offset = onset + note._duration.quarterLength

                start_idx = int(onset * self.num_parts_of_beat)
                end_idx = int(offset * self.num_parts_of_beat + 1)

                num_item = int(end_idx - start_idx)
                note_seq[start_idx:end_idx] = [note] * num_item

        return note_seq

    def get_onehot_note_seq(self) -> np.ndarray:
        note_seq = self.get_note_seq()
        note_num_seq = [
            int(n.pitch.midi - self.min_note_num) if n is not None else -1
            for n in note_seq
        ]
        num_note = self.max_note_num - self.min_note_num + 1
        onehot_note_seq = np.identity(num_note)[note_num_seq]
        return onehot_note_seq

    def get_chord_seq(self) -> List[Union[None, Chord]]:
        chord_seq: List[Union[None, Chord]] = [None] * int(
            self.max_measure_num * self.num_beats * self.num_parts_of_beat
        )

        for measure in self.score.parts[0].getElementsByClass("Measure"):
            for note in measure.getElementsByClass("ChordSymbol"):
                offset = measure.offset + note.offset

                start_idx = int(offset * self.num_parts_of_beat)
                end_idx = (
                    int(
                        (measure.offset + self.num_beats)
                        * self.num_parts_of_beat
                    )
                    + 1
                )
                num_item = int(end_idx - start_idx)
                chord_seq[start_idx:end_idx] = [note] * num_item

        return chord_seq

    def get_onehot_chord_seq(self) -> np.ndarray:
        chord_seq = self.get_chord_seq()
        onehot_chord_seq = np.zeros((len(chord_seq), 12))
        for i, chord in enumerate(chord_seq):
            if chord is None:
                continue
            for note in chord._notes:
                onehot_chord_seq[i, note.pitch.midi % 12] = 1
        return onehot_chord_seq


def get_music_xml(cfg: Config) -> None:
    subprocess.run(
        "echo -n Download Omnibook MusicXML ...", text=True, shell=True
    )

    xml_url = "https://homepages.loria.fr/evincent/omnibook/omnibook_xml.zip"
    command = "wget " + xml_url
    subprocess.run(command, text=True, shell=True, capture_output=True)

    zip_file = os.path.basename(xml_url)
    command = "unzip -q " + zip_file
    subprocess.run(command, text=True, shell=True)

    command = str("rm " + zip_file)
    subprocess.run(command, text=True, shell=True)

    xml_dir = os.path.join("/workspace/data", "xml/")
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

    xml_dir = os.path.join("/workspace/data", "xml/")
    for xml_file in track(glob.glob(xml_dir + "/*.xml")):
        feat = MusicXMLFeature(
            xml_file,
            cfg.feature.key_root,
            num_beats=cfg.feature.n_beats,
            num_parts_of_beat=cfg.feature.beat_reso,
            max_measure_num=cfg.feature.total_measures,
            min_note_num=cfg.feature.notenum_from,
            max_note_num=cfg.feature.notenum_thru,
        )

        if feat.get_mode() == "major":
            note_seq = feat.get_onehot_note_seq()
            chord_seq = feat.get_onehot_chord_seq()

            divide_seq(cfg, note_seq, chord_seq, data_all, label_all)

    return np.array(data_all), np.array(label_all)


def save_features(
    cfg: Config, data_all: np.ndarray, label_all: np.ndarray
) -> None:
    feat_dir = os.path.join("/workspace/data", "feats/")
    os.makedirs(feat_dir, exist_ok=True)

    feat_file = os.path.join(feat_dir, "benzaiten_feats.pkl")
    joblib.dump({"data": data_all, "label": label_all}, feat_file)

    print("Save extracted features to " + feat_file)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    # Download Omnibook MusicXML
    get_music_xml(cfg)
    # Extract features from MusicXM.
    data_all, label_all = extract_features(cfg)
    # Save extracted features.
    save_features(cfg, data_all, label_all)


if __name__ == "__main__":
    main()
