import glob
import os
import subprocess
from pathlib import Path
from typing import List, Tuple, Union

import hydra
import music21
import numpy as np
from music21.chord import Chord
from music21.note import Note
from music21.stream.base import Score
from rich.progress import track

from config import Config
from utils import make_sequence


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

        self.notes, self.chords = self.get_notes_and_chords()

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

    def get_notes_and_chords(
        self,
    ) -> Tuple[List[Union[None, Note]], List[Union[None, Chord]]]:
        notes = []  # type: ignore
        chords = []  # type: ignore
        for measure in self.score.parts[0].getElementsByClass("Measure"):
            m_notes = [None] * self.num_beats * self.num_parts_of_beat
            for note in measure.getElementsByClass("Note"):
                onset = note._activeSiteStoredOffset
                offset = onset + note._duration.quarterLength

                start_idx = int(onset * self.num_parts_of_beat)
                end_idx = int(offset * self.num_parts_of_beat) + 1
                end_idx = end_idx if end_idx < 16 else 16

                num_item = int(end_idx - start_idx)
                m_notes[start_idx:end_idx] = [note] * num_item
            notes.extend(m_notes)

            m_chords = [None] * self.num_beats * self.num_parts_of_beat
            for chord in measure.getElementsByClass("ChordSymbol"):
                offset = chord.offset

                start_idx = int(offset * self.num_parts_of_beat)
                end_idx = int(self.num_beats * self.num_parts_of_beat) + 1
                end_idx = end_idx if end_idx < 16 else 16

                num_item = int(end_idx - start_idx)
                m_chords[start_idx:end_idx] = [chord] * num_item
            chords.extend(m_chords)

        return notes, chords

    def get_seq_notenum(self) -> np.ndarray:
        # NOTE: 0 is empty note number.
        seq_notenum = [
            int(n.pitch.midi) - self.min_note_num + 1 if n is not None else 0
            for n in self.notes
        ]
        return np.array(seq_notenum)

    def get_seq_note_onehot(self) -> np.ndarray:
        notenum = self.get_seq_notenum()

        num_note = self.max_note_num - self.min_note_num + 1
        seq_note_onehot = np.identity(num_note)[notenum]
        return seq_note_onehot  # type: ignore

    def get_seq_chord_chorma(self) -> np.ndarray:
        onehot_chord_seq = np.zeros((len(self.chords), 12))
        for i, chord in enumerate(self.chords):
            if chord is not None:
                for note in chord._notes:
                    onehot_chord_seq[i, note.pitch.midi % 12] = 1
        return onehot_chord_seq


def get_music_xml(output_dir: str) -> None:
    subprocess.run(
        "echo -n Download Omnibook MusicXML ...", text=True, shell=True
    )

    xml_url = "https://homepages.loria.fr/evincent/omnibook/omnibook_xml.zip"
    command = "wget " + output_dir
    subprocess.run(command, text=True, shell=True, capture_output=True)

    zip_file = os.path.basename(xml_url)
    command = "unzip -q " + zip_file
    subprocess.run(command, text=True, shell=True)

    command = str("rm " + zip_file)
    subprocess.run(command, text=True, shell=True)

    xml_dir = os.path.join("/workspace/data", "xml/")
    os.makedirs(xml_dir, exist_ok=True)
    command = str("mv " + "Omnibook\\ xml/*.xml " + output_dir)
    subprocess.run(command, text=True, shell=True)

    command = str("rm -rf Omnibook\\ xml")
    subprocess.run(command, text=True, shell=True)

    command = str("rm -rf __MACOSX")
    subprocess.run(command, text=True, shell=True)

    print(" done.")


def extract_features(cfg: Config, save_dirpath: str) -> None:
    save_dir = Path(save_dirpath)
    feat_notenum_all = []
    feat_note_onehot_all = []
    feat_chord_chroma_all = []
    feat_mode_all = []

    xml_dir = os.path.join("/workspace/data", "xml/")
    for xml_file in track(glob.glob(xml_dir + "/*.xml")):
        feat = MusicXMLFeature(
            xml_file,
            cfg.feature.key_root,
            num_beats=cfg.feature.n_beats,
            num_parts_of_beat=cfg.feature.beat_reso,
            min_note_num=cfg.feature.notenum_from,
            max_note_num=cfg.feature.notenum_thru,
        )

        # TODO: Save mode sequence, major or minor
        mode_map = {"major": 0.0, "minor": 1.0}
        mode = feat.get_mode()

        seq_notenum = feat.get_seq_notenum()
        seq_note_onehot = feat.get_seq_note_onehot()
        seq_chord_chroma = feat.get_seq_chord_chorma()

        feat_notenum = make_sequence(seq_notenum, cfg.feature.max_seq_len)
        feat_note_onehot = make_sequence(
            seq_note_onehot, cfg.feature.max_seq_len
        )
        feat_chord_chroma = make_sequence(
            seq_chord_chroma, cfg.feature.max_seq_len
        )

        feat_notenum_all.append(feat_notenum)
        feat_note_onehot_all.append(feat_note_onehot)
        feat_chord_chroma_all.append(feat_chord_chroma)
        feat_mode_all.append(np.tile([mode_map[mode]], len(feat_notenum)))

    feat_notenum_all_np = np.vstack(feat_notenum_all)
    feat_note_onehot_all_np = np.vstack(feat_note_onehot_all)
    feat_chord_chroma_all_np = np.vstack(feat_chord_chroma_all)
    feat_mode_all_np = np.concatenate(feat_mode_all)

    np.save(save_dir / "notenum.npy", feat_notenum_all_np)
    np.save(save_dir / "note_onehot.npy", feat_note_onehot_all_np)
    np.save(save_dir / "chord_chroma.npy", feat_chord_chroma_all_np)
    np.save(save_dir / "mode.npy", feat_mode_all_np)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    # Download Omnibook MusicXML
    get_music_xml(cfg.benzaiten.xml_dir)

    # Extract features from MusicXM.
    extract_features(
        cfg, os.path.join(cfg.benzaiten.root_dir, cfg.benzaiten.feature_dir)
    )


if __name__ == "__main__":
    main()
