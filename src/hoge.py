import glob
import os

import hydra
import music21
import numpy as np

from config import Config
from utils import (
    add_rest_nodes,
    chord_seq_to_chroma,
    divide_seq,
    make_note_and_chord_seq_from_musicxml,
    note_seq_to_onehot,
)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    data_all = []
    label_all = []

    xml_dir = os.path.join(cfg.benzaiten.root_dir, cfg.benzaiten.xml_dir)
    for xml_file in glob.glob(xml_dir + "/*.xml"):
        score = music21.converter.parse(xml_file)
        key = score.analyze("key")

        if key.mode == "major":  # type: ignore
            interval = music21.interval.Interval(
                key.tonic,  # type: ignore
                music21.pitch.Pitch(cfg.feature.key_root),
            )
            score = score.transpose(interval)

            # NOTE:
            # MusicXMLを読み込んだScoreからメロディのseqとchordのseqを取得
            note_seq, chord_seq = make_note_and_chord_seq_from_musicxml(
                score,
                cfg.feature.total_measures,
                cfg.feature.n_beats,
                cfg.feature.beat_reso,
            )

            # note seq を one-hot に変換
            onehot_seq = note_seq_to_onehot(
                note_seq,
                cfg.feature.notenum_thru,
                cfg.feature.notenum_from,
            )

            # 休符（one_hotがすべて0の部分）にflagを追加. onehot_seqの次元が1増える
            onehot_seq = add_rest_nodes(onehot_seq)

            # C, Cm7などのコードをmany_to_hotの形に変換
            chroma_seq = chord_seq_to_chroma(chord_seq)
            print(np.unique(chroma_seq.sum(axis=1)))

            # chaos
            # 学習に使う上限まで、4小節ずつデータを分割
            # unit_measures(=4)ごとを特徴量とし、その時のメロディをラベルデータにする
            # data_all, label_allに都度追加(この関数内で定義したものを内部で操作してるのはやめたい)
            divide_seq(cfg, onehot_seq, chroma_seq, data_all, label_all)
        break


if __name__ == "__main__":
    main()
