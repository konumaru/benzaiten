import numpy as np

from src.preprocess import MusicXMLFeature


def test_music_xml_feature() -> None:
    smpl_xml_filepath = "/workspace/data/xml/Anthropology.xml"
    feat = MusicXMLFeature(xml_file=smpl_xml_filepath)

    seq_notenum = feat.get_seq_notenum()
    seq_note_onehot = feat.get_seq_note_onehot()
    seq_chord_chroma = feat.get_seq_chord_chorma()

    assert len(seq_note_onehot) == len(seq_note_onehot)
    assert len(seq_note_onehot) == len(seq_chord_chroma)
    assert np.array_equal(seq_notenum, seq_note_onehot.argmax(axis=1))
