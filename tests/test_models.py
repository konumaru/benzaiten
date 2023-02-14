import pytest
import torch

from src.model import EmbeddedLstmVAE, OnehotLstmVAE

SMPL_DATA_SIZE = 16


@pytest.fixture
def seq_notenum() -> torch.Tensor:
    seq_len = 64
    notes = torch.randint(20, (SMPL_DATA_SIZE, seq_len))
    return notes


@pytest.fixture
def seq_note_onehot() -> torch.Tensor:
    num_notes = 49
    seq_len = 64

    note_seq = torch.rand((SMPL_DATA_SIZE, seq_len, num_notes))
    return note_seq


@pytest.fixture
def seq_chord_chroma() -> torch.Tensor:
    chord_dim = 12
    seq_len = 64

    chord_seq = torch.rand((SMPL_DATA_SIZE, seq_len, chord_dim))
    return chord_seq


def test_onehot_model(
    seq_note_onehot: torch.Tensor, seq_chord_chroma: torch.Tensor
) -> None:
    note_feat_dim = seq_note_onehot.shape[2]
    chord_feat_dim = seq_chord_chroma.shape[2]
    label = seq_note_onehot.argmax(dim=1)

    model = OnehotLstmVAE(
        input_dim=note_feat_dim,
        hidden_dim=16,
        latent_dim=16,
        condition_dim=chord_feat_dim,
        num_lstm_layers=1,
        num_fc_layers=1,
        bidirectional=True,
    )
    x_hat, mean, logvar = model(seq_note_onehot, seq_chord_chroma)

    loss = model.criterion(label, x_hat, mean, logvar)
    loss.backward()


def test_embedded_model(
    seq_notenum: torch.Tensor, seq_chord_chroma: torch.Tensor
) -> None:
    model = EmbeddedLstmVAE(
        input_dim=64,
        embedding_dim=16,
        hidden_dim=16,
        latent_dim=16,
        condition_dim=12,
        num_lstm_layers=1,
        num_fc_layers=1,
        bidirectional=False,
    )
    x_hat, mean, logvar = model(seq_notenum, seq_chord_chroma)

    loss = model.criterion(seq_notenum, x_hat, mean, logvar)
    loss.backward()
