# import pytest
# import torch

# from src.model.cvae import CVAELoss, CVAEModel

# SMPL_DATA_SIZE = 16


# @pytest.fixture
# def onehot_note_seq() -> torch.Tensor:
#     num_notes = 49
#     seq_len = 64

#     note_seq = torch.rand((SMPL_DATA_SIZE, seq_len, num_notes))
#     return note_seq


# @pytest.fixture
# def chromatic_chord_seq() -> torch.Tensor:
#     chord_dim = 12
#     seq_len = 64

#     chord_seq = torch.rand((SMPL_DATA_SIZE, seq_len, chord_dim))
#     return chord_seq


# def test_cvae_model(
#     onehot_note_seq: torch.Tensor, chromatic_chord_seq: torch.Tensor
# ) -> None:
#     note_feat_dim = onehot_note_seq.shape[2]
#     chord_feat_dim = chromatic_chord_seq.shape[2]
#     model = CVAEModel(
#         input_dim=note_feat_dim,
#         condition_dim=chord_feat_dim,
#         hidden_dim=16,
#         latent_dim=16,
#     )
#     x_hat, mean, logvar = model(onehot_note_seq, chromatic_chord_seq)

#     label = onehot_note_seq.argmax(dim=2)

#     criterion = CVAELoss()
#     loss = criterion(label, x_hat, mean, logvar)
#     loss.backward()
