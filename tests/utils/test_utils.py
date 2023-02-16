import numpy as np

from src.utils import make_sequence


def test_make_sequence() -> None:
    max_seq_len = 64

    # NOTE: Test that the size of data is divisible by max_seq_len.
    a = np.random.randint(100, size=(max_seq_len * 10))
    seq = make_sequence(a, max_seq_len, drop_last=False)
    assert np.array_equal(seq[0], a[:max_seq_len])

    # NOTE: Test that the size of data is not divisible by max_seq_len.
    a = np.random.randint(100, size=(1234))
    seq_drop_last = make_sequence(a, max_seq_len, drop_last=True)
    assert np.array_equal(seq_drop_last[0], a[:max_seq_len])

    # NOTE: Test that the size of data is not divisible by max_seq_len
    # and is not drop_last.
    a = np.random.randint(100, size=(1234))
    seq_non_drop_last = make_sequence(a, max_seq_len, drop_last=False)
    assert np.array_equal(seq_non_drop_last[0], a[:max_seq_len])
