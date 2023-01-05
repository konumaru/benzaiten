import torch
import torch.nn as nn
import torch.nn.functional as F


def concate_inputs(
    note_on: torch.Tensor,
    note_off: torch.Tensor,
    time_shift: torch.Tensor,
    set_velocity: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        (
            F.one_hot(note_on),
            F.one_hot(note_off),
            F.one_hot(time_shift),
            F.one_hot(set_velocity),
        ),
        dim=2,
    ).to(torch.float32)


def main() -> None:
    # NOTE: 何もないときは、そもそもeventが発生しないのでダミートークンが必要ない
    # ただし、eventがないときの分は必要
    seq_len = 256
    note_on = torch.randint(0, 89, size=(1, seq_len))
    note_off = torch.randint(0, 89, size=(1, seq_len))
    time_shift = torch.randint(0, 101, size=(1, seq_len))
    set_velocity = torch.randint(0, 33, size=(1, seq_len))

    hidden_dim = 89 + 89 + 101 + 33

    inputs = concate_inputs(note_on, note_off, time_shift, set_velocity)

    model = nn.Transformer(hidden_dim, 2, 4, 4, 512, batch_first=True)
    z = model(inputs, inputs)
    print(z.shape)

    out = torch.sigmoid(z[:, -1, :]).argmax(dim=1)
    print(out)

    if out < 89:
        print("NOTE_ON:", out.item())
    elif out < (89 + 89):
        print("NOTE_OFF:", (out - 89).item())
    elif out < (89 + 89 + 101):
        print("TIME_SIFT:", (out - (89 + 89)).item())
    else:
        print("SET_VELOCITY:", (out - (89 + 89 + 101)).item())


if __name__ == "__main__":
    main()
