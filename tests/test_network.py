import torch

from tricked.model.network import MuZeroNet

def test_network_initial_inference() -> None:
    model = MuZeroNet()
    state = torch.zeros(2, 20, 96)

    h, v, p, hole_logits = model.initial_inference(state)
    assert v.shape == (2, 1)
    assert p.shape == (2, 288)
    assert h.shape == (2, 128, 96)
    assert hole_logits.shape == (2, 96)

    assert torch.allclose(p.sum(dim=-1), torch.ones(2))

def test_network_recurrent_inference() -> None:
    model = MuZeroNet()
    h = torch.zeros(2, 128, 96)
    a = torch.randint(0, 288, (2,))
    piece_id = torch.randint(0, 12, (2,))

    h_next, r, v, p, hole_logits = model.recurrent_inference(h, a, piece_id)

    assert v.shape == (2, 1)
    assert r.shape == (2, 1)
    assert p.shape == (2, 288)
    assert h_next.shape == (2, 128, 96)
    assert hole_logits.shape == (2, 96)
