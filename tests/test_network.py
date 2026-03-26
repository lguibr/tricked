import torch

from tricked.model.network import MuZeroNet


def test_network_initial_inference() -> None:
    model = MuZeroNet()
    state = torch.zeros(2, 20, 96)

    h, v, p, hole_logits = model.initial_inference(state)
    assert v.shape == (2,)
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

    assert v.shape == (2,)
    assert r.shape == (2,)
    assert p.shape == (2, 288)
    assert h_next.shape == (2, 128, 96)
    assert hole_logits.shape == (2, 96)


def test_scalar_support_bidirectionality() -> None:
    model = MuZeroNet(support_size=200)

    # Test extreme values and typical values
    scalars = torch.tensor([-300.0, -200.0, -10.5, 0.0, 1.0, 15.3, 200.0, 300.0])

    # scalar -> support
    support_probs = model.scalar_to_support(scalars)
    assert support_probs.shape == (8, 401)

    # support -> scalar
    # support_to_scalar expects logits. Since scalar_to_support output probabilities,
    # we take log to mimic logits (adding a small epsilon to avoid -inf).
    # Since support_to_scalar internally applies softmax, treating log(probs + eps) as logits recovers probs
    logits = torch.log(support_probs + 1e-8)
    recovered_scalars = model.support_to_scalar(logits)

    # Check if they are close. Note that MuZero scalar transform compresses large values,
    # so we clamp the expected scalars to the support size limits just in case.
    expected = (
        scalars.sign() * (torch.sqrt(torch.abs(scalars) + 1.0) - 1.0) + model.epsilon * scalars
    )
    expected = expected.clamp(-model.support_size, model.support_size)
    expected_recovered = torch.sign(expected) * (
        (
            (
                (
                    -1.0
                    + torch.sqrt(1.0 + 4.0 * model.epsilon * (1.0 + model.epsilon + expected.abs()))
                )
                / (2.0 * model.epsilon)
            )
            ** 2
        )
        - 1.0
    )

    # The transform mathematically compresses everything to fit in the support sizes,
    # ensuring the inverted value matches the clamped theoretical behavior.
    assert torch.allclose(recovered_scalars, expected_recovered, atol=1e-1)
