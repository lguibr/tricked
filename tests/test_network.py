import torch

from tricked.model.network import MuZeroNet


def test_network_initial_inference() -> None:
    model = MuZeroNet(d_model=64, num_blocks=2)
    # Batch size 2, 9 channels, 96 triangles
    dummy_input = torch.zeros(2, 9, 96)

    hidden_state, val, policy_prob = model.initial_inference(dummy_input)
    assert val.shape == (2, 1)
    assert policy_prob.shape == (2, 288)
    assert hidden_state.shape == (2, 64, 8, 15)

    # Verify policy probs sum to 1 over the last dimension
    assert torch.allclose(policy_prob.sum(dim=-1), torch.ones(2))


def test_network_recurrent_inference() -> None:
    model = MuZeroNet(d_model=64, num_blocks=2)
    dummy_hidden = torch.zeros(2, 64, 8, 15)
    dummy_action = torch.zeros(2, dtype=torch.long)

    next_hidden, reward, val, policy_prob = model.recurrent_inference(dummy_hidden, dummy_action)

    assert val.shape == (2, 1)
    assert reward.shape == (2, 1)
    assert policy_prob.shape == (2, 288)
    assert next_hidden.shape == (2, 64, 8, 15)
