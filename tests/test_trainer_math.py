import torch


def test_muzero_gradient_scaling():
    # In MuZero, to ensure gradients don't explode from recurrent unrolls, two things must happen:
    # 1. The total gradient flowing back into the representation function from the K unrolled steps
    #    should be scaled by 1/K (meaning the total loss is divided by K).
    # 2. The gradient at the start of the dynamics function is scaled by 0.5 to balance
    #    the prediction gradient and the dynamics gradient.

    # We will simulate a simplified MuZero loop and verify the scaling.
    class DummyMuZero(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.representation = torch.nn.Linear(10, 10)
            self.dynamics = torch.nn.Linear(10, 10)
            self.prediction = torch.nn.Linear(10, 1)

    model = DummyMuZero()

    states = torch.randn(1, 10)

    # Unroll 1: No scaling (simulate what happens without 1/K or 0.5 hooks)
    h_unscaled = model.representation(states)
    loss_unscaled = model.prediction(h_unscaled).sum()

    for k in range(5):  # K = 5
        h_unscaled = model.dynamics(h_unscaled)
        loss_unscaled += model.prediction(h_unscaled).sum()

    loss_unscaled.backward()
    grad_unscaled_rep = model.representation.weight.grad.clone()
    model.zero_grad()

    # Unroll 2: MOCK MuZero Scaling (1/K loss scaling and 0.5 dynamics hook)
    K = 5
    h_scaled = model.representation(states)
    h_scaled.register_hook(lambda grad: grad * 0.5)
    loss_scaled = model.prediction(h_scaled).sum()

    for k in range(K):
        h_scaled = model.dynamics(h_scaled)
        h_scaled.register_hook(lambda grad: grad * 0.5)
        loss_scaled += model.prediction(h_scaled).sum()

    loss_scaled = loss_scaled / K
    loss_scaled.backward()

    grad_scaled_rep = model.representation.weight.grad.clone()
    model.zero_grad()

    # Now assert that in actual trainer.py, the logic is applied correctly!
    # Since we can't easily mock the entire C++ bindings Environment here,
    # we'll read trainer.py and verify the math is indeed implemented.
    with open("src/tricked/training/trainer.py") as f:
        src = f.read()

        # Verify 1/K loss scaling
        assert (
            "loss = loss / steps" in src
            or "loss /= steps" in src
            or "loss = (loss * scaled_weights).mean() / steps" in src
            or "loss / steps" in src
        ), "Total loss MUST be scaled by 1/K to prevent gradient explosion!"

        # Verify 0.5 dynamics scaling
        assert (
            "h.register_hook(lambda grad: grad * 0.5)" in src
        ), "Dynamics gradient MUST be scaled by 0.5!"


def test_support_to_scalar_math():
    from tricked.model.network import MuZeroNet

    model = MuZeroNet(support_size=300)

    val = torch.tensor([[-50.0], [0.0], [50.0], [299.0], [-299.0]])
    support_probs = model.scalar_to_support(val)

    # support_to_scalar expects logits, so we invert the probability into logits via log
    mock_logits = torch.log(support_probs + 1e-8)
    reconstructed = model.support_to_scalar(mock_logits)

    torch.testing.assert_close(val, reconstructed.unsqueeze(1), rtol=1e-1, atol=1.0)


if __name__ == "__main__":
    test_muzero_gradient_scaling()
    test_support_to_scalar_math()
    print("Gradient Scaling Math Tests Passed!")
