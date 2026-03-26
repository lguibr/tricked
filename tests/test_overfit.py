import torch
import torch.optim as optim

from tricked.model.network import MuZeroNet


def test_overfit_single_batch():
    torch.manual_seed(42)
    device = torch.device("cpu")

    model = MuZeroNet(d_model=32, num_blocks=2).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Synthetic batch: 1 element, 5 unroll steps
    batch_size = 2
    unroll_steps = 5

    states = torch.randn(batch_size, 20, 96, device=device)
    acts = torch.randint(0, 288, (batch_size, unroll_steps), device=device)
    pids = torch.randint(0, 10, (batch_size, unroll_steps), device=device)
    rews = torch.randn(batch_size, unroll_steps, device=device)
    t_pols = torch.softmax(torch.randn(batch_size, unroll_steps + 1, 288, device=device), dim=-1)
    t_vals = torch.randn(batch_size, unroll_steps + 1, device=device)
    t_states = torch.randn(batch_size, unroll_steps, 20, 96, device=device)
    masks = torch.ones(batch_size, unroll_steps + 1, device=device)
    weights = torch.ones(batch_size, device=device)

    initial_loss = None
    final_loss = None

    model.train()
    for i in range(10):
        optimizer.zero_grad()

        h = model.representation(states)
        v_l, p_p, h_l = model.prediction(h)

        import torch.nn.functional as F

        v_loss_0 = -(model.scalar_to_support(t_vals[:, 0]) * F.log_softmax(v_l, dim=-1)).sum(-1)
        p_loss_0 = -torch.sum(t_pols[:, 0] * torch.log(p_p + 1e-8), dim=-1)

        loss = v_loss_0 + p_loss_0
        loss = (loss * weights).mean()

        if i == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    assert (
        final_loss < initial_loss
    ), "Model failed to overfit / reduce loss on a single synthetic batch!"


def test_overfit_complex_batch_and_hooks():
    torch.manual_seed(42)
    device = torch.device("cpu")

    model = MuZeroNet(d_model=32, num_blocks=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    batch_size = 2
    unroll_steps = 3

    states = torch.randn(batch_size, 20, 96, device=device)
    acts = torch.randint(0, 288, (batch_size, unroll_steps), device=device)
    pids = torch.randint(0, 10, (batch_size, unroll_steps), device=device)

    initial_loss = None
    final_loss = None

    for i in range(5):
        optimizer.zero_grad()
        h = model.representation(states)
        scale_hook_called = []
        h.register_hook(
            lambda grad, h_called=scale_hook_called: h_called.append(True) or grad * 0.5
        )

        v_l, p_p, h_l = model.prediction(h)
        loss = v_l.sum() + p_p.sum()

        unrolled_h_called = []
        for k in range(unroll_steps):
            h, r = model.dynamics(h, acts[:, k], pids[:, k])
            h.register_hook(
                lambda grad, uh_called=unrolled_h_called: uh_called.append(True) or grad * 0.5
            )
            loss += h.sum() + r.sum()

        if i == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()
        final_loss = loss.item()

        assert len(scale_hook_called) > 0, "Gradient scale hook for representation h did not fire!"
        assert (
            len(unrolled_h_called) == unroll_steps
        ), "Gradient scale hook for recurrent dynamics h did not fire!"

    assert final_loss != initial_loss, "Model failed to train on a complex batch!"


def test_ema_weights_update():
    torch.manual_seed(42)
    model = MuZeroNet(d_model=32, num_blocks=1)
    ema_model = MuZeroNet(d_model=32, num_blocks=1)

    # Initialize differently
    for p, p_ema in zip(model.parameters(), ema_model.parameters()):
        p.data.fill_(1.0)
        p_ema.data.fill_(0.0)

    # Apply EMA step
    alpha = 0.99
    with torch.no_grad():
        for p, p_ema in zip(model.parameters(), ema_model.parameters()):
            p_ema.data.copy_(alpha * p_ema.data + (1.0 - alpha) * p.data)

    # Assert EMA drifted towards model parameters
    for p_ema in ema_model.parameters():
        assert torch.allclose(
            p_ema.data, torch.full_like(p_ema.data, 0.01)
        ), "EMA update incorrect!"
