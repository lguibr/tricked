import torch
import torch.optim as optim
from tricked.model.network import MuZeroNet
from tricked.training.trainer import make_train_step


def test_core_flow_convergence():
    """
    Tests the core training loop to guarantee convergence.
    We inject identical synthetic batches repeatedly. If the architecture is sound,
    the loss MUST continuously decrease as the network minimizes against the static targets.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize networks
    model = MuZeroNet().to(device)
    ema_model = MuZeroNet().to(device)
    ema_model.load_state_dict(model.state_dict())

    # 2. Setup optimizer and mock config
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    cfg = {
        "train_batch_size": 4,
        "unroll_steps": 5,
        "device": device.type,
        "use_cuda_graphs": False,
    }

    # 3. Create single-step compiler
    train_step = make_train_step(model, ema_model, optimizer, cfg, cfg["unroll_steps"], device)

    # 4. Generate synthetic batch of identical static data
    b_states = torch.randn((4, 20, 96), device=device)
    b_acts = torch.randint(0, 288, (4, 5), device=device)
    b_pids = torch.randint(0, 12, (4, 5), device=device)
    b_rews = torch.randn((4, 5), device=device)
    b_t_pols = torch.softmax(torch.randn((4, 6, 288), device=device), dim=-1)
    b_t_vals = torch.randn((4, 6), device=device)
    b_m_vals = torch.randn((4, 6), device=device)
    b_t_states = torch.randn((4, 5, 20, 96), device=device)
    b_masks = torch.ones((4, 6), device=device)
    b_weights = torch.ones(4, device=device)

    # 5. Execute learning loop
    initial_loss = None
    final_loss = None

    print("\n[Convergence Test] Starting loss minimization loop...")
    for i in range(15):
        loss, v, p, r, td = train_step(
            b_states,
            b_acts,
            b_pids,
            b_rews,
            b_t_pols,
            b_t_vals,
            b_m_vals,
            b_t_states,
            b_masks,
            b_weights,
        )

        if i == 0:
            initial_loss = loss
            print(f"Step {i:02d} | Initial Loss: {loss:.4f} (V: {v:.4f}, P: {p:.4f}, R: {r:.4f})")
        elif i == 14:
            final_loss = loss
            print(f"Step {i:02d} | Final Loss:   {loss:.4f} (V: {v:.4f}, P: {p:.4f}, R: {r:.4f})")
        else:
            print(f"Step {i:02d} | Loss: {loss:.4f}")

    # 6. Assert Guaranteed Convergence
    assert initial_loss is not None and final_loss is not None
    assert final_loss < initial_loss, f"Convergence Failed! Final Loss ({final_loss:.4f}) >= Initial Loss ({initial_loss:.4f})"
    print(f"✅ Guaranteed Convergence: Loss dropped from {initial_loss:.4f} to {final_loss:.4f}")


if __name__ == "__main__":
    test_core_flow_convergence()
