import torch


@torch.jit.script
def support_to_scalar_fused(
    logits: torch.Tensor, support_size: int, epsilon: float
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    support = torch.arange(
        -float(support_size),
        float(support_size + 1),
        1.0,
        dtype=torch.float32,
        device=logits.device,
    )
    expected_value = torch.sum(probs * support, dim=-1)
    sgn = torch.sign(expected_value)
    abs_x = expected_value.abs()
    inv = sgn * (
        (
            ((1.0 + 4.0 * epsilon * (abs_x + 1.0 + epsilon)).sqrt() - 1.0)
            / (2.0 * epsilon)
        ).pow(2.0)
        - 1.0
    )
    return inv


@torch.jit.script
def scalar_to_support_fused(
    scalar: torch.Tensor, support_size: int, epsilon: float
) -> torch.Tensor:
    safe_scalar = scalar.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    transformed = (
        torch.sign(safe_scalar) * ((safe_scalar.abs() + 1.0).sqrt() - 1.0)
        + epsilon * safe_scalar
    )
    clamped = transformed.reshape(-1).clamp(
        min=-float(support_size), max=float(support_size)
    )

    shifted = clamped + float(support_size)
    floor_val = shifted.floor()
    ceil_val = shifted.ceil()

    upper_prob = shifted - floor_val
    lower_prob = 1.0 - upper_prob

    lower_idx = floor_val.to(torch.int64)
    upper_idx = ceil_val.to(torch.int64)

    batch_size = shifted.size(0)
    support_probs = torch.zeros(
        (batch_size, 2 * support_size + 1), dtype=torch.float32, device=scalar.device
    )

    batch_indices = torch.arange(batch_size, dtype=torch.int64, device=scalar.device)

    support_probs.index_put_((batch_indices, lower_idx), lower_prob, accumulate=True)
    support_probs.index_put_((batch_indices, upper_idx), upper_prob, accumulate=True)

    return support_probs


import os

if os.path.exists("../tricked_ops.so"):
    torch.ops.load_library("../tricked_ops.so")
elif os.path.exists("tricked_ops.so"):
    torch.ops.load_library("tricked_ops.so")


class MuZeroMathOps(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.jit.export
    def support_to_scalar(
        self, logits: torch.Tensor, support_size: int, epsilon: float
    ) -> torch.Tensor:
        return support_to_scalar_fused(logits, support_size, epsilon)

    @torch.jit.export
    def scalar_to_support(
        self, scalar: torch.Tensor, support_size: int, epsilon: float
    ) -> torch.Tensor:
        return scalar_to_support_fused(scalar, support_size, epsilon)

    @torch.jit.export
    def extract_unrolled_features(
        self, boards: torch.Tensor, hist: torch.Tensor
    ) -> torch.Tensor:
        return torch.ops.tricked.extract_unrolled_features(boards, hist)


if __name__ == "__main__":
    import sys

    model = MuZeroMathOps()
    scripted = torch.jit.script(model)
    out_path = sys.argv[1] if len(sys.argv) > 1 else "math_kernels.pt"
    scripted.save(out_path)
    print(f"Exported dynamic fused math kernels to {out_path}")
