import torch
x = torch.randn(2, 20, 96)
gather_indices = torch.zeros(96, 4, dtype=torch.long)
gather_weights = torch.zeros(96, 4)

try:
    msg = x[:, :, gather_indices]
    print("msg shape:", msg.shape)
    msg = msg * gather_weights
    print("msg * weights shape:", msg.shape)
except Exception as e:
    print("ERROR:", e)
