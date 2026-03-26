import torch

class Mod(torch.nn.Module):
    def forward(self, x):
        with torch.autocast(device_type="cuda", enabled=False):
            return x + 1

try:
    m = torch.jit.script(Mod())
    print("success")
except Exception as e:
    print(e)
