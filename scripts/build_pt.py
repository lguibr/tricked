import torch
import json
import os

if not os.path.exists("../tricked_ops.so"):
    raise RuntimeError("Could not find compiled ../tricked_ops.so")
torch.ops.load_library("../tricked_ops.so")


class FeatureExtractor(torch.nn.Module):
    def __init__(self, canonical, compact, standard):
        super().__init__()
        self.register_buffer("canonical", torch.tensor(canonical, dtype=torch.int32))
        self.register_buffer("compact", torch.tensor(compact, dtype=torch.int64))
        self.register_buffer("standard", torch.tensor(standard, dtype=torch.int64))

    @torch.jit.export
    def forward(
        self,
        boards: torch.Tensor,
        avail: torch.Tensor,
        hist: torch.Tensor,
        acts: torch.Tensor,
        diff: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.tricked.extract_feature(
            boards, avail, hist, acts, diff, self.canonical, self.compact, self.standard
        )


def main():
    with open("masks.json") as f:
        data = json.load(f)

    padded_canonical = []
    for mask in data["canonical"]:
        padded = mask + [-1] * (128 - len(mask))
        padded_canonical.append(padded)

    padded_compact = []
    for mask_list in data["compact"]:
        padded = mask_list + [[0, 0]] * (64 - len(mask_list))
        padded_compact.append(padded)

    standard = data["standard"]

    extractor = FeatureExtractor(padded_canonical, padded_compact, standard)
    scripted = torch.jit.script(extractor)
    scripted.save("../feature_extractor.pt")
    print("Exported feature_extractor.pt")


if __name__ == "__main__":
    main()
