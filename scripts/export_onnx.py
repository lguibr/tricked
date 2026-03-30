import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

HEXAGONAL_TO_CARTESIAN_MAP_ARRAY = [
    (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12),
    (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13),
    (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14),
    (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15),
    (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15),
    (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (5, 12), (5, 13), (5, 14),
    (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (6, 12), (6, 13),
    (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12)
]

def get_valid_spatial_mask_8x8(device="cpu"):
    mask = torch.zeros((1, 1, 8, 8), dtype=torch.float32, device=device)
    for r, c in HEXAGONAL_TO_CARTESIAN_MAP_ARRAY:
        mask[0, 0, r, c // 2] = 1.0
    return mask

class FlattenedResNetBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.register_buffer('spatial_mask', get_valid_spatial_mask_8x8())

    def forward(self, x):
        res = x
        out = self.conv1(x) * self.spatial_mask
        out = out.permute(0, 2, 3, 1).contiguous()
        out = F.mish(self.norm1(out))
        out = out.permute(0, 3, 1, 2).contiguous() * self.spatial_mask
        
        out = self.conv2(out) * self.spatial_mask
        out = out.permute(0, 2, 3, 1).contiguous()
        out = self.norm2(out)
        out = out.permute(0, 3, 1, 2).contiguous() * self.spatial_mask
        
        return F.mish(res + out) * self.spatial_mask

class RepresentationNet(nn.Module):
    def __init__(self, hidden_dim, num_blocks):
        super().__init__()
        self.proj_in = nn.Conv2d(40, hidden_dim, 3, padding=1)
        self.blocks = nn.Sequential(*[FlattenedResNetBlock(hidden_dim) for _ in range(num_blocks)])
        self.scale_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        B = x.shape[0]
        x_reshaped = x.view(B, 20, 8, 8, 2).permute(0, 1, 4, 2, 3).reshape(B, 40, 8, 8)
        h = self.proj_in(x_reshaped)
        h = self.blocks(h)
        h = h.permute(0, 2, 3, 1).contiguous()
        h = self.scale_norm(h)
        h = h.permute(0, 3, 1, 2).contiguous()
        return h

class DynamicsNet(nn.Module):
    def __init__(self, hidden_dim, num_blocks, support_size):
        super().__init__()
        self.piece_emb = nn.Embedding(48, hidden_dim)
        self.pos_emb = nn.Embedding(96, hidden_dim)
        self.proj_in = nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1)
        self.blocks = nn.Sequential(*[FlattenedResNetBlock(hidden_dim) for _ in range(num_blocks)])
        self.scale_norm = nn.LayerNorm(hidden_dim)
        
        self.reward_cond = nn.Conv2d(hidden_dim * 2, hidden_dim, 1)
        self.reward_fc1 = nn.Linear(hidden_dim, 64)
        self.reward_norm = nn.LayerNorm(64)
        self.reward_fc2 = nn.Linear(64, 2 * support_size + 1)
        
    def forward(self, hidden_state, batched_action, batched_piece_identifier):
        pos_indices = batched_action % 96
        action_embeddings = self.piece_emb(batched_piece_identifier) + self.pos_emb(pos_indices)
        action_expanded = action_embeddings.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 8, 8)
        concatenated_features = torch.cat([hidden_state, action_expanded], dim=1)
        
        reward_convolutions_mish = F.mish(self.reward_cond(concatenated_features))
        hidden_state_avg = reward_convolutions_mish.mean(dim=(2, 3))
        
        reward_features = F.mish(self.reward_norm(self.reward_fc1(hidden_state_avg)))
        reward_logits = self.reward_fc2(reward_features)
        
        hidden_state_next = self.proj_in(concatenated_features)
        hidden_state_next = self.blocks(hidden_state_next)
        hidden_state_next = hidden_state_next.permute(0, 2, 3, 1).contiguous()
        hidden_state_next = self.scale_norm(hidden_state_next)
        hidden_state_next = hidden_state_next.permute(0, 3, 1, 2).contiguous()
        
        return hidden_state_next, reward_logits

class HolePredictor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # In rust: 0 and 2
        setattr(self, '0', nn.Linear(hidden_dim, 64))
        setattr(self, '2', nn.Linear(64, 2))
        
    def forward(self, x):
        l1 = getattr(self, '0')
        l2 = getattr(self, '2')
        return l2(F.mish(l1(x)))

class PredictionNet(nn.Module):
    def __init__(self, hidden_dim, support_size, action_count):
        super().__init__()
        self.val_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        self.val_norm = nn.LayerNorm(hidden_dim // 2)
        self.value_fc1 = nn.Linear(hidden_dim // 2, 64)
        self.value_fc2 = nn.Linear(64, 2 * support_size + 1)
        
        self.pol_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        self.pol_norm = nn.LayerNorm(hidden_dim // 2)
        self.policy_fc1 = nn.Linear(hidden_dim // 2, action_count)
        
        self.hole_predictor = HolePredictor(hidden_dim)
        
    def forward(self, hidden_state):
        transposed = hidden_state.permute(0, 2, 3, 1)
        
        val_feat = F.mish(self.val_norm(self.val_proj(transposed))).mean(dim=(1, 2))
        val_inter = F.mish(self.value_fc1(val_feat))
        val_logits = self.value_fc2(val_inter)
        
        pol_feat = F.mish(self.pol_norm(self.pol_proj(transposed))).mean(dim=(1, 2))
        pol_logits = self.policy_fc1(pol_feat)
        
        hole_logits = self.hole_predictor(transposed).flatten(1, 3)
        return val_logits, pol_logits, hole_logits

class InitialInferenceModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, batched_state):
        hidden_state = self.model.representation(batched_state)
        val_logits, pol_logits, hole_logits = self.model.prediction(hidden_state)
        pol_probs = F.softmax(pol_logits, dim=-1)
        return hidden_state, val_logits, pol_probs, hole_logits

class RecurrentInferenceModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, hidden_state, batched_action, batched_piece_id):
        hidden_state_next, reward_logits = self.model.dynamics(hidden_state, batched_action, batched_piece_id)
        val_logits, pol_logits, hole_logits = self.model.prediction(hidden_state_next)
        pol_probs = F.softmax(pol_logits, dim=-1)
        return hidden_state_next, reward_logits, val_logits, pol_probs, hole_logits

class MuZeroNet(nn.Module):
    def __init__(self, hidden_dim=256, num_blocks=4, support_size=300):
        super().__init__()
        self.representation = RepresentationNet(hidden_dim, num_blocks)
        self.dynamics = DynamicsNet(hidden_dim, num_blocks, support_size)
        self.prediction = PredictionNet(hidden_dim, support_size, 288)

def export_onnx(model_path=None):
    import os
    model = MuZeroNet()
    out_dir = ""
    if model_path:
        out_dir = os.path.dirname(model_path)
        if out_dir: out_dir += "/"
        print(f"Loading {model_path} dict into Python architecture...")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        
        # Map the state_dict appropriately
        model.load_state_dict(state_dict, strict=False)
    else:
        print("No model path provided. Generating ONNX with randomly initialized dummy weights...")
    
    model.eval()
    
    print("Exporting initial_inference to ONNX...")
    initial_model = InitialInferenceModel(model).eval()
    dummy_initial_in = torch.randn(1, 20, 8, 16)
    torch.onnx.export(
        initial_model,
        dummy_initial_in,
        f"{out_dir}initial_inference.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['batched_state'],
        output_names=['hidden_state', 'value_logits', 'policy_probs', 'hole_logits'],
        dynamic_axes={'batched_state': {0: 'batch_size'},
                      'hidden_state': {0: 'batch_size'},
                      'value_logits': {0: 'batch_size'},
                      'policy_probs': {0: 'batch_size'},
                      'hole_logits': {0: 'batch_size'}}
    )

    print("Exporting recurrent_inference to ONNX...")
    recurrent_model = RecurrentInferenceModel(model).eval()
    dummy_hidden = torch.randn(1, 256, 8, 8)
    dummy_action = torch.randint(0, 96, (1,), dtype=torch.int64)
    dummy_piece = torch.randint(0, 48, (1,), dtype=torch.int64)
    torch.onnx.export(
        recurrent_model,
        (dummy_hidden, dummy_action, dummy_piece),
        f"{out_dir}recurrent_inference.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['hidden_state', 'action', 'piece'],
        output_names=['hidden_state_next', 'reward_logits', 'value_logits', 'policy_probs', 'hole_logits'],
        dynamic_axes={'hidden_state': {0: 'batch_size'},
                      'action': {0: 'batch_size'},
                      'piece': {0: 'batch_size'},
                      'hidden_state_next': {0: 'batch_size'},
                      'reward_logits': {0: 'batch_size'},
                      'value_logits': {0: 'batch_size'},
                      'policy_probs': {0: 'batch_size'},
                      'hole_logits': {0: 'batch_size'}}
    )
    print("Export complete!")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    export_onnx(path)
