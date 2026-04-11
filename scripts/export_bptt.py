import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

from export_onnx import MuZeroNet

# Create a JIT-compatible version of HolePredictor
class JitHolePredictor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.layers = nn.ModuleDict({
            '0': nn.Linear(hidden_dim, 64),
            '2': nn.Linear(64, 2)
        })

    def forward(self, x):
        l1 = self.layers['0']
        l2 = self.layers['2']
        return l2(F.mish(l1(x)))

# Projector for SimSiam
class ProjectorNet(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.layers = nn.ModuleDict({
            '0': nn.Linear(hidden_dim, 512),
            '1': nn.LayerNorm(512),
            '3': nn.Linear(512, 128)
        })
        
    def forward(self, hidden_state):
        x = hidden_state.flatten(1)
        x = self.layers['0'](x)
        x = self.layers['1'](x)
        x = F.mish(x)
        x = self.layers['3'](x)
        return x

class BPTTKernel(nn.Module):
    def __init__(self, hidden_dim=256, num_blocks=4, support_size=300, spatial_channels=20):
        super().__init__()
        self.active_net = MuZeroNet(hidden_dim, num_blocks, support_size, spatial_channels)
        self.active_net.prediction.hole_predictor = JitHolePredictor(hidden_dim)
        self.active_net.projector = ProjectorNet(hidden_dim)
        
        self.target_net = MuZeroNet(hidden_dim, num_blocks, support_size, spatial_channels)
        self.target_net.prediction.hole_predictor = JitHolePredictor(hidden_dim)
        self.target_net.projector = ProjectorNet(hidden_dim)
        
        # We need these manually inside TorchScript
        self.value_support_size = support_size
        self.epsilon = 0.001

    def support_to_scalar(self, logits):
        probs = torch.softmax(logits, dim=-1)
        support = torch.arange(2 * self.value_support_size + 1, dtype=torch.float32, device=logits.device) - self.value_support_size
        expected_value = (probs * support).sum(dim=-1)
        sgn = torch.sign(expected_value)
        abs_x = torch.abs(expected_value)
        eps = self.epsilon
        term1 = torch.sqrt((abs_x + (1.0 + eps)) * (4.0 * eps) + 1.0) - 1.0
        term2 = term1 / (2.0 * eps)
        inv = sgn * (torch.pow(term2, 2.0) - 1.0)
        return inv

    def scalar_to_support(self, scalar):
        B = scalar.size(0)
        out = torch.zeros((B, 2 * self.value_support_size + 1), dtype=torch.float32, device=scalar.device)
        safe_scalar = torch.nan_to_num(scalar, 0.0, 0.0, 0.0)
        transformed = torch.sign(safe_scalar) * (torch.sqrt(torch.abs(safe_scalar) + 1.0) - 1.0) + safe_scalar * self.epsilon
        clamped = torch.clamp(transformed.view(-1), -self.value_support_size, self.value_support_size)
        shifted = clamped + self.value_support_size
        
        floor_val = torch.floor(shifted)
        ceil_val = torch.ceil(shifted)
        
        upper_prob = shifted - floor_val
        lower_prob = 1.0 - upper_prob
        
        lower_idx = floor_val.to(torch.int64)
        upper_idx = ceil_val.to(torch.int64)
        
        batch_indices = torch.arange(B, dtype=torch.int64, device=scalar.device)
        
        out.index_put_((batch_indices, lower_idx), lower_prob, accumulate=True)
        out.index_put_((batch_indices, upper_idx), upper_prob, accumulate=True)
        return out

    def forward(self, state_features, actions, piece_identifiers, value_prefixs, target_policies, target_values, unrolled_state_features_gpu, loss_masks, importance_weights):
        batch_size = state_features.size(0)
        unroll_steps = actions.size(1)

        # 1. Initial State Forward
        running_hidden = self.active_net.representation(state_features)
        initial_value_logits, initial_policy_logits, initial_hole_logits = self.active_net.prediction(running_hidden)
        
        initial_value_targets = self.scalar_to_support(target_values[:, 0])
        initial_value_loss = -(initial_value_targets * F.log_softmax(initial_value_logits, dim=-1)).sum(dim=-1)
        
        initial_policy_targets = target_policies[:, 0] + 1e-8
        initial_policy_loss = -(initial_policy_targets * F.log_softmax(initial_policy_logits, dim=-1)).sum(dim=-1)
        
        cumulative_loss = initial_value_loss + initial_policy_loss
        
        # 2. Unroll Loop
        for k in range(unroll_steps):
            mask = loss_masks[:, k + 1]
            scale = 1.0 / unroll_steps
            
            action_k = actions[:, k]
            piece_k = piece_identifiers[:, k]
            
            # Gradient scaling trick x0.5 for dynamics input (native TorchScript limitation, so we just .detach() + 0.5 diff?)
            scaled_hidden = running_hidden * 0.5 + running_hidden.detach() * 0.5
            
            next_hidden, value_prefix_logits = self.active_net.dynamics(scaled_hidden, action_k, piece_k)
            running_hidden = next_hidden
            
            vp_targets = self.scalar_to_support(value_prefixs[:, k])
            vp_loss = -(vp_targets * F.log_softmax(value_prefix_logits, dim=-1)).sum(dim=-1) * mask
            cumulative_loss = cumulative_loss + vp_loss * scale
            
            val_logits, pol_logits, hole_logits = self.active_net.prediction(running_hidden)
            
            val_targets = self.scalar_to_support(target_values[:, k + 1])
            val_loss = -(val_targets * F.log_softmax(val_logits, dim=-1)).sum(dim=-1) * mask
            cumulative_loss = cumulative_loss + val_loss * scale
            
            pol_targets = target_policies[:, k + 1] + 1e-8
            pol_loss = -(pol_targets * F.log_softmax(pol_logits, dim=-1)).sum(dim=-1) * mask
            cumulative_loss = cumulative_loss + pol_loss * scale
            
            # SimSiam Forward
            unroll_state_k = unrolled_state_features_gpu[:, k]
            with torch.no_grad():
                target_embedded = self.target_net.representation(unroll_state_k)
                target_projected = self.target_net.projector(target_embedded)
            
            active_projected = self.active_net.projector(running_hidden)
            similarity = -F.cosine_similarity(active_projected, target_projected, dim=-1) * mask
            cumulative_loss = cumulative_loss + similarity * scale

        final_loss = (cumulative_loss * importance_weights).mean()
        return final_loss, initial_value_logits

if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--blocks', type=int, default=10, help='Number of resnet blocks')
    parser.add_argument('--channels', type=int, default=256, help='Resnet hidden dimension')
    parser.add_argument('--support', type=int, default=300, help='Value support size')
    parser.add_argument('--spatial-channels', type=int, default=20, help='Spatial Channel Count')
    parser.add_argument('--output', type=str, default='bptt_kernel.pt', help='Output path')
    args = parser.parse_args()

    print(f"🚀 Compiling BPTT Kernel (Blocks: {args.blocks}, Channels: {args.channels}, Support: {args.support}, Spatial: {args.spatial_channels}) into fused TorchScript...")
    
    try:
        model = BPTTKernel(hidden_dim=args.channels, num_blocks=args.blocks, support_size=args.support, spatial_channels=args.spatial_channels)
        scripted_model = torch.jit.script(model)
        scripted_model.save(args.output)
        print(f"✅ Successfully exported to {args.output}")
    except Exception as e:
        print("❌ Scripting failed:", str(e))
