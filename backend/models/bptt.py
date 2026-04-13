import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

from .muzero import MuZeroNet

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
        x = hidden_state.mean(dim=(2, 3))
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
        eps = self.epsilon
        sgn = torch.sign(scalar)
        abs_x = torch.abs(scalar)
        transformed = sgn * (torch.sqrt(abs_x + 1.0) - 1.0) + eps * scalar
        clamped = self.value_support_size * torch.tanh(transformed / self.value_support_size)
        
        shifted = clamped + self.value_support_size
        floor_val = torch.floor(shifted)
        upper_prob = shifted - floor_val
        lower_prob = 1.0 - upper_prob
        
        S = 2 * self.value_support_size + 1
        probs = torch.zeros(scalar.size(0), S, device=scalar.device)
        
        lower_idx = torch.clamp(floor_val.long(), 0, S - 1)
        upper_idx = torch.clamp(lower_idx + 1, 0, S - 1)
        
        probs.scatter_add_(1, lower_idx.unsqueeze(-1), lower_prob.unsqueeze(-1))
        probs.scatter_add_(1, upper_idx.unsqueeze(-1), upper_prob.unsqueeze(-1))
        return probs

    def forward(self, state_features, actions, piece_identifiers, value_prefixs, target_policies, target_values, unrolled_state_features_gpu, loss_masks, importance_weights):
        batch_size = state_features.size(0)
        unroll_steps = actions.size(1)

        # 1. Initial State Forward
        running_hidden = self.active_net.representation(state_features)
        initial_value_logits, initial_policy_logits, initial_hole_logits = self.active_net.prediction(running_hidden)
        
        initial_value_targets = self.scalar_to_support(target_values[:, 0])
        initial_value_loss = -(initial_value_targets * F.log_softmax(initial_value_logits, dim=-1)).sum(dim=-1)
        
        initial_policy_targets = target_policies[:, 0]
        initial_policy_loss = -(initial_policy_targets * F.log_softmax(initial_policy_logits, dim=-1)).sum(dim=-1)
        
        # Tracking variables for metrics
        cumulative_loss = initial_value_loss + initial_policy_loss
        cumulative_val_loss = initial_value_loss.clone()
        cumulative_pol_loss = initial_policy_loss.clone()
        cumulative_vp_loss = torch.zeros_like(initial_value_loss)
        cumulative_similarity = torch.zeros_like(initial_value_loss)
        
        # Entropies
        initial_action_space_entropy = -(initial_policy_targets * torch.log(initial_policy_targets)).sum(dim=-1)
        initial_policy_probs = torch.softmax(initial_policy_logits, dim=-1)
        initial_policy_entropy = -(initial_policy_probs * F.log_softmax(initial_policy_logits, dim=-1)).sum(dim=-1)
        
        # 2. Unroll Loop
        for k in range(unroll_steps):
            mask = loss_masks[:, k + 1]
            scale = 1.0 / unroll_steps
            
            action_k = actions[:, k]
            piece_k = piece_identifiers[:, k]
            
            # Correct mathematically robust scaling instead of static detachment
            scaled_hidden = running_hidden * 0.5 + running_hidden.detach() * (1.0 - 0.5)
            
            next_hidden, value_prefix_logits = self.active_net.dynamics(scaled_hidden, action_k, piece_k)
            running_hidden = next_hidden
            
            vp_targets = self.scalar_to_support(value_prefixs[:, k])
            vp_loss = -(vp_targets * F.log_softmax(value_prefix_logits, dim=-1)).sum(dim=-1) * mask
            cumulative_loss = cumulative_loss + vp_loss * scale
            cumulative_vp_loss = cumulative_vp_loss + vp_loss * scale
            
            val_logits, pol_logits, hole_logits = self.active_net.prediction(running_hidden)
            
            val_targets = self.scalar_to_support(target_values[:, k + 1])
            val_loss = -(val_targets * F.log_softmax(val_logits, dim=-1)).sum(dim=-1) * mask
            cumulative_loss = cumulative_loss + val_loss * scale
            cumulative_val_loss = cumulative_val_loss + val_loss * scale
            
            pol_targets = target_policies[:, k + 1] + 1e-8
            pol_loss = -(pol_targets * F.log_softmax(pol_logits, dim=-1)).sum(dim=-1) * mask
            cumulative_loss = cumulative_loss + pol_loss * scale
            cumulative_pol_loss = cumulative_pol_loss + pol_loss * scale
            
            # SimSiam Forward
            unroll_state_k = unrolled_state_features_gpu[:, k]
            with torch.no_grad():
                target_embedded = self.target_net.representation(unroll_state_k)
                target_projected = self.target_net.projector(target_embedded)
            
            active_projected = self.active_net.projector(running_hidden)
            similarity = -F.cosine_similarity(active_projected, target_projected, dim=-1) * mask
            cumulative_loss = cumulative_loss + similarity * scale
            cumulative_similarity = cumulative_similarity + similarity * scale

        final_loss = (cumulative_loss * importance_weights).mean()
        weighted_val_loss = (cumulative_val_loss * importance_weights).mean()
        weighted_pol_loss = (cumulative_pol_loss * importance_weights).mean()
        weighted_vp_loss = (cumulative_vp_loss * importance_weights).mean()
        weighted_sim = (cumulative_similarity * importance_weights).mean()
        
        return final_loss, initial_value_logits, weighted_val_loss, weighted_pol_loss, weighted_vp_loss, initial_policy_entropy.mean(), initial_action_space_entropy.mean(), weighted_sim

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
