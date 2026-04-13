import torch
import pytest
import os
import sys

# Ensure backend module is accessible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tricked.models.muzero import MuZeroNet

def test_model_dimensions_and_initialization():
    hidden_dim = 16
    context_size = 96  # from env/tricked NATIVE_FEATURE_CHANNELS
    
    net = MuZeroNet(
        hidden_dim=hidden_dim,
        num_blocks=1,
        support_size=10,
        spatial_channels=context_size
    )
    
    # Check nan free initialization
    for param in net.parameters():
        assert not torch.isnan(param).any(), "NaN found in neural network initialization!"
        
    # Check forward flow and dimensions
    B = 2
    dummy_obs = torch.zeros(B, context_size, 8, 16)
    
    # 1. Representation
    hidden = net.representation(dummy_obs)
    assert not torch.isnan(hidden).any()
    assert hidden.shape == (B, hidden_dim, 8, 8)
    
    # 2. Prediction
    v, p, hole = net.prediction(hidden)
    assert not torch.isnan(p).any()
    assert not torch.isnan(v).any()
    assert not torch.isnan(hole).any()
    # v is batch, value_support*2 + 1
    assert v.shape == (B, 10 * 2 + 1)
    assert p.shape == (B, 288)
    
    # 3. Dynamics
    actions = torch.zeros(B, dtype=torch.long)
    pieces = torch.zeros(B, dtype=torch.long)
    next_hidden, r = net.dynamics(hidden, actions, pieces)
    assert next_hidden.shape == (B, hidden_dim, 8, 8)
    assert r.shape == (B, 10 * 2 + 1)

def test_model_action_dimensions_robustness():
    # Test that shape [B, 1] for actions does not break the model, imitating rust inference shapes
    hidden_dim = 16
    context_size = 96
    
    net = MuZeroNet(
        hidden_dim=hidden_dim,
        num_blocks=1,
        support_size=10,
        spatial_channels=context_size
    )
    
    B = 2
    hidden = torch.zeros(B, hidden_dim, 8, 8)
    
    # Simulating what happens during rust ffi interaction when batched dimension gets unsqueezed to B, 1
    actions_2d = torch.zeros((B, 1), dtype=torch.long)
    pieces_2d = torch.zeros((B, 1), dtype=torch.long)
    
    next_hidden, r = net.dynamics(hidden, actions_2d, pieces_2d)
    assert next_hidden.shape == (B, hidden_dim, 8, 8)

from tricked.models.muzero import InitialInferenceModel, RecurrentInferenceModel

def test_model_torchscript_serialization():
    # Test that the wrappers can be compiled down to TorchScript graphs successfully
    # resolving previous indexing and dimension tracking bugs inside TorchScript parsing.
    hidden_dim = 16
    context_size = 96
    
    net = MuZeroNet(
        hidden_dim=hidden_dim,
        num_blocks=1,
        support_size=10,
        spatial_channels=context_size
    )
    
    initial = InitialInferenceModel(net).eval()
    recurrent = RecurrentInferenceModel(net).eval()
    
    # Ensure they can be scripted without type casting errors
    try:
        scripted_initial = torch.jit.script(initial)
        scripted_recurrent = torch.jit.script(recurrent)
    except Exception as e:
        pytest.fail(f"TorchScript compilation failed: {e}")
    
    # Let's ensure the scripted graphs still run correctly
    B = 1
    dummy_obs = torch.zeros(B, context_size, 8, 16)
    scripted_hidden, scripted_val, scripted_pol, scripted_hole = scripted_initial(dummy_obs)
    
    assert scripted_hidden.shape == (B, hidden_dim, 8, 8)
