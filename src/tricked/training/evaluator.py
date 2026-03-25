import time
import os
import torch
import numpy as np
import wandb

from tricked.model.network import MuZeroNet

def greedy_play(model, device, diff, max_steps=1000):
    from tricked_engine import GameStateExt, extract_feature

    state = GameStateExt(pieces=None, board_state=0, current_score=0, difficulty=diff, pieces_left=0)
    history = [state.board, state.board]
    action_history = []
    
    step = 0
    while not state.terminal and step < max_steps:
        if state.pieces_left == 0:
            state.refill_tray()
            if state.terminal:
                break
                
        feat = extract_feature(state, history, action_history, diff)
        s = torch.tensor(feat, dtype=torch.float32).view(1, 20, 96).to(device)
        
        with torch.no_grad():
            _, _, pol, _ = model.initial_inference(s)
            pol = torch.softmax(pol, dim=-1).squeeze(0).cpu().numpy()
            
        valid_mask = np.zeros(288, dtype=bool)
        valid_actions = []
        for slot in range(3):
            if state.available[slot] != -1:
                piece_id = state.available[slot]
                for pos in range(96):
                    if state.apply_move(slot, pos) is not None:
                        action_idx = piece_id * 96 + pos
                        valid_mask[action_idx] = True
                        valid_actions.append((action_idx, slot, pos))
                        
        if len(valid_actions) == 0:
            break
            
        pol[~valid_mask] = 0.0
        
        if pol.sum() > 0:
            pol /= pol.sum()
            best_action_idx = int(np.argmax(pol))
        else:
            best_action_idx = valid_actions[0][0]
            
        slot, pos = 0, 0
        for a, s_idx, p_idx in valid_actions:
            if a == best_action_idx:
                slot, pos = s_idx, p_idx
                break
                
        state = state.apply_move(slot, pos)
        action_history.append(best_action_idx)
        history.append(state.board)
        if len(history) > 8:
            history.pop(0)
            
        step += 1
        
    return state.score

import ray

@ray.remote(num_cpus=1, num_gpus=0.1)
class EvaluatorActor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = MuZeroNet(d_model=cfg.d_model, num_blocks=cfg.num_blocks).to(self.device)
        self.model.eval()
        wandb.init(project="tricked-ai-eval", name="Continuous_Evaluator")

    def evaluate(self, weights):
        self.model.load_state_dict(weights)
        self.model.eval()
        
        scores = []
        for _ in range(50):
            scores.append(greedy_play(self.model, self.device, self.cfg.difficulty))
            
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        
        wandb.log({"eval/avg_score": avg_score, "eval/max_score": max_score})
        print(f"🏅 Evaluator - Avg: {avg_score:.1f}, Max: {max_score}")
        return avg_score
