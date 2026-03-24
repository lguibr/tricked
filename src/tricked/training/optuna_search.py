from typing import Any

import numpy as np
import optuna
import torch
import torch.optim as optim
from wandb.integration.optuna import WeightsAndBiasesCallback

from tricked.config import get_hardware_config
from tricked.model.network import MuZeroNet
from tricked.training.buffer import ReplayBuffer
from tricked.training.self_play import self_play
from tricked.training.trainer import train


def objective(trial: Any) -> float:
    torch.set_num_threads(1)
    
    cfg = get_hardware_config()
    device = torch.device(cfg["device"])
    
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    simulations = trial.suggest_int("simulations", 30, 200)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    temp_decay = trial.suggest_int("temp_decay_steps", 10, 50)
    
    cfg["d_model"] = d_model
    cfg["simulations"] = simulations
    cfg["temp_decay_steps"] = temp_decay
    cfg["num_games"] = 120
    cfg["train_epochs"] = 1
    
    model = MuZeroNet(d_model=d_model, num_blocks=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    buffer = ReplayBuffer(capacity=cfg["capacity"], unroll_steps=cfg["unroll_steps"], td_steps=cfg["td_steps"])
    
    best_score_overall = 0.0
    for i in range(3):
        buffer, scores = self_play(model, buffer, cfg)
        if len(scores) > 0:
            max_s = float(np.max(scores))
            best_score_overall = max(best_score_overall, max_s)
        
        if len(buffer) > 0:
            train(model, buffer, optimizer, cfg, i)
            
        trial.report(best_score_overall, i)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    return best_score_overall

def run_study() -> None:
    wandb_kwargs = {"project": "tricked_hpo"}
    wandbc = WeightsAndBiasesCallback(metric_name="score", wandb_kwargs=wandb_kwargs)

    study = optuna.create_study(direction="maximize", study_name="tricked_hpo", storage="sqlite:///optuna.db", load_if_exists=True)
    study.optimize(objective, n_trials=20, callbacks=[wandbc])
    print("Best params:", study.best_params)
    print("Best value:", study.best_value)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    run_study()
