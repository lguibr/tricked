import optuna
import optunahub
import sqlite3
import json
import os
import sys
import argparse
import time

# We need the training loop logic from train.py but modified to evaluate and return a score.
# Alternatively, I can just write the objective function right here!

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "engine", "target", "release")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "extensions")))

import tricked_engine
import torch
import numpy as np
import threading
from backend.models.muzero import MuZeroNet, InitialInferenceModel, RecurrentInferenceModel
from backend.models.bptt import BPTTKernel

def generate_trial_config(base_config, bounds, trial):
    # Base config is cloned
    trial_config = json.loads(json.dumps(base_config))
    
    # Simple parse of bounds and override using Optuna suggestions
    bounds_dict = json.loads(bounds) if isinstance(bounds, str) else bounds
    for key, range_val in bounds_dict.items():
        if isinstance(range_val, list) and len(range_val) == 2:
            min_v = float(range_val[0])
            max_v = float(range_val[1])
            suggested = trial.suggest_float(key, min_v, max_v, log=True)
            
            # Key comes as "optimizer.lr_init"
            parts = key.split('.')
            d = trial_config
            for p in parts[:-1]:
                d = d[p]
            d[parts[-1]] = suggested
            
    return trial_config

def update_trial_metrics(db_path, study_id, trial_id, step, value, elapsed):
    # This manually inserts metrics simulating a TRIAL so the front-end renders it!
    # A trial will act like a sub-run
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        cursor = conn.cursor()
        
        # Ensure the trial is registered in `runs`
        cursor.execute("INSERT OR IGNORE INTO runs (id, name, type, status, config, tags, artifacts_dir) VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (trial_id, f"Trial {trial_id}", "TRIAL", "RUNNING", "{}", json.dumps([study_id]), ""))
        
        cursor.execute("INSERT INTO metrics (run_id, step, total_loss, elapsed_time) VALUES (?, ?, ?, ?)",
                       (trial_id, step, value, elapsed))
        
        conn.commit()
    except Exception as e:
        print(f"Failed to update metrics: {e}")
    finally:
        conn.close()

def optimize_objective(trial, db_path, study_id, base_config, config_bounds, max_steps):
    config = generate_trial_config(base_config, config_bounds, trial)
    trial_id = f"{study_id}_{trial.number}"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Trial {trial.number}] Initializing engine...")
    
    capacity = config["optimizer"]["buffer_capacity_limit"]
    inf_batch = config["hardware"]["inference_batch_size_limit"]
    inf_timeout = config["hardware"]["inference_timeout_ms"]
    
    # We serialize the config for engine
    config_str = json.dumps(config)
    engine = tricked_engine.TrickedEngine(capacity, config_str)
    
    hidden_dim = config["architecture"]["hidden_dimension_size"]
    blocks = config["architecture"]["num_blocks"]
    support = config["optimizer"].get("value_support_size", 300)
    
    net = MuZeroNet(hidden_dim, blocks, support, 20).to(device)
    bptt = BPTTKernel(hidden_dim, blocks, support, 20).to(device)
    bptt.active_net.load_state_dict(net.state_dict(), strict=False)
    
    lr = config["optimizer"]["lr_init"]
    optimizer = torch.optim.AdamW(bptt.active_net.parameters(), lr=lr, weight_decay=config["optimizer"]["weight_decay"])
    
    # Export Models for Native Rust Engine
    os.makedirs(f"optuna_models_{trial_id}", exist_ok=True)
    initial_scripted = torch.jit.script(InitialInferenceModel(net).eval())
    initial_scripted.save(os.path.join(f"optuna_models_{trial_id}", "initial_model.pt"))
    recurrent_scripted = torch.jit.script(RecurrentInferenceModel(bptt.active_net).eval())
    recurrent_scripted.save(os.path.join(f"optuna_models_{trial_id}", "recurrent_model.pt"))
    
    engine.start_workers(
        config["hardware"]["num_processes"], 
        os.path.join(f"optuna_models_{trial_id}", "initial_model.pt"), 
        os.path.join(f"optuna_models_{trial_id}", "recurrent_model.pt"), 
        device.type == "cuda"
    )
    
    start_time = time.time()
    print(f"[Trial {trial.number}] Engine polling engaged, running for {max_steps} metrics steps.")
    best_loss = 9999.0
    
    try:
        for step in range(max_steps):
            time.sleep(1) # We would realistically do backward pass here!
            current_loss = 1.0 / (step + 1) * lr * 1000.0 # Just a simulated loss for POC
            
            elapsed = time.time() - start_time
            update_trial_metrics(db_path, study_id, trial_id, step, current_loss, elapsed)
            
            trial.report(current_loss, step)
            if trial.should_prune():
                raise optuna.pruners.TrialPruned()
                
            if current_loss < best_loss:
                best_loss = current_loss
                
    finally:
        engine.stop_workers()
        
        # Complete Trial in Database
        conn = sqlite3.connect(db_path, timeout=10)
        cursor = conn.cursor()
        cursor.execute("UPDATE runs SET status = 'COMPLETED' WHERE id = ?", (trial_id,))
        conn.commit()
        conn.close()

    return best_loss

def run_study(study_id, db_path):
    print(f"🚀 Initializing Optuna Study for: {study_id}")
    conn = sqlite3.connect(db_path, timeout=10)
    cursor = conn.cursor()
    cursor.execute("SELECT config, artifacts_dir FROM runs WHERE id = ?", (study_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        print("Tuning run not found in DB!")
        sys.exit(1)
        
    config_json_raw = row[0]
    study_config = json.loads(config_json_raw)
    
    # Grab base config!
    artifacts_dir = row[1]
    base_config_path = os.path.join(artifacts_dir, "base_config.json")
    
    with open(base_config_path, "r") as f:
        base_config = json.loads(f.read())
        
    bounds = study_config["bounds"]
    trials = study_config["trials"]
    max_steps = study_config["max_steps"]
    
    # Execute Optuna natively within SQLite so it logs properly!
    storage_url = f"sqlite:///optuna_{study_id}.db"
    
    module_cma = optunahub.load_module(package="samplers/cma_es_refinement")
    sampler = module_cma.CmaEsRefinementSampler(seed=42)
    
    module_wilcoxon = optunahub.load_module(package="pruners/wilcoxon")
    pruner = module_wilcoxon.WilcoxonPruner(p_threshold=0.1)

    study = optuna.create_study(
        study_name=study_id,
        storage=storage_url,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=pruner
    )
    
    def objective(trial):
        return optimize_objective(trial, db_path, study_id, base_config, bounds, max_steps)
        
    study.optimize(objective, n_trials=trials)


