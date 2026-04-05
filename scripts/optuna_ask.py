import optuna
import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--bounds", type=str, default="{}")
args = parser.parse_args()

with open(args.config, "r") as f:
    base_config = json.load(f)

bounds = json.loads(args.bounds)


def get_bound(key, default_min, default_max):
    if key in bounds:
        return bounds[key]["min"], bounds[key]["max"]
    return default_min, default_max


os.makedirs("studies", exist_ok=True)
storage_name = "sqlite:///studies/unified_optuna_study.db"
study = optuna.create_study(
    study_name="tricked_ai_holistic_tuning",
    directions=["minimize", "minimize"],
    storage=storage_name,
    load_if_exists=True,
)

trial = study.ask()

config = base_config.copy()
w_min, w_max = get_bound("num_processes", 8, 32)
config["num_processes"] = trial.suggest_int("num_processes", w_min, w_max)

if "train_batch_size" in bounds:
    b_min, b_max = get_bound("train_batch_size", 64, 4096)
    config["train_batch_size"] = trial.suggest_int(
        "train_batch_size", b_min, b_max, step=64
    )

s_min, s_max = get_bound("simulations", 10, 2000)
config["simulations"] = trial.suggest_int("simulations", s_min, s_max, step=10)

g_min, g_max = get_bound("max_gumbel_k", 4, 64)
config["max_gumbel_k"] = trial.suggest_int("max_gumbel_k", g_min, g_max)

lr_min, lr_max = get_bound("lr_init", 1e-5, 1e-2)
config["lr_init"] = trial.suggest_float("lr_init", lr_min, lr_max, log=True)

out = {"trial_number": trial.number, "config": config}
print(json.dumps(out))
