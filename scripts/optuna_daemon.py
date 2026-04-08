import sys
import json
import optuna  # type: ignore
import os

study_name = sys.argv[1] if len(sys.argv) > 1 else "unified_tune"
workspace_db = sys.argv[2] if len(sys.argv) > 2 else "tricked_workspace.db"

storage_name = f"sqlite:///{workspace_db}"
os.makedirs("studies", exist_ok=True)
study = optuna.create_study(
    study_name=study_name,
    directions=["minimize", "minimize"],
    storage=storage_name,
    load_if_exists=True,
    sampler=optuna.samplers.GPSampler(),
    pruner=optuna.pruners.WilcoxonPruner(),
)


def get_bound(bounds, key, default_min, default_max):
    if key in bounds:
        return bounds[key]["min"], bounds[key]["max"]
    return default_min, default_max


def save_study_state():
    trials_data = []
    for t in study.trials:
        val = t.values if hasattr(t, "values") else t.value  # type: ignore
        trials_data.append(
            {
                "number": t.number,  # type: ignore
                "state": t.state.name,  # type: ignore
                "value": val,
                "params": t.params,  # type: ignore
                "intermediate_values": t.intermediate_values or {},  # type: ignore
            }
        )

    try:
        importance = optuna.importance.get_param_importances(
            study,
            target=lambda t: (
                t.values[1] if t.values and len(t.values) > 1 else float("inf")  # type: ignore
            ),
        )
    except Exception:
        importance = {}

    tmp_file = f"studies/{study_name}_optuna_study.json.tmp"
    with open(tmp_file, "w") as f:
        json.dump({"trials": trials_data, "importance": importance}, f)
    os.replace(tmp_file, f"studies/{study_name}_optuna_study.json")


print(json.dumps({"status": "ready"}), flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    try:
        req = json.loads(line)
        action = req.get("action")

        if action == "ask":
            base_config = req.get("config", {})
            bounds = req.get("bounds", {})

            trial = study.ask()
            config = base_config.copy()

            w_min, w_max = get_bound(bounds, "num_processes", 8, 32)
            config["num_processes"] = trial.suggest_int("num_processes", w_min, w_max)

            if "train_batch_size" in bounds:
                b_min, b_max = get_bound(bounds, "train_batch_size", 64, 4096)
                config["train_batch_size"] = trial.suggest_int(
                    "train_batch_size", b_min, b_max, step=64
                )

            s_min, s_max = get_bound(bounds, "simulations", 10, 2000)
            config["simulations"] = trial.suggest_int(
                "simulations", s_min, s_max, step=10
            )

            g_min, g_max = get_bound(bounds, "max_gumbel_k", 4, 64)
            config["max_gumbel_k"] = trial.suggest_int("max_gumbel_k", g_min, g_max)

            lr_min, lr_max = get_bound(bounds, "lr_init", 1e-5, 1e-1)
            config["lr_init"] = trial.suggest_float("lr_init", lr_min, lr_max, log=True)

            df_min, df_max = get_bound(bounds, "discount_factor", 0.9, 0.999)
            config["discount_factor"] = trial.suggest_float(
                "discount_factor", df_min, df_max
            )

            td_min, td_max = get_bound(bounds, "td_lambda", 0.5, 1.0)
            config["td_lambda"] = trial.suggest_float("td_lambda", td_min, td_max)

            wd_min, wd_max = get_bound(bounds, "weight_decay", 0.0, 0.1)
            config["weight_decay"] = trial.suggest_float("weight_decay", wd_min, wd_max)

            out = {"trial_number": trial.number, "config": config}
            save_study_state()
            print(json.dumps(out), flush=True)

        elif action == "tell":
            trial_number = req["trial_number"]
            pruned = req.get("pruned", False)
            if pruned:
                study.tell(trial_number, state=optuna.trial.TrialState.PRUNED)
            else:
                loss = req["loss"]
                hardware = req["hardware"]
                study.tell(trial_number, [hardware, loss])

            save_study_state()

            print(json.dumps({"status": "ok"}), flush=True)

    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}), flush=True)
