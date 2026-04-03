#!/usr/bin/env python3
import optuna
import optunahub
import subprocess
import time
import pandas as pd
import os
import signal
import sys
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to base config JSON already optimized by hardware_tune",
)
parser.add_argument(
    "--trials", type=int, default=50, help="Number of hyperparameter suggestions"
)
parser.add_argument(
    "--max-steps",
    type=int,
    default=50,
    help="Number of training steps to evaluate per trial",
)
parser.add_argument(
    "--timeout",
    type=int,
    default=1800,
    help="Timeout in seconds before pruning a trial",
)
args = parser.parse_args()

with open(args.config, "r") as f:
    BASE_CONFIG = json.load(f)


def export_callback(study, trial):
    import json
    import os
    import optuna

    trials_data = []
    for t in study.trials:
        trials_data.append(
            {
                "number": t.number,
                "state": t.state.name,
                "value": t.value,
                "params": t.params,
                "intermediate_values": t.intermediate_values or {},
            }
        )
    try:
        importance = optuna.importance.get_param_importances(
            study, target=lambda t: t.value if t.value is not None else float("inf")
        )
    except Exception:
        importance = {}

    tmp_file = "studies/optuna_study.json.tmp"
    os.makedirs("studies", exist_ok=True)
    with open(tmp_file, "w") as f:
        json.dump({"trials": trials_data, "importance": importance}, f)
    os.replace(tmp_file, "studies/optuna_study.json")


def objective(trial):
    # Take the hardware-approved config as our static bedrock baseline
    config = BASE_CONFIG.copy()

    # Target: Algorithmic Learning Parameters
    config["lr_init"] = trial.suggest_float("lr_init", 1e-5, 1e-2, log=True)
    config["gumbel_scale"] = trial.suggest_float("gumbel_scale", 0.1, 10.0, log=True)
    config["reanalyze_ratio"] = trial.suggest_float("reanalyze_ratio", 0.0, 1.0)
    config["unroll_steps"] = trial.suggest_int("unroll_steps", 3, 10)
    config["temporal_difference_steps"] = trial.suggest_int(
        "temporal_difference_steps", 3, 15
    )
    config["temp_decay_steps"] = trial.suggest_int(
        "temp_decay_steps", 1000, 100000, log=True
    )
    config["max_gumbel_k"] = trial.suggest_int("max_gumbel_k", 2, 16)
    config["simulations"] = trial.suggest_int("simulations", 10, 200, log=True)

    try:
        export_callback(trial.study, trial)
    except Exception:
        pass

    experiment_name = f"learn_tune_trial_{trial.number:03d}"
    metrics_file = f"runs/{experiment_name}/{experiment_name}_metrics.csv"
    config_file = f"runs/{experiment_name}/config.json"

    os.makedirs(f"runs/{experiment_name}", exist_ok=True)
    with open(config_file, "w") as f:
        json.dump(config, f)

    run_info = {
        "id": experiment_name,
        "name": experiment_name,
        "type": "TUNING_TRIAL",
        "status": "COMPLETED",
        "config": json.dumps(config),
        "tag": "Learning Study",
    }
    with open(f"runs/{experiment_name}/run_info.json", "w") as f:
        json.dump(run_info, f)

    # We run slightly longer for learning metrics to stabilize and actually show gradient descent bounds
    cmd = [
        "cargo",
        "run",
        "--release",
        "--features=hotpath,hotpath-alloc",
        "--bin",
        "tricked_engine",
        "--",
        "train",
        "--experiment-name",
        experiment_name,
        "--config",
        config_file,
        "--max-steps",
        str(args.max_steps),
    ]

    print(
        f"\n[Learning Tune Trial {trial.number}] Starting Semantic Learning Search..."
    )

    import select
    from subprocess import Popen

    process: "Popen[str]" = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )

    final_loss = float("inf")
    last_reported_step = -1

    stdout = process.stdout
    assert stdout is not None

    try:
        start_time = time.time()
        while process.poll() is None:
            reads, _, _ = select.select([stdout], [], [], 2.0)
            if stdout in reads:
                line = stdout.readline()
                if not line:
                    continue
                sys.stdout.write(line)
                sys.stdout.flush()

                # Snag the very last evaluation score dumped by the system if it finishes clean
                if "FINAL_EVAL_SCORE:" in line:
                    try:
                        final_loss = float(line.strip().split("FINAL_EVAL_SCORE:")[1])
                    except Exception:
                        pass

            if os.path.exists(metrics_file):
                try:
                    df = pd.read_csv(metrics_file)
                    if (
                        not df.empty
                        and "step" in df.columns
                        and "total_loss" in df.columns
                    ):
                        last_step = df["step"].iloc[-1]
                        last_loss = df["total_loss"].iloc[-1]

                        if last_step > last_reported_step:
                            # Forward live loss back to optuna
                            trial.report(last_loss, last_step)
                            last_reported_step = last_step

                        # Extremely aggressive pruning: If the algorithm is learning worse than median runs, kill it to save compute!
                        if trial.should_prune():
                            print(
                                f"[Trial {trial.number}] PRUNED: Learning curve collapsed compared to prior thresholds."
                            )
                            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                            raise optuna.TrialPruned()
                except Exception:
                    pass

            if time.time() - start_time > args.timeout:  # Timeout max for learning runs
                print(
                    f"[Trial {trial.number}] TIMEOUT: Exceeded {args.timeout}s maximum."
                )
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                raise optuna.TrialPruned()

        for line in stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if "FINAL_EVAL_SCORE:" in line:
                try:
                    final_loss = float(line.strip().split("FINAL_EVAL_SCORE:")[1])
                except Exception:
                    pass

    finally:
        if process.poll() is None:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait()

    if process.returncode != 0:
        print(f"[Trial {trial.number}] KILLED: C++ Engine Panic.")
        raise optuna.TrialPruned()

    return final_loss


if __name__ == "__main__":
    storage_name = "sqlite:///studies/learning_optuna_study.db"

    print("📦 Loading HEBO Sampler for optimal continuous hyperparameter searching...")
    try:
        hebo_module = optunahub.load_module("samplers/hebo")
        sampler = hebo_module.HEBOSampler()
        print("✅ HEBOSampler Armed via OptunaHub (Option 3)")
    except Exception as e:
        print(f"⚠️ Could not load HEBOSampler. Error: {e}. Falling back to CMA-ES...")
        try:
            cmaes_module = optunahub.load_module("samplers/cma_es_refinement")
            sampler = cmaes_module.CmaEsRefinementSampler()
            print("✅ CMA-ES Refinement Armed (Option 1)")
        except Exception:
            sampler = None

    try:
        # Use Wilcoxon since we are comparing strict statistical distributions of live training loss curves
        wilcoxon_module = optunahub.load_module("pruners/wilcoxon")
        pruner = wilcoxon_module.WilcoxonPruner(p_threshold=0.1)
        print("✅ Wilcoxon Pruner Armed")
    except Exception:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    # We strictly want to MINIMIZE network loss iteratively
    study = optuna.create_study(
        study_name="tricked_ai_learning_velocity",
        direction="minimize",
        storage=storage_name,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    print("⚙️  Starting Learning Velocity Tune...")

    import signal

    def sigterm_handler(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, sigterm_handler)

    try:
        export_callback(study, None)
    except Exception:
        pass

    try:
        study.optimize(objective, n_trials=args.trials, callbacks=[export_callback])
    except KeyboardInterrupt:
        print("\n🛑 Optimization interrupted by user.")
        for t in study.trials:
            if t.state == optuna.trial.TrialState.RUNNING:
                try:
                    study.tell(t.number, state=optuna.trial.TrialState.FAIL)
                except Exception:
                    pass
    finally:
        try:
            export_callback(study, None)
        except Exception:
            pass

    print("\n✅ Algorithmic Learning Tuning Complete!")
    try:
        print(f"Best Trial Final Evaluation Score/Loss: {study.best_trial.value}")
        print("Optimal Learning Hyperparameters:")
        for k, v in study.best_trial.params.items():
            print(f"  {k}: {v}")

        best_config = BASE_CONFIG.copy()
        best_config.update(study.best_trial.params)
        with open("studies/best_learning_config.json", "w") as f:
            json.dump(best_config, f, indent=4)
    except Exception as e:
        print("⚠️ Could not write best_learning_config.json:", e)
