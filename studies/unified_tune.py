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
import select
from subprocess import Popen

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, required=True, help="Path to base config JSON"
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
parser.add_argument("--resnet-blocks", type=int, default=10)
parser.add_argument("--resnet-channels", type=int, default=256)
parser.add_argument("--bounds", type=str, default="{}")
args = parser.parse_args()

with open(args.config, "r") as f:
    BASE_CONFIG = json.load(f)

BASE_CONFIG["resnet_blocks"] = args.resnet_blocks
BASE_CONFIG["resnet_channels"] = args.resnet_channels

BOUNDS = json.loads(args.bounds)


def get_bound(key, default_min, default_max):
    if key in BOUNDS:
        return BOUNDS[key]["min"], BOUNDS[key]["max"]
    return default_min, default_max


def export_callback(study, trial):
    import json
    import os
    import optuna

    trials_data = []
    for t in study.trials:
        # Multi-objective trial value is an array of values
        val = t.values if hasattr(t, "values") else t.value
        trials_data.append(
            {
                "number": t.number,
                "state": t.state.name,
                "value": val,
                "params": t.params,
                "intermediate_values": t.intermediate_values or {},
            }
        )

    try:
        # Note: Feature importance for multi-objective is currently limited in some Optuna versions, fallback to default if failing.
        importance = optuna.importance.get_param_importances(
            study,
            target=lambda t: (
                t.values[1] if t.values and len(t.values) > 1 else float("inf")
            ),
        )
    except Exception:
        importance = {}

    tmp_file = "studies/optuna_study.json.tmp"
    os.makedirs("studies", exist_ok=True)
    with open(tmp_file, "w") as f:
        json.dump({"trials": trials_data, "importance": importance}, f)
    os.replace(tmp_file, "studies/optuna_study.json")


def objective(trial):
    config = BASE_CONFIG.copy()

    # Hardware Parms
    w_min, w_max = get_bound("num_processes", 8, 32)
    config["num_processes"] = trial.suggest_int("num_processes", w_min, w_max)

    # Optional logic: only sweep batch sizes if they actually provided bounds for them
    if "train_batch_size" in BOUNDS:
        b_min, b_max = get_bound("train_batch_size", 64, 4096)
        config["train_batch_size"] = trial.suggest_int(
            "train_batch_size", b_min, b_max, step=64
        )

    # MCTS Params
    s_min, s_max = get_bound("simulations", 10, 2000)
    config["simulations"] = trial.suggest_int("simulations", s_min, s_max, step=10)

    g_min, g_max = get_bound("max_gumbel_k", 4, 64)
    config["max_gumbel_k"] = trial.suggest_int("max_gumbel_k", g_min, g_max)

    # Learning Params
    lr_min, lr_max = get_bound("lr_init", 1e-5, 1e-2)
    config["lr_init"] = trial.suggest_float("lr_init", lr_min, lr_max, log=True)

    experiment_name = f"unified_tune_trial_{trial.number:03d}"
    import sqlite3
    from datetime import datetime

    workspace_db = "tricked_workspace.db"
    conn = sqlite3.connect(workspace_db)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            status TEXT NOT NULL,
            config JSON,
            tags JSON,
            artifacts_dir TEXT,
            start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            end_time DATETIME
        )
    """)
    artifacts_dir = f"artifacts/{experiment_name}"
    conn.execute(
        """
        INSERT OR REPLACE INTO runs (id, name, type, status, config, tags, artifacts_dir, start_time) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            experiment_name,
            experiment_name,
            "TUNING_TRIAL",
            "RUNNING",
            json.dumps(config),
            json.dumps(["unified_tune", config["worker_device"]]),
            artifacts_dir,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    conn.commit()
    conn.close()

    import os

    bin_path = "target/release/tricked_engine"
    if not os.path.exists(bin_path):
        bin_path = "target/debug/tricked_engine"

    cmd = [
        bin_path,
        "train",
        "--run-id",
        experiment_name,
        "--workspace-db",
        workspace_db,
        "--max-steps",
        str(args.max_steps),
    ]

    print(f"\n[Unified Tune Trial {trial.number}] Starting Holistic Evaluation...")

    env = os.environ.copy()

    import glob

    cargo_libtorch_paths = glob.glob(
        "target/release/build/torch-sys-*/out/libtorch/libtorch/lib"
    )
    if not cargo_libtorch_paths:
        cargo_libtorch_paths = glob.glob(
            "target/debug/build/torch-sys-*/out/libtorch/libtorch/lib"
        )

    if cargo_libtorch_paths:
        torch_lib_path = os.path.abspath(cargo_libtorch_paths[0])
        if "LD_LIBRARY_PATH" in env:
            env["LD_LIBRARY_PATH"] = f"{torch_lib_path}:{env['LD_LIBRARY_PATH']}"
        else:
            env["LD_LIBRARY_PATH"] = torch_lib_path

    process: "Popen[str]" = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
        env=env,
    )

    hotpath_mcts_avg = float("inf")
    final_loss = float("inf")
    last_reported_step = -1

    def parse_hotpath_line(line):
        nonlocal hotpath_mcts_avg
        if "search::mcts_search" in line and "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) > 4:
                avg_str = parts[3]
                try:
                    if "ms" in avg_str:
                        hotpath_mcts_avg = (
                            float(avg_str.replace("ms", "").strip()) / 1000.0
                        )
                    elif "µs" in avg_str:
                        hotpath_mcts_avg = (
                            float(avg_str.replace("µs", "").strip()) / 1000000.0
                        )
                    elif "ns" in avg_str:
                        hotpath_mcts_avg = (
                            float(avg_str.replace("ns", "").strip()) / 1e9
                        )
                    elif "s" in avg_str:
                        hotpath_mcts_avg = float(avg_str.replace("s", "").strip())
                except ValueError:
                    pass

    stdout = process.stdout
    assert stdout is not None

    try:
        start_time = time.time()
        last_db_check_time = start_time

        while process.poll() is None:
            reads, _, _ = select.select([stdout], [], [], 1.0)

            if stdout in reads:
                # Make stdout non-blocking
                import fcntl

                try:
                    fd = stdout.fileno()
                    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
                    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
                except Exception:
                    pass

                while True:
                    try:
                        line = stdout.readline()
                        if not line:
                            break
                        sys.stdout.write(f"[CHILD_ID:{experiment_name}] {line}")
                        sys.stdout.flush()

                        parse_hotpath_line(line)

                        if "FINAL_EVAL_SCORE:" in line:
                            try:
                                final_loss = float(
                                    line.strip().split("FINAL_EVAL_SCORE:")[1]
                                )
                            except Exception:
                                pass
                    except (BlockingIOError, IOError):
                        break

                try:
                    fcntl.fcntl(fd, fcntl.F_SETFL, fl)
                except Exception:
                    pass

            current_time = time.time()
            if current_time - last_db_check_time >= 2.0:
                last_db_check_time = current_time
                try:
                    import sqlite3

                    conn_metrics = sqlite3.connect("tricked_workspace.db")
                    # Optimize: only fetch the latest step
                    df = pd.read_sql_query(
                        f"SELECT step, total_loss, vram_usage_mb FROM metrics WHERE run_id = '{experiment_name}' ORDER BY step DESC LIMIT 1",
                        conn_metrics,
                    )
                    conn_metrics.close()

                    if not df.empty:
                        last_vram = df["vram_usage_mb"].iloc[0]
                        last_step = df["step"].iloc[0]
                        last_loss = df["total_loss"].iloc[0]

                        if last_vram > 11500:
                            print(
                                f"[CHILD_ID:{experiment_name}] [Trial {trial.number}] PRUNED: VRAM limit."
                            )
                            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                            raise optuna.TrialPruned()

                        if last_step > last_reported_step:
                            trial.report(last_loss, last_step)
                            last_reported_step = last_step
                            try:
                                export_callback(study, trial)
                            except Exception:
                                pass

                        if trial.should_prune():
                            print(
                                f"[CHILD_ID:{experiment_name}] [Trial {trial.number}] PRUNED: Trajectory unpromising."
                            )
                            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                            raise optuna.TrialPruned()
                except Exception:
                    pass

            if time.time() - start_time > args.timeout:
                print(f"[Trial {trial.number}] TIMEOUT: Exceeded {args.timeout}s.")
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                raise optuna.TrialPruned()

        for line in stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            parse_hotpath_line(line)
            if "FINAL_EVAL_SCORE:" in line:
                try:
                    final_loss = float(line.strip().split("FINAL_EVAL_SCORE:")[1])
                except Exception:
                    pass
    finally:
        end_time = time.time()
        if process.poll() is None:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait()

    if process.returncode != 0:
        print(f"[Trial {trial.number}] KILLED: Engine Panic.")
        raise optuna.TrialPruned()

    wall_clock_time = end_time - start_time
    hardware_metric = (
        wall_clock_time + (hotpath_mcts_avg * 100)
        if hotpath_mcts_avg != float("inf")
        else wall_clock_time
    )

    # Return Multi-Objective: Minimize Hardware Penalty, Minimize Loss
    return hardware_metric, final_loss


if __name__ == "__main__":
    storage_name = "sqlite:///studies/unified_optuna_study.db"

    # Modern Optuna handles multi-objective natively in TPESampler
    sampler = optuna.samplers.TPESampler()

    try:
        wilcoxon_module = optunahub.load_module("pruners/wilcoxon")
        pruner = wilcoxon_module.WilcoxonPruner(p_threshold=0.1)
        print("✅ Wilcoxon Pruner Armed")
    except Exception:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        study_name="tricked_ai_holistic_tuning",
        directions=["minimize", "minimize"],
        storage=storage_name,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    print("⚙️  Starting Unified Holistic Tuning Phase...")

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

    print("\n✅ Holistic Tuning Complete!")
    try:
        best_trials = study.best_trials
        print(
            f"Discovered {len(best_trials)} optimal configurations along Pareto front."
        )
        if len(best_trials) > 0:
            best_trial = best_trials[0]
            best_config = BASE_CONFIG.copy()
            best_config.update(best_trial.params)
            with open("studies/best_unified_config.json", "w") as f:
                json.dump(best_config, f, indent=4)
    except Exception as e:
        print("⚠️ Could not write best_unified_config.json:", e)
