import optuna
import json
import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument("--trial", type=int, required=True)
parser.add_argument("--loss", type=float, required=True)
parser.add_argument("--hardware", type=float, required=True)
parser.add_argument("--pruned", action="store_true")
args = parser.parse_args()

storage_name = "sqlite:///studies/unified_optuna_study.db"
study = optuna.create_study(
    study_name="tricked_ai_holistic_tuning",
    directions=["minimize", "minimize"],
    storage=storage_name,
    load_if_exists=True,
)

try:
    if args.pruned:
        study.tell(args.trial, state=optuna.trial.TrialState.PRUNED)
    else:
        study.tell(args.trial, [args.hardware, args.loss])
except Exception as e:
    sys.exit(1)

trials_data = []
for t in study.trials:
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

print("Told Optuna successfully!")
