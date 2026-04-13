import argparse
import sys
from tricked.training.loop import run_training
from tricked.tuning.optuna_study import run_study

def main():
    parser = argparse.ArgumentParser(description="Tricked Orchestrator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    train_parser.add_argument("--id", type=str, required=True, help="Run ID")
    train_parser.add_argument("--db", type=str, required=True, help="SQLite DB path")

    # Tune command
    tune_parser = subparsers.add_parser("tune")
    tune_parser.add_argument("--id", type=str, required=True, help="Study ID (run ID in DB)")
    tune_parser.add_argument("--db", type=str, required=True, help="Path to SQLite Workspace DataBase")

    args = parser.parse_args()

    if args.command == "train":
        print(f"Starting standard training loop with config: {args.config}", flush=True)
        run_training(args.config, args.id, args.db)
    elif args.command == "tune":
        print(f"Starting Optuna Tuning for Study ID: {args.id}")
        run_study(args.id, args.db)
    else:
        print("Unknown command.")
        sys.exit(1)

if __name__ == "__main__":
    main()
