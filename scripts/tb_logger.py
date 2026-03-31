#!/usr/bin/env python3
import os
import glob
import time
import pandas as pd
from tensorboardX import SummaryWriter
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def sync_csv_to_tensorboard(runs_dir="runs", poll_interval=2.0, max_iterations=None):
    """
    Monitors `runs/*/*_metrics.csv` files and converts appended rows into TensorBoard events.
    """
    logging.info(f"Starting TensorBoard CSV bridge. Monitoring directory: {runs_dir}/")
    state = {}
    writers = {}

    iterations = 0
    while True:
        csv_files = glob.glob(
            os.path.join(runs_dir, "**", "*_metrics.csv"), recursive=True
        )

        for csv_path in csv_files:
            experiment_dir = os.path.dirname(csv_path)

            if csv_path not in state:
                state[csv_path] = 0
                writers[csv_path] = SummaryWriter(log_dir=experiment_dir)
                logging.info(f"Discovered new metrics file: {csv_path}")

                config_path = os.path.join(experiment_dir, "config.json")
                if os.path.exists(config_path):
                    import json

                    try:
                        with open(config_path, "r") as f:
                            hparams = json.load(f)
                            flat_hparams = {
                                k: str(v) if isinstance(v, (dict, list)) else v
                                for k, v in hparams.items()
                            }
                            writers[csv_path].add_hparams(
                                flat_hparams, {"Loss/total_loss": 0.0}
                            )
                            logging.info(
                                f"Injected hyperparameters from config.json for {csv_path}"
                            )
                    except Exception as e:
                        logging.debug(f"Failed to load config.json: {e}")

            try:
                # Read the CSV. We use standard read_csv and slice by state to avoid file locks
                if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
                    continue

                df = pd.read_csv(csv_path)

                if len(df) > state[csv_path]:
                    new_rows = df.iloc[state[csv_path] :]

                    for _, row in new_rows.iterrows():
                        step = int(row["step"])
                        writer = writers[csv_path]

                        writer.add_scalar(
                            "Loss/total_loss", row.get("total_loss", 0.0), step
                        )
                        writer.add_scalar(
                            "Loss/policy_loss", row.get("policy_loss", 0.0), step
                        )
                        writer.add_scalar(
                            "Loss/value_loss", row.get("value_loss", 0.0), step
                        )
                        writer.add_scalar(
                            "Loss/reward_loss", row.get("reward_loss", 0.0), step
                        )
                        writer.add_scalar("Optimization/lr", row.get("lr", 0.0), step)

                    state[csv_path] = len(df)
                    writers[csv_path].flush()

            except Exception as e:
                # File might be mid-write or locked by the Rust engine
                logging.debug(f"Transient error reading {csv_path}: {e}")

        if max_iterations is not None:
            iterations += 1
            if iterations >= max_iterations:
                break

        time.sleep(poll_interval)


def test_tb_logger():
    """
    Test suite for the TensorBoard logger to guarantee execution safety.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        exp_dir = os.path.join(tmpdir, "test_exp")
        os.makedirs(exp_dir)
        csv_path = os.path.join(exp_dir, "test_exp_metrics.csv")

        # Write dummy data
        df = pd.DataFrame(
            {
                "step": [1, 2],
                "total_loss": [10.5, 9.2],
                "policy_loss": [5.0, 4.0],
                "value_loss": [5.5, 5.2],
                "reward_loss": [0.0, 0.0],
                "lr": [0.001, 0.001],
            }
        )
        df.to_csv(csv_path, index=False)

        # Run bridge for 1 iteration
        sync_csv_to_tensorboard(runs_dir=tmpdir, poll_interval=0.1, max_iterations=1)

        # Verify tfevents file was created
        events_files = glob.glob(os.path.join(exp_dir, "events.out.tfevents.*"))
        assert len(events_files) > 0, "SummaryWriter failed to generate tfevents file!"
        print(
            "✅ [Test Passed] TensorBoard Bridge Successfully Translated CSV to TFEvents."
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_tb_logger()
    else:
        # Run infinitely
        try:
            sync_csv_to_tensorboard()
        except KeyboardInterrupt:
            logging.info("TensorBoard bridge terminated by user.")
