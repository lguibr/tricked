import os

# Monkey-patch os.makedirs to fix python 3.13 + tensorboardX FileExistsError
_orig_makedirs = os.makedirs
os.makedirs = lambda name, mode=0o777, exist_ok=False: _orig_makedirs(
    name, mode, exist_ok=True
)

import json
import redis
from tensorboardX import SummaryWriter

r = redis.Redis(host="localhost", port=6379, decode_responses=True)
print("🔗 Connected to Redis. TensorBoard Logger listening for metrics...")

current_exp = r.get("tricked_current_exp")
writer = None


def get_writer():
    global writer, current_exp
    if writer is None:
        if not current_exp:
            current_exp = "tricked_headless"
        writer = SummaryWriter(log_dir=f"runs/{current_exp}")
    return writer


pubsub = r.pubsub()
pubsub.subscribe(
    "tricked_training",
    "tricked_events",
    "tricked_games",
    "tricked_metrics",
    "tricked_config",
)

import time

train_step = 0
eval_step = 0
metric_step = 0
start_time = time.time()
try:
    for message in pubsub.listen():
        if message["type"] == "message":
            channel = message["channel"]
            try:
                data = json.loads(message["data"])
            except:
                continue

            if channel == "tricked_config":
                if "experiment_name_identifier" in data:
                    new_exp = data["experiment_name_identifier"]
                    if new_exp != current_exp:
                        current_exp = new_exp
                        print(f"🔄 Switching TensorBoard log dir to runs/{current_exp}")
                        if writer is not None:
                            writer.close()
                        writer = SummaryWriter(log_dir=f"runs/{current_exp}")
                        train_step = 0
                        eval_step = 0
                        metric_step = 0

                config_str = json.dumps(data, indent=2)
                get_writer().add_text(
                    "Metadata/Config", f"```json\n{config_str}\n```", train_step
                )

            elif channel == "tricked_training":
                if "loss" in data:
                    get_writer().add_scalar("train/loss", data.get("loss"), train_step)
                if "policy_loss" in data:
                    get_writer().add_scalar(
                        "train/policy_loss", data.get("policy_loss"), train_step
                    )
                if "value_loss" in data:
                    get_writer().add_scalar(
                        "train/value_loss", data.get("value_loss"), train_step
                    )
                if "reward_loss" in data:
                    get_writer().add_scalar(
                        "train/reward_loss", data.get("reward_loss"), train_step
                    )
                train_step += 1
            elif channel == "tricked_games":
                if "score" in data:
                    get_writer().add_scalar("eval/score", data.get("score"), eval_step)
                if "steps" in data:
                    get_writer().add_scalar("eval/steps", data.get("steps"), eval_step)
                eval_step += 1
            elif channel == "tricked_metrics":
                if "value" in data and "name" in data:
                    metric_name = data.get("name")
                    if "/" not in metric_name:
                        metric_name = f"metrics/{metric_name}"
                    get_writer().add_scalar(metric_name, data.get("value"), metric_step)
                    metric_step += 1
                    if metric_step % 5 == 0:
                        get_writer().flush()

        # Periodically flush in loop if idle? No, listen() blocks.
except KeyboardInterrupt:
    print("🛑 Shutting down TensorBoard logger")
    if writer is not None:
        writer.close()
