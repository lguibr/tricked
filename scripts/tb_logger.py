import os
# Monkey-patch os.makedirs to fix python 3.13 + tensorboardX FileExistsError
_orig_makedirs = os.makedirs
os.makedirs = lambda name, mode=0o777, exist_ok=False: _orig_makedirs(name, mode, exist_ok=True)

import json
import redis
from tensorboardX import SummaryWriter

r = redis.Redis(host='localhost', port=6379, decode_responses=True)
print("🔗 Connected to Redis. TensorBoard Logger listening for metrics...")

current_exp = r.get("tricked_current_exp") or "tricked_headless"
writer = SummaryWriter(log_dir=f"runs/{current_exp}")

pubsub = r.pubsub()
pubsub.subscribe('tricked_training', 'tricked_events', 'tricked_games', 'tricked_metrics', 'tricked_config')

import time

step = 0
start_time = time.time()
try:
    for message in pubsub.listen():
        if message['type'] == 'message':
            channel = message['channel']
            try:
                data = json.loads(message['data'])
            except:
                continue

            if channel == 'tricked_config':
                if "exp_name" in data:
                    new_exp = data["exp_name"]
                    if new_exp != current_exp:
                        current_exp = new_exp
                        print(f"🔄 Switching TensorBoard log dir to runs/{current_exp}")
                        writer.close()
                        writer = SummaryWriter(log_dir=f"runs/{current_exp}")
                        step = 0 # reset steps for new run
                
                config_str = json.dumps(data, indent=2)
                writer.add_text("Metadata/Config", f"```json\n{config_str}\n```", step)

            elif channel == 'tricked_training':
                if "loss" in data:
                    writer.add_scalar("train/loss", data.get("loss"), step)
                if "policy_loss" in data:
                    writer.add_scalar("train/policy_loss", data.get("policy_loss"), step)
                if "value_loss" in data:
                    writer.add_scalar("train/value_loss", data.get("value_loss"), step)
                if "reward_loss" in data:
                    writer.add_scalar("train/reward_loss", data.get("reward_loss"), step)
                step += 1
            elif channel == 'tricked_games':
                if "score" in data:
                    writer.add_scalar("eval/score", data.get("score"), step)
                if "steps" in data:
                    writer.add_scalar("eval/steps", data.get("steps"), step)
            elif channel == 'tricked_metrics':
                if "value" in data and "name" in data:
                    metric_name = data.get("name")
                    if "/" not in metric_name:
                        metric_name = f"metrics/{metric_name}"
                    writer.add_scalar(metric_name, data.get("value"), step)
                    if step % 5 == 0:
                        writer.flush()
        
        # Periodically flush in loop if idle? No, listen() blocks.
except KeyboardInterrupt:
    print("🛑 Shutting down TensorBoard logger")
    writer.close()
