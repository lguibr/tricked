import os
import json
import redis
import wandb
import time
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("WANDB_API_KEY")
if not api_key:
    print("❌ WANDB_API_KEY not found in .env")
    exit(1)

base_url = os.getenv("WANDB_BASE_URL", "http://localhost:8081")
os.environ["WANDB_BASE_URL"] = base_url

print(f"🔄 Initializing W&B Local against host: {base_url}")
wandb.login(key=api_key, host=base_url)

run = wandb.init(
    project="tricked-ai-native",
    name="rust-headless-training",
    resume="allow"
)

r = redis.Redis(host='localhost', port=6379, decode_responses=True)
print("🔗 Connected to Redis. Listening for metrics...")

pubsub = r.pubsub()
pubsub.subscribe('tricked_training', 'tricked_events', 'tricked_games', 'tricked_metrics')

step: int = 0
try:
    for message in pubsub.listen():
        if message['type'] == 'message':
            channel = message['channel']
            data = json.loads(message['data'])
            if channel == 'tricked_training':
                wandb.log({"train/loss": data["loss"]})
            elif channel == 'tricked_games':
                wandb.log({
                    "eval/difficulty": data.get("difficulty"),
                    "eval/score": data.get("score"),
                    "eval/steps": data.get("steps")
                })
            elif channel == 'tricked_metrics':
                # Track arbitrary high frequency metrics like MCTS search ms
                wandb.log({data["name"]: data["value"]})
except KeyboardInterrupt:
    print("🛑 Shutting down W&B logger")
finally:
    wandb.finish()
