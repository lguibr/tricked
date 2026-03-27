import json
import redis
from aim import Run

r = redis.Redis(host='localhost', port=6379, decode_responses=True)
print("🔗 Connected to Redis. Aim Logger listening for metrics...")

run = Run(experiment="tricked_headless")

pubsub = r.pubsub()
pubsub.subscribe('tricked_training', 'tricked_events', 'tricked_games', 'tricked_metrics')

try:
    for message in pubsub.listen():
        if message['type'] == 'message':
            channel = message['channel']
            try:
                data = json.loads(message['data'])
            except:
                continue

            if channel == 'tricked_training':
                if "loss" in data:
                    run.track(data.get("loss"), name="loss", context={"subset": "train"})
                if "policy_loss" in data:
                    run.track(data.get("policy_loss"), name="policy_loss", context={"subset": "train"})
                if "value_loss" in data:
                    run.track(data.get("value_loss"), name="value_loss", context={"subset": "train"})
                if "reward_loss" in data:
                    run.track(data.get("reward_loss"), name="reward_loss", context={"subset": "train"})
            elif channel == 'tricked_games':
                if "score" in data:
                    run.track(data.get("score"), name="score", context={"subset": "eval"})
                if "steps" in data:
                    run.track(data.get("steps"), name="steps", context={"subset": "eval"})
            elif channel == 'tricked_metrics':
                if "value" in data and "name" in data:
                    run.track(data.get("value"), name=data.get("name"), context={"subset": "metrics"})
except KeyboardInterrupt:
    print("🛑 Shutting down Aim logger")
