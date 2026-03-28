import redis
import json

r = redis.Redis(host='localhost', port=6379, decode_responses=True)
p = r.pubsub()
p.subscribe('tricked_metrics', 'tricked_training', 'tricked_config')
print("Listening...")
for msg in p.listen():
    print(msg)
