
import sys, json
sys.stdout.write("READY\n")
sys.stdout.flush()
for line in sys.stdin:
    data = json.loads(line)
    if data["action"] == "ask":
        sys.stdout.write(json.dumps({"trial_number": 1, "config": {}}) + "\n")
        sys.stdout.flush()
    elif data["action"] == "tell":
        sys.stdout.write("ACK\n")
        sys.stdout.flush()
        break
