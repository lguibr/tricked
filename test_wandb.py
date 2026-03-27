import os
import wandb
from dotenv import load_dotenv

load_dotenv()
os.environ["WANDB_BASE_URL"] = "http://localhost:8081"
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")

print("Key length:", len(os.environ["WANDB_API_KEY"]))
print("Base URL:", os.environ["WANDB_BASE_URL"])

try:
    wandb.init(project="test_proj")
    print("Success!")
except Exception as e:
    print("Error:", repr(e))
