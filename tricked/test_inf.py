import sys, os
os.environ["LIBTORCH_USE_PYTORCH"] = "1"
os.environ["LIBTORCH_BYPASS_VERSION_CHECK"] = "1"
import torch
from tricked.engine.src import tricked_engine
print("Engine loaded.")
