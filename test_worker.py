import torch

from tricked.mcts.search import MuZeroMCTS
from tricked.model.network import MuZeroNet
from tricked.training.self_play import play_one_game

print("Booting fake worker...")
model = MuZeroNet()
mcts = MuZeroMCTS(model, torch.device("cpu"))
print("Starting play_one_game...")
try:
    episode, score = play_one_game(1, mcts, 5, 1, 1)
    print("Game finished with score:", score)
except Exception as e:
    import traceback
    traceback.print_exc()
    print("CRASHED:", e)
