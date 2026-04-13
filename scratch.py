import sys
import os
sys.path.append(os.path.abspath("backend"))
from engine import get_tricked_engine

engine = get_tricked_engine()

import random
scores = []
for _ in range(100):
   state = engine.GameStateExt(None, 0, 0, 6, 0)
   while not state.terminal:
       valid_moves = []
       for slot in range(3):
           if state.available[slot] != -1:
               valid_moves.append(slot)
       if not valid_moves:
           break
       # We need to find standard pieces
       pass # to be written correctly
