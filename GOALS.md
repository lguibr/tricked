# GOALS: Tricked AI Learning Objectives & Milestones

Based on the massive **100,000,000-game** `METRICS.html` Monte Carlo uniform distribution analysis, the mathematical baseline for a completely blind agent is now absolute. An AI that merely places pieces without spatial planning or topological understanding will:
- Average exactly **103.8 points**.
- Suffocate and die instantly clearing **0 lines** in **51.8%** of all games.
- Never place a piece larger than 5 triangles (Size 6 does not exist).
- Peak at a 99th Percentile (P99) score of **337 points**.

To mathematically prove that the AlphaZero/MuZero representation has transcended random geometric variance, the following milestones must be sequentially achieved during the Auto-Tuning reinforcement loop.

## 🎯 Benchmark 1: "Sight" (Escaping the Gravity Well)
The first goal of the localized Neural Network is to prove it can "see" the board and avoid the immediate topological traps that kill random agents 51.8% of the time.
- **Target Survival:** > 65 turns (Consistent)
- **Target Score:** > 180 points
- **Target Mechanics:** The agent must reliably clear at least 2 to 3 lines per episode, proving it understands that clearing lines frees up board space. 

## 🎯 Benchmark 2: "Planning" (P99 Mathematical Parity)
At this stage, the network's value head has stabilized, and the MCTS Gumbel tree successfully connects multiple piece placements together.
- **Target Survival:** > 100 turns (Consistent)
- **Target Score:** > 340 points (Officially beating 99% of 100 Million random games)
- **Target Mechanics:** The system recognizes that waiting for a specific piece to complete a 9-length coordinate axis is more valuable than uniformly placing pieces.

## 🎯 Benchmark 3: "Mastery" (Super-Human Intersection)
Achieving this state proves the AI understands the `apply_move` multi-axis intersection multiplier. Since clearing two intersecting lines simultaneously doubles the value of the overlapping trianguler nodes, the agent begins intentionally building "stars" before clearing them.
- **Target Survival:** > 300 turns
- **Target Score:** > 1,500 points
- **Target Mechanics:** The agent heavily prioritizes playing pieces at grid intersections (0,0,0 coordinate centers) to cascade line clears.

## 🎯 Benchmark 4: "God-Level" (Infinite Play)
Unlike Chess or Go, Tricked is a survival puzzle. Theoretical topological God-Level mastery means the agent can survive indefinitely by achieving a strictly positive clearance-to-clutter ratio.
- **Target Survival:** 1,000+ turns (Effectively unbounded)
- **Target Score:** 10,000+ points
- **Target Mechanics:** Board density stabilizes at < 40%. The agent strictly maintains open 5-length vectors to accommodate the massive 5-triangle piece geometry at all times, ensuring the 3-piece buffer is never choked.

---
### Hardware Implementation Note: D6 Dihedral Augmentation (12x Symmetry)
To reach God-Level rapidly, we will implement the **Dihedral D6 Data Augmentation (Phase 9)** algorithm. Tricked has 12 topological symmetries (6 Rotations, 6 Reflections). This allows us to multiply a single MCTS simulation into 12 distinct trajectories for the Neural Network.
**Crucial Bottleneck Prevention:** MCTS operates at blazing speeds. Appending 12x the data to the buffer could stall memory buses. Therefore, the symmetry module will be implemented purely as a `[usize; 96]` statically pre-computed coordinate lookup map. At the exact moment traversing the MCTS tree creates a training record, we will use this static map to instantly bit-shift the probabilities and `available_pieces` IDs across the 12 symmetries with zero overhead `Vec` allocations.
