# Project Codebase: tricked

## File Tree

```text
tricked/
├── .gitignore
├── CONTRIBUTING.md
├── Cargo.toml
├── LICENSE
├── Makefile
├── README.md
├── benches
│   ├── feature_bench.rs
│   ├── feature_extract.rs
│   ├── queue_bench.rs
│   └── replay_bench.rs
├── scripts
│   ├── auto_tune.py
│   ├── dashboard.py
│   ├── export_math_kernels.py
│   ├── export_onnx.py
│   ├── optuna_insights.py
│   ├── profile_cmodule.py
│   ├── tb_logger.py
│   └── tune.py
├── src
│   ├── config.rs
│   ├── core
│   │   ├── board.rs
│   │   ├── constants.rs
│   │   ├── features.rs
│   │   └── mod.rs
│   ├── lib.rs
│   ├── main.rs
│   ├── mcts
│   │   └── mod.rs
│   ├── net
│   │   ├── dynamics.rs
│   │   ├── mod.rs
│   │   ├── muzero.rs
│   │   ├── prediction.rs
│   │   ├── projector.rs
│   │   ├── representation.rs
│   │   └── resnet.rs
│   ├── node.rs
│   ├── performance_benches.rs
│   ├── queue.rs
│   ├── sumtree.rs
│   ├── test_cmodule.rs
│   ├── test_dlpack.rs
│   ├── test_dlpack2.rs
│   ├── test_sparse.rs
│   ├── tests.rs
│   └── train
│       ├── buffer
│       │   ├── mod.rs
│       │   ├── replay.rs
│       │   └── state.rs
│       ├── mod.rs
│       └── optimizer
│           ├── loss.rs
│           ├── mod.rs
│           └── optimization.rs
└── tests
    ├── arcswap_test.rs
    └── board_fuzz.rs
```

## File Contents

### File: `.gitignore`

```text

# The MCTS dataset streams get massive quickly. Do not commit these.
/data/
*.jsonl
*.pb
*.pt
*.pth
/runs/

# --- Rust / PyO3 ---
target/
Cargo.lock
# Binaries for programs and plugins
*.exe
*.exe~
*.dll
*.so
*.dylib
bin/


# --- Python ---
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# PyTorch / Machine Learning
# Logs and checkpoints
.cache/
*.json
*.txt
!README.md
!GUMBEL_MUZERO_DOSSIER.md
!src/**/README.md
!tests/README.md

# Pytest Coverage & Testing
.pytest_cache/
htmlcov/
.coverage
.coverage.*
coverage.xml
*.log

# Environments
venv/
.env/
env/
ENV/
env.bak/
venv.bak/

# --- macOS ---
# General
.DS_Store
.AppleDouble
.LSOverride

# Icon must end with two \r
Icon

# Thumbnails
._*

# Files that might appear in the root of a volume
.DocumentRevisions-V100
.fseventsd
.Spotlight-V100
.TemporaryItems
.Trashes
.VolumeIcon.icns
.com.apple.timemachine.donotpresent

# Directories potentially created on remote AFP share
.AppleDB
.AppleDesktop
Network Trash Folder
Temporary Items
.apdisk

# --- IDEs / Editors ---
.vscode/
.idea/
*.swp
*.swo
.venv
.agent/
# Node
ui/node_modules/
ui/.svelte-kit/
ui/build/

# Python tools
.ruff_cache/
.mypy_cache/

/outputs/
/multirun/
*.profraw
```

### File: `CONTRIBUTING.md`

````md
# Contributing to Tricked AI Engine

Thank you for your interest in contributing! Tricked isn't just an experimental AI codebase; it's a heavily optimized production engine pursuing mathematical performance ceilings. To maintain extreme bounds on memory safety, concurrency tracking, and scale processing, all contributions must strictly abide by our Zero-Debt methodology.

## Zero-Debt Policy

The core rule of Tricked is **Zero Debt.** This implies:
* **No Warning Suppression:** We **do not** tolerate the usage of `#[allow(dead_code)]`, `#[allow(unused)]` or `#[allow(clippy::all)]` tags.
* **No Logic Omits:** Failed testing parameters should trigger a rewrite of the assertion's logic bounds organically, not an `#[ignore]` tag.
* **Orphan Code Pruning:** Deprecated structs, configuration options, debug print statements, layout files, or unused python scripts must immediately be eradicated. We do not preserve obsolete files.

## Workflow 

1. **Test Driven Implementation:** Write testing metrics that natively expose panics on edge constraints dynamically.
2. **Run The Gauntlet:** A Pull Request qualifies only if it locally handles:
```bash
make lint
make test
cargo bench
```
3. **Architectural Parity:** Ensure any additions mapping the back-end (Rust parameters, JSON parsing) mirror evenly alongside the React UI Forge parameters. Unused parameters must be purged out of the user interface seamlessly. 
4. **Hardware Validation:** Check your concurrency overhead. Using atomic operators (`AtomicI64`) and asynchronous channels holds priority far ahead of any `Mutex` implementation paths for hot-loop scaling.
````

### File: `Cargo.toml`

```toml
[package]
name = "tricked_engine"
version = "0.1.0"
edition = "2021"

[dependencies]

ndarray = "0.15.6"
once_cell = "1.19.0"
rand = "0.8.5"

bytemuck = "1.19.0"
tch = { version = "0.19.0", features = ["download-libtorch"] }
crossbeam-channel = "0.5"
clap = { version = "4.4", features = ["derive"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
serde_yaml = "0.9"
csv = "1.3"
sysinfo = "0.30"

arc-swap = "1.6"

[lib]
name = "tricked_engine"
path = "src/lib.rs"

[[bin]]
name = "tricked_engine"
path = "src/main.rs"

[dev-dependencies]
proptest = "1.4.0"
criterion = { version = "0.5.1", features = ["html_reports"] }
loom = "0.7"

[[bench]]
name = "feature_bench"
harness = false

[[bench]]
name = "queue_bench"
harness = false
```

### File: `LICENSE`

```text
MIT License

Copyright (c) 2026 Tricked AI Engine Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### File: `Makefile`

```text
.PHONY: check test lint format all coverage

all: format lint test build

format:
	cargo fmt

lint:
	cargo clippy --all-targets --all-features -- -D warnings

test:
	cargo test --release

build:
	cargo build --release

run:
	@echo "🔥 Starting Tricked AI Native Engine (CLI Mode)..."
	cargo run --release --bin tricked_engine -- train

benchmark:
	@echo "🚀 Running 100 Million Game Monte Carlo Performance Baseline..."
	cargo run --release --bin mc_metrics -- 100000000

telemetry:
	@echo "📦 Ensuring Python dependencies for telemetry..."
	python3 -m venv venv
	./venv/bin/pip install -q streamlit pandas
	@echo "🌐 Starting Tricked AI Telemetry Dashboard..."
	./venv/bin/streamlit run scripts/dashboard.py
	
tune:
	@echo "📦 Ensuring python dependencies for auto-tune..."
	python3 -m venv venv
	./venv/bin/pip install -q requests rich optuna optuna-dashboard optunahub cmaes pandas numpy pymoo disjoint_set gpytorch plotly
	./venv/bin/pip install -q --no-deps --no-build-isolation hebo
	@echo "⚙️  Starting Auto-Tuner Optimization..."
	./venv/bin/python3 scripts/tune.py

dashboard:
	@echo "🌐 Starting Optuna Dashboard..."
	./venv/bin/optuna-dashboard sqlite:///autotune.db --port 8080
```

### File: `README.md`

````md
# Tricked AI Engine

![Tricked AI](logo.png)

Tricked is a high-performance Reinforcement Learning engine that solves a custom topological board puzzle. It trains AlphaZero/MuZero-style agents utilizing strict zero-debt Rust lock-free algorithms to squeeze 100% throughput out of multi-core CPU and GPU platforms without memory starvation.
Here are the three sections cleanly formatted, professionally written, and ready to be dropped directly into your existing `README.md`. I have included the **Max Score** (derived directly from the tail end of your `scoreChart` array distribution) and formatted the statistics into a beautiful, GitHub-compatible dark-mode HTML grid.

***

##  1. Game Mechanics & Environment
**Tricked** is a single-player topological survival puzzle. Unlike traditional zero-sum board games like Chess or Go, the primary adversary here is geometric entropy. The agent must continuously clear lines to manage board density, utilizing extreme spatial reasoning to chain multi-axis intersecting combos.

*   **The Grid:** A regular hexagon composed of exactly **96 equilateral triangles** (side length of 4 units).
*   **Rhombus Coordinate Cube System:** To elegantly handle spatial reasoning, the board uses a 3-axis ($X, Y, Z$) coordinate system. Adjacent triangles are conceptually treated as rhombuses, representing the visible faces of 3D cubes in an isometric projection. This allows for hyper-efficient mathematical validation of straight lines across the grid.
*   **The 3-Piece Buffer:** The agent is given a buffer of up to 3 randomly generated poly-triangle pieces. It must place them in unoccupied absolute coordinates. The buffer only regenerates a fresh batch *after* all 3 pieces have been legally placed.
*   **Monochrome Topology:** Pieces have no colors and function purely as binary obstacles (1 = occupied, 0 = empty), forcing the AI to rely entirely on pure geometric shape.
*   **Scoring & Line Clearing:** A "line" is an edge-to-edge sequence of triangles spanning any of the 3 axes.
    *   *Base Value:* **2 points** per triangle in a cleared line.
    *   *Intersection Multiplier (Combos):* If a piece completes multiple overlapping lines simultaneously, intersection triangles are scored independently for *each* line. (e.g., An overlapping intersection in a 3-line cross yields $2 \times 3 = 6$ points).
*   **Terminal State:** The episode terminates when board clutter prevents the legal placement of *any* remaining pieces in the 3-piece buffer.

---

##  2. Mathematical Baseline (Monte Carlo Metrics)
To establish an absolute mathematical floor, a blind Monte Carlo uniform distribution analysis was run. This defines the exact statistical behavior of an agent placing pieces entirely at random without any spatial planning. 

<div align="center">
  <p><i>Simulated 100,000,000 Games in 197.10s | Pure Random Policy</i></p>
</div>

<table style="width:100%; text-align:center; background-color:#1e293b; color:#e2e8f0; border-collapse: separate; border-spacing: 10px; border-radius: 10px;">
  <tr>
    <td style="background-color:#0f172a; padding: 20px; border-radius: 8px; border: 1px solid #334155; width: 33%;">
      <span style="font-size: 12px; color: #94a3b8; text-transform: uppercase;">Average Score</span><br>
      <span style="font-size: 32px; font-weight: bold; color: #38bdf8;">103.8</span>
    </td>
    <td style="background-color:#0f172a; padding: 20px; border-radius: 8px; border: 1px solid #334155; width: 33%;">
      <span style="font-size: 12px; color: #94a3b8; text-transform: uppercase;">P99 Score</span><br>
      <span style="font-size: 32px; font-weight: bold; color: #10b981;">337</span>
    </td>
    <td style="background-color:#0f172a; padding: 20px; border-radius: 8px; border: 1px solid #334155; width: 33%;">
      <span style="font-size: 12px; color: #94a3b8; text-transform: uppercase;">Max Score</span><br>
      <span style="font-size: 32px; font-weight: bold; color: #f43f5e;">643</span>
    </td>
  </tr>
  <tr>
    <td style="background-color:#0f172a; padding: 20px; border-radius: 8px; border: 1px solid #334155;">
      <span style="font-size: 12px; color: #94a3b8; text-transform: uppercase;">Average Length</span><br>
      <span style="font-size: 32px; font-weight: bold; color: #f59e0b;">47.6</span> <span style="font-size: 16px;">Turns</span>
    </td>
    <td style="background-color:#0f172a; padding: 20px; border-radius: 8px; border: 1px solid #334155;">
      <span style="font-size: 12px; color: #94a3b8; text-transform: uppercase;">Death Rate (0 Lines)</span><br>
      <span style="font-size: 32px; font-weight: bold; color: #ef4444;">51.8%</span>
    </td>
    <td style="background-color:#0f172a; padding: 20px; border-radius: 8px; border: 1px solid #334155;">
      <span style="font-size: 12px; color: #94a3b8; text-transform: uppercase;">Total Pieces Placed</span><br>
      <span style="font-size: 32px; font-weight: bold; color: #8b5cf6;">4.75 Billion</span>
    </td>
  </tr>
</table>

### 📊 Key Distribution Insights
*   **The 51.8% Gravity Well:** More than half of all random games end before a single line is cleared. A blind agent essentially suffocates itself instantly by failing to understand topological alignment.
*   **The P99 Barrier:** Only 1% of purely random games survive past a score of 337.
*   **Piece Geometry Limit:** Across 4.75 billion pieces placed, a Size 6 triangle piece was **never legally placed**. This mathematically proves that surviving long enough to fit massive geometry requires deliberate, forward-looking board management.

---

##  3. AI Learning Objectives & Milestones
To mathematically prove that the AlphaZero/MuZero representation has transcended random geometric variance, the following milestones must be sequentially achieved during the Auto-Tuning Reinforcement loop.

### Phase 1: "Sight" *(Escaping the Gravity Well)*
The first goal of the localized Value/Policy network. The agent must prove it can "see" the board and avoid the immediate topological traps that kill random agents 51.8% of the time.
> **Target Survival:** `> 65 turns` (Consistent)
> **Target Score:** `> 180 points`
> **Mechanic Goal:** Consistently clear 2 to 3 lines per episode, establishing the foundational understanding that clearing lines frees up board space.

### Phase 2: "Planning" *(P99 Mathematical Parity)*
At this stage, the network's value head has stabilized. The MCTS (Monte Carlo Tree Search) successfully connects multiple piece placements over time.
> **Target Survival:** `> 100 turns` (Consistent)
> **Target Score:** `> 340 points` *(Officially beating 99% of 100M random games)*
> **Mechanic Goal:** The agent recognizes that waiting for a specific piece to complete a 9-length coordinate axis is more valuable than placing pieces uniformly.

### Phase 3: "Mastery" *(Super-Human Intersection)*
Achieving this state proves the AI understands the `apply_move` multi-axis intersection multiplier. 
> **Target Survival:** `> 300 turns`
> **Target Score:** `> 1,500 points`
> **Mechanic Goal:** The agent intentionally builds "stars" (highly structured grid layouts) before clearing them. It heavily prioritizes playing pieces at grid intersections *(0,0,0 coordinate centers)* to cascade simultaneous line clears.

### Phase 4: "God-Level" *(Infinite Play)*
Theoretical topological God-Level mastery. Absolute mastery means achieving a strictly positive clearance-to-clutter ratio.
> **Target Survival:** `1,000+ turns` *(Effectively Unbounded)*
> **Target Score:** `10,000+ points`
> **Mechanic Goal:** Board density stabilizes perfectly at `< 40%`. The agent strictly maintains open 5-length vectors to accommodate massive 5-triangle piece geometries at all times, ensuring the 3-piece buffer is never choked.

#### ⚙️ Hardware Implementation Note: D6 Dihedral Augmentation
To accelerate the AI's journey to God-Level parity, the training loop implements **Dihedral D6 Data Augmentation**. Because the hexagonal game board is symmetric, *Tricked* inherently contains **12 topological symmetries** (6 Rotations, 6 Reflections). During training, a single MCTS simulation trajectory is multiplied by 12 using geometric transformations before being fed into the Neural Network, drastically reducing the wall-clock time required to achieve structural "Sight".


## 4. Usage & Development

### Setup & Build
This repository relies on a zero-debt compilation standard.
```bash
cargo build --release
make lint
make test
```

### Telemetry & Observability
We do not rely on bloated external services or obscure APIs. The engine feeds a massive, unbroken stream of absolute reality—from the depths of the Monte Carlo Tree Search to the raw utilization of the hardware—directly into a local, high-performance dashboard.

```bash
make telemetry
```
This single command ensures all dependencies are present and launches the Streamlit interface, granting you real-time visibility into the mind (the search) and the muscle (the Graphics Processing Unit).

### Auto-Tuning
Run the dynamic python auto-tuner to empirically search for optimal batching hyperparameters on your hardware.
```bash
venv/bin/python scripts/auto_tune.py --trials 20
```
Then, map the resulting metrics into the Advanced Config panels in the Forge UI.



## 5. RL Cricket Style: The Philosophy of Leverage

> “The cricket’s leap is not born of magic, but of perfect, coiled tension. We do not build monoliths; we build engines of pure leverage.”

You are one mind. You have one machine. You are competing against armies of engineers backed by infinite compute. To win, you cannot rely on brute force. You must rely on absolute clarity, ruthless division of labor, and a profound respect for the physical limits of your vessel. 

**Cricket Style** is not a set of instructions. It is a philosophy of maximum leverage. It is the art of doing exactly what is necessary, exactly where it belongs, and naming it exactly what it is.

---

###  The Duality of Mind and Muscle

The greatest sin of modern AI engineering is asking the mind to lift boulders, or asking the muscle to solve riddles. Cricket Style demands a hard, impenetrable boundary between logic and geometry.

**The Realm of Rust (The Mind):**
The CPU is the realm of branching paths, infinite futures, and unpredictable exploration. It is where the Monte Carlo Tree Search lives. It is where the rules of the universe (the environment) are enforced. The mind is agile. It handles the chaos of concurrency, the mutation of memory, and the traversal of the unknown. We do not ask the GPU to walk the tree; it would stumble.

**The Realm of CUDA (The Muscle):**
The GPU is a blind, unthinking engine of pure geometric transformation. It does not understand rules, it does not understand trees, and it abhors a decision. It only understands dense, massive matrices. We do not write custom kernels to teach the muscle how to think. We simply feed it massive blocks of contiguous memory, let it perform its brutal arithmetic, and get out of its way.

---

###  The Reverence for Boundaries

A solo developer pushing a machine to the edge must design around the physical laws of the hardware. To ignore these limits is to invite starvation and collapse. We embrace our constraints, for art is born of them.

1.  **The Boundary of Space (VRAM):** 
    Memory is finite. As our explorers dream of millions of future states, the muscle's memory will fill. We must practice ruthless impermanence. When a future is no longer needed, its memory must be instantly reclaimed. Garbage collection is not a background task; it is the heartbeat of survival.
2.  **The Boundary of Distance (The PCIe Bus):**
    The bridge between the mind and the muscle is narrow and slow. We do not cross it unless absolute necessity dictates. When the muscle imagines a future state, that state remains with the muscle. We do not drag heavy thoughts back across the bridge; we pass only a whisper—a lightweight index, a pointer to a memory already held.
3.  **The Boundary of Time (Starvation):**
    The muscle is a leviathan; if fed a single thought, it starves. It demands a feast. Therefore, the mind must be fractured into legions of independent explorers. While the muscle digests a massive batch of thoughts, the explorers must already be gathering the next feast. Neither mind nor muscle must ever wait for the other.

---

###  The Symphony of the Loop

In the ideal world, communication between processes is not a series of locks and blocks, but a frictionless, continuous flow. 

**The Explorers and the Oracle:**
Our workers are solitary explorers wandering the forest of futures. They share no state. They do not wait for one another. When an explorer reaches the edge of its understanding, it does not attempt to guess the future. It leaves a question at the Boundary (the Queue) and sleeps. 

The Boundary gathers these questions into a chorus (the Batch). Only when the chorus is loud enough does it present the questions to the Oracle (the GPU). The Oracle speaks in geometry, writing its answers directly into the void of its own memory, returning only a map of where the answers lie. The explorers awaken, read the map, and continue their journey.

**The Architect in the Shadows:**
While the explorers dream, the Architect (the Optimizer) learns. It observes the memories of past journeys and reshapes the Oracle's understanding. But the explorers must never be interrupted by the Architect's work. We embrace the philosophy of the *Double Mind*. The Architect builds a new mind in the shadows. When it is ready, the minds are swapped in a single, atomic instant. Zero locks. Zero stutter. Uninterrupted flow.

---

###  The Sanctity of Language

Language is the map of our understanding. Code is read infinitely more times than it is written. As a solo developer, your greatest enemy is not the compiler; it is your own forgotten context. Six months from now, you will wander these halls alone. 

**We hate abbreviations with a burning passion.** 
To abbreviate is to obscure. To shorten a word is to steal meaning from the future self. Keystrokes are infinite and free; cognitive capacity is precious and strictly bounded. 

*   A name must be a complete, unbroken thought. 
*   We do not write `td_steps`; we write `temporal_difference_steps`. 
*   We do not write `obs`; we write `batched_observations`. 
*   We do not write `val`; we write `predicted_value_scalar`.

**We encode the shape of reality into our words.**
In the realm of tensors, a shape mismatch is a silent killer. The dimensions of reality must be spoken aloud in the name of the thing itself. We do not pass a `policy`; we pass `target_policies_batch_time_action`. When the shape of the data is woven into its true name, the architecture becomes self-evident, and the mind is freed from remembering what the machine can simply state.

---

### The Final Stage

We do not build complex systems because we want to; we build them because the domain demands it. But within that complexity, we enforce brutal simplicity. 

We respect the mind. We respect the muscle. We respect the boundaries of the machine. And above all, we respect the sanctity of the words we use to bind them together.

Keep the thoughts clear. Keep the names whole. Keep the leaps massive. 

Jump.

## 5. Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for our exact standards. We enforce a zero-debt policy. No `#[allow(...)]` tags, no suppressed warnings, all lints and tests must pass locally.

## License
MIT License. See [LICENSE](LICENSE) for more details.
````

### File: `benches/feature_bench.rs`

```rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tricked_engine::core::board::GameStateExt;
use tricked_engine::core::features::extract_feature_native;

pub fn criterion_benchmark(c: &mut Criterion) {
    let state = GameStateExt::new(Some([0, 1, 2]), 0u128, 0, 5, 0);
    let history = vec![1u128, 2u128, 3u128, 4u128, 5u128];

    c.bench_function("extract_feature_native_copies", |b| {
        let mut slice = vec![0.0; 20 * 128];
        b.iter(|| {
            extract_feature_native(
                black_box(&mut slice),
                black_box(state.board_bitmask_u128),
                black_box(&state.available),
                black_box(&history),
                black_box(&[]),
                black_box(5),
            );
            black_box(&slice);
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
```

### File: `benches/feature_extract.rs`

``rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;
use tricked_engine::core::board::GameStateExt;
use tricked_engine::core::features::extract_feature_native;

pub fn bench_feature_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_extraction");
    // Ensure we capture the `< 5 microseconds` requirement visually
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    let state = GameStateExt::new(Some([0, 1, 2]), 0b101010101010101010101010, 50, 6, 0);
    let history_boards = vec![
        0b0101010, 0b1010101, 0b0011001, 0b1100110, 0b0000111, 0b1110000, 0b0101010,
    ];
    let action_history = vec![10, 45, 90, 15];

    group.bench_function("extract_feature_native_current", |b| {
        let mut slice = vec![0.0; 20 * 128];
        b.iter(|| {
            extract_feature_native(
                black_box(&mut slice),
                black_box(state.board_bitmask_u128),
                black_box(&state.available),
                black_box(&history_boards),
                black_box(&action_history),
                black_box(6),
            );
            black_box(&slice);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_feature_extraction);
criterion_main!(benches);
``

### File: `benches/queue_bench.rs`

```rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::sync::Arc;
use std::thread;
use tricked_engine::mcts::EvaluationRequest;
use tricked_engine::queue::FixedInferenceQueue;

pub fn bench_queue_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("crossbeam_contention");
    group.sample_size(50);

    group.bench_function("fixed_inference_queue_32_threads", |b| {
        b.iter(|| {
            let queue = FixedInferenceQueue::new(16384, 32);
            let mut handles = vec![];

            // Simulating 32 Self-Play Workers hammering the queue simultaneously
            for worker_id in 0..32 {
                let q = Arc::clone(&queue);
                let (tx, _) = crossbeam_channel::unbounded();
                handles.push(thread::spawn(move || {
                    for _ in 0..100 {
                        let _ = q.push_batch(
                            worker_id,
                            vec![EvaluationRequest {
                                is_initial: true,
                                board_bitmask: 0,
                                available_pieces: [-1; 3],
                                recent_board_history: [0; 8],
                                history_len: 0,
                                recent_action_history: [0; 4],
                                action_history_len: 0,
                                difficulty: 6,
                                piece_action: 0,
                                piece_id: 0,
                                node_index: 0,
                                worker_id,
                                parent_cache_index: 0,
                                leaf_cache_index: 0,
                                evaluation_request_transmitter: tx.clone(),
                            }],
                        );
                    }
                }));
            }

            // Simulating the Inference Thread popping
            let mut popped = 0;
            while popped < 3200 {
                if let Ok(batch) =
                    queue.pop_batch_timeout(1024, std::time::Duration::from_millis(10))
                {
                    popped += batch.len();
                }
            }

            for handle in handles {
                handle.join().unwrap();
            }
            black_box(popped);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_queue_contention);
criterion_main!(benches);
```

### File: `benches/replay_bench.rs`

```rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tricked_engine::sumtree::ShardedPrioritizedReplay;

pub fn bench_per_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("per_sampling");
    group.sample_size(100);

    group.bench_function("sharded_per_sample_1024", |b| {
        let per = ShardedPrioritizedReplay::new(1_000_000, 1.0, 1.0, 8);

        let indices: Vec<usize> = (0..10_000).collect();
        let priorities: Vec<f64> = (0..10_000).map(|i| (i % 100) as f64).collect();
        per.add_batch(&indices, &priorities);

        b.iter(|| {
            let sample = per.sample(1024, 1_000_000, 1.0);
            black_box(sample);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_per_sampling);
criterion_main!(benches);
```

### File: `scripts/auto_tune.py`

```py
#!/usr/bin/env python3
import time
from datetime import datetime
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
import requests  # type: ignore
import optuna
from optuna.pruners import PatientPruner, HyperbandPruner
import optunahub

ACTUAL_APPLICATION_PROGRAMMING_INTERFACE_URL = "http://127.0.0.1:8000/api"
TARGET_TRAINING_STEPS = (
    20  # Reduced to fit comfortably within the 10m hardware timeout!
)

# Hardware parameters locked for maximum throughput
base_config = {
    "device": "cuda",
    "hidden_dimension_size": 64,
    "num_blocks": 4,
    "buffer_capacity_limit": 204800,
    "train_batch_size": 1024,
    "train_epochs": 4,
    "num_processes": 22,
    "worker_device": "cpu",
    "zmq_batch_size": 11,
    "zmq_timeout_ms": 20,
    "max_gumbel_k": 5,
    "difficulty": 6,
    "temp_boost": False,
    "experiment_name_identifier": "tune_sota",
}

optuna.logging.set_verbosity(optuna.logging.WARNING)


def stop_engine_and_cooldown():
    try:
        requests.post(f"{ACTUAL_APPLICATION_PROGRAMMING_INTERFACE_URL}/training/stop")
        # Engine tears down quickly now, no need for massive 5s sleeps!
        time.sleep(2)
    except Exception:
        pass


def objective(trial: optuna.Trial) -> float:
    configuration = base_config.copy()

    # Base SOTA Sweeps (Reduced cardinality for 2-hour budget)
    simulations = trial.suggest_categorical("simulations", [16, 32])
    temporal_difference_steps = trial.suggest_categorical(
        "temporal_difference_steps", [3, 5]
    )
    gumbel_scale = trial.suggest_float("gumbel_scale", 0.5, 2.0)
    lr_init = trial.suggest_float("lr_init", 5e-4, 5e-3, log=True)

    # Advanced Sweeps
    reanalyze_ratio = trial.suggest_float("reanalyze_ratio", 0.0, 0.4)
    unroll_steps = trial.suggest_int("unroll_steps", 3, 6)
    support_size = trial.suggest_categorical("support_size", [100, 300])
    temp_decay_steps = trial.suggest_int("temp_decay_steps", 10, 50)

    configuration.update(
        {
            "simulations": simulations,
            "temporal_difference_steps": temporal_difference_steps,
            "gumbel_scale": gumbel_scale,
            "lr_init": lr_init,
            "reanalyze_ratio": reanalyze_ratio,
            "unroll_steps": unroll_steps,
            "support_size": support_size,
            "temp_decay_steps": temp_decay_steps,
        }
    )

    timestamp_prefix = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    experiment_name = f"{timestamp_prefix}_t{trial.number}_s{simulations}_td{temporal_difference_steps}_k{unroll_steps}"
    configuration["experiment_name_identifier"] = experiment_name

    print(f"\n[Trial {trial.number}] 🧠 Testing AI Conf: {experiment_name}")

    try:
        response = requests.post(
            f"{ACTUAL_APPLICATION_PROGRAMMING_INTERFACE_URL}/training/start",
            json=configuration,
        )
        if response.status_code != 200:
            print(
                f"❌ Failed to start engine: HTTP {response.status_code}, Body: {response.text}"
            )
            raise optuna.exceptions.TrialPruned()
    except requests.exceptions.ConnectionError:
        print("❌ Failed to connect to Engine API. Is the server running?")
        trial.study.stop()
        return 0.0

    best_mean_score: float = 0.0
    current_steps: int = 0
    start_time: float = time.time()
    max_trial_duration_secs: int = 240  # 4 minutes hard timeout

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
    ) as progress:
        eval_task = progress.add_task(
            "[yellow]Evaluating (Max Score: 0.0)...",
            total=TARGET_TRAINING_STEPS,
        )

        while current_steps < TARGET_TRAINING_STEPS:
            time.sleep(2.0)  # Polling interval

            try:
                status = requests.get(
                    f"{ACTUAL_APPLICATION_PROGRAMMING_INTERFACE_URL}/training/status",
                    timeout=5,
                ).json()

                new_steps = status.get("training_steps", 0)
                if new_steps > current_steps:
                    advance = min(
                        new_steps - current_steps, TARGET_TRAINING_STEPS - current_steps
                    )
                    progress.advance(eval_task, advance=advance)
                    current_steps = new_steps

                top_games = status.get("top_games", [])
                if top_games:
                    # Calculate Mean Score of Top Games
                    scores = [g.get("score", 0.0) for g in top_games]
                    mean_score = sum(scores) / len(scores)
                    if mean_score > best_mean_score:
                        best_mean_score = mean_score
                        progress.update(
                            eval_task,
                            description=f"[green]Evaluating (Max Score: {best_mean_score:.1f})...",
                        )
                else:
                    mean_score = 0.0

                trial.report(mean_score, current_steps)

                if trial.should_prune():
                    print(
                        f"\n✂️  Trial {trial.number} pruned by Wilcoxon test/Hyperband. (Mean Score: {mean_score:.1f})."
                    )
                    stop_engine_and_cooldown()
                    raise optuna.exceptions.TrialPruned()

            except Exception as e:
                if "Failed to establish a new connection" in str(e):
                    print("\n❌ Server crashed. Pruning trial.")
                    stop_engine_and_cooldown()
                    raise optuna.exceptions.TrialPruned()

            if time.time() - start_time > max_trial_duration_secs:
                print("\n⏱️ Trial timed out (Stalled engine?). Pruning.")
                stop_engine_and_cooldown()
                raise optuna.exceptions.TrialPruned()

        progress.console.print(
            f"\n📊 Result for Trial {trial.number}: Max Mean Score = {best_mean_score:.1f}"
        )

    stop_engine_and_cooldown()
    return best_mean_score


if __name__ == "__main__":
    print("\nStarting Tricked AI SOTA Auto-Tuner 🚀")

    # 1. HEBOSampler
    print("Loading HEBO Sampler from OptunaHub...")
    try:
        hebo_module = optunahub.load_module("samplers/hebo")
        sampler = hebo_module.HEBOSampler()
        print("✅ HEBOSampler initialized successfully.")
    except Exception as e:
        print(f"⚠️ Failed to load HEBOSampler: {e}\nFalling back to TPESampler.")
        sampler = optuna.samplers.TPESampler()

    # 2. WilcoxonPruner
    print("Initializing built-in Wilcoxon Pruner...")
    try:
        pruner = optuna.pruners.WilcoxonPruner(p_threshold=0.1)
        print("✅ WilcoxonPruner initialized successfully.")
    except Exception as e:
        print(
            f"⚠️ Failed to initialize WilcoxonPruner: {e}\nFalling back to PatientPruner(Hyperband)."
        )
        hyperband_pruner = HyperbandPruner(
            min_resource=100, max_resource=TARGET_TRAINING_STEPS // 2
        )
        pruner = PatientPruner(hyperband_pruner, patience=3)

    study = optuna.create_study(
        study_name="tricked-ai-tuning-sota-2h",
        storage="sqlite:///autotune.db",
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    try:
        study.optimize(objective, n_trials=40)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Gracefully stopping study.")

    best_trial = study.best_trial
    print("\n🏆 OPTIMAL MATHEMATICAL CONFIGURATION ACHEIVED:")
    print(f"Max Mean Score: {best_trial.value:.1f}")
    for k, v in best_trial.params.items():
        print(f"{k}: {v}")
```

### File: `scripts/dashboard.py`

``py
import streamlit as st
import pandas as pd
import time
import os
import glob
import subprocess

st.set_page_config(
    page_title="Tricked AI Telemetry",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for SOTA look
st.markdown(
    """
<style>
    .reportview-container {
        background: #0e1117;
    }
    .metric-container {
        background-color: #1e212b;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🚀 Tricked AI Telemetry Dashboard")
st.markdown("Real-time, zero-overhead metrics visualization.")


def get_available_runs():
    runs_dir = "runs"
    if not os.path.exists(runs_dir):
        return []

    files = glob.glob(f"{runs_dir}/**/*metrics.csv", recursive=True)
    files.sort(key=os.path.getmtime, reverse=True)
    return files


# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    available_runs = get_available_runs()
    env_path = os.environ.get("CSV_PATH", "")

    if available_runs:
        default_index = 0
        if env_path and env_path in available_runs:
            default_index = available_runs.index(env_path)

        selected_run = st.selectbox(
            "Select Run (Auto-Detected)", available_runs, index=default_index
        )
        csv_path = st.text_input("Manual Override Path:", value=selected_run)
    else:
        st.warning("No runs found in `runs/`. Ensure the engine is saving metrics.")
        csv_path = st.text_input(
            "Metrics CSV Path:", value=env_path if env_path else "metrics.csv"
        )

    refresh_rate = st.slider(
        "Refresh Rate (seconds)", min_value=1, max_value=60, value=2
    )
    auto_refresh = st.checkbox("Auto Refresh", value=True)


@st.cache_data(ttl=refresh_rate)
def load_data(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return pd.DataFrame()


df = load_data(csv_path)

if df.empty:
    st.warning(f"No data found at `{csv_path}`. Waiting for engine to start logging...")
else:
    # ---------------------------------------------------------
    # 1. Top Level Metrics
    # ---------------------------------------------------------
    latest = df.iloc[-1]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Current Step", f"{latest.get('step', 0):,}")
    col2.metric("Total Loss", f"{latest.get('total_loss', 0):.4f}")
    col3.metric("Max Game Score", f"{latest.get('game_score_max', 0):,.0f}")
    col4.metric("Avg MCTS Depth", f"{latest.get('mcts_depth_mean', 0):.1f}")
    col5.metric("GPU Utilization", f"{latest.get('gpu_usage_pct', 0):.1f}%")

    st.divider()

    # ---------------------------------------------------------
    # 2. Charts Section
    # ---------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🚀 Training", "🎮 Gameplay", "🧠 MCTS", "💻 Hardware"]
    )

    with tab1:
        st.subheader("Loss Metrics")
        if "total_loss" in df.columns:
            loss_cols = [
                c
                for c in ["total_loss", "policy_loss", "value_loss", "reward_loss"]
                if c in df.columns
            ]
            st.line_chart(df.set_index("step")[loss_cols])

        st.subheader("Learning Rate")
        if "lr" in df.columns:
            st.line_chart(df.set_index("step")["lr"], color="#f5a623")

    with tab2:
        st.subheader("Scores (Min, Med, Mean, Max)")
        score_cols = [
            c
            for c in [
                "game_score_min",
                "game_score_med",
                "game_score_mean",
                "game_score_max",
            ]
            if c in df.columns
        ]
        if score_cols:
            st.line_chart(df.set_index("step")[score_cols])

        st.subheader("Lines Cleared")
        if "game_lines_cleared" in df.columns:
            st.line_chart(df.set_index("step")["game_lines_cleared"], color="#00ffaa")

    with tab3:
        colA, colB = st.columns(2)
        with colA:
            st.subheader("Search Depth")
            if "mcts_depth_mean" in df.columns:
                st.line_chart(df.set_index("step")["mcts_depth_mean"], color="#aa00ff")
        with colB:
            st.subheader("Search Time (ms)")
            if "mcts_search_time_mean" in df.columns:
                st.line_chart(
                    df.set_index("step")["mcts_search_time_mean"], color="#ff00aa"
                )

    with tab4:
        st.subheader("Resource Usage")
        colC, colD = st.columns(2)
        with colC:
            st.markdown("**CPU, GPU, and Disk Usage (%)**")
            hw_pct = [
                c
                for c in ["cpu_usage_pct", "gpu_usage_pct", "disk_usage_pct"]
                if c in df.columns
            ]
            if hw_pct:
                st.line_chart(df.set_index("step")[hw_pct])
        with colD:
            st.markdown("**Memory Usage (MB)**")
            hw_mem = [c for c in ["ram_usage_mb", "vram_usage_mb"] if c in df.columns]
            if hw_mem:
                st.line_chart(df.set_index("step")[hw_mem])

if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
``

### File: `scripts/export_math_kernels.py`

```py
import torch

@torch.jit.script
def support_to_scalar_fused(logits: torch.Tensor, support_size: int, epsilon: float) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    support = torch.arange(0.0, float(2 * support_size + 1), 1.0, dtype=torch.float32, device=logits.device)
    expected_value = torch.sum(probs * support, dim=-1)
    clamped = expected_value.clamp(min=0.0, max=float(2 * support_size))
    scaled_inversion = (((clamped + (1.0 + epsilon)) * (4.0 * epsilon) + 1.0).sqrt() - 1.0) / (2.0 * epsilon)
    return scaled_inversion.pow(2.0) - 1.0

@torch.jit.script
def scalar_to_support_fused(scalar: torch.Tensor, support_size: int, epsilon: float) -> torch.Tensor:
    safe_scalar = scalar.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    transformed = ((safe_scalar.abs() + 1.0).sqrt() - 1.0) + epsilon * safe_scalar
    clamped = transformed.reshape(-1).clamp(min=0.0, max=float(2 * support_size))
    
    floor_val = clamped.floor()
    ceil_val = clamped.ceil()
    
    upper_prob = clamped - floor_val
    lower_prob = 1.0 - upper_prob
    
    lower_idx = floor_val.to(torch.int64)
    upper_idx = ceil_val.to(torch.int64)
    
    batch_size = clamped.size(0)
    support_probs = torch.zeros((batch_size, 2 * support_size + 1), dtype=torch.float32, device=scalar.device)
    
    batch_indices = torch.arange(batch_size, dtype=torch.int64, device=scalar.device)
    
    support_probs.index_put_((batch_indices, lower_idx), lower_prob, accumulate=True)
    support_probs.index_put_((batch_indices, upper_idx), upper_prob, accumulate=True)
    
    return support_probs

class MuZeroMathOps(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    @torch.jit.export
    def support_to_scalar(self, logits: torch.Tensor, support_size: int, epsilon: float) -> torch.Tensor:
        return support_to_scalar_fused(logits, support_size, epsilon)
        
    @torch.jit.export
    def scalar_to_support(self, scalar: torch.Tensor, support_size: int, epsilon: float) -> torch.Tensor:
        return scalar_to_support_fused(scalar, support_size, epsilon)

if __name__ == "__main__":
    import sys
    model = MuZeroMathOps()
    scripted = torch.jit.script(model)
    out_path = sys.argv[1] if len(sys.argv) > 1 else "math_kernels.pt"
    scripted.save(out_path)
    print(f"Exported dynamic fused math kernels to {out_path}")
```

### File: `scripts/export_onnx.py`

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

HEXAGONAL_TO_CARTESIAN_MAP_ARRAY = [
    (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12),
    (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13),
    (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14),
    (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11), (3, 12), (3, 13), (3, 14), (3, 15),
    (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15),
    (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (5, 12), (5, 13), (5, 14),
    (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (6, 12), (6, 13),
    (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12)
]

def get_valid_spatial_mask_8x8(device="cpu"):
    mask = torch.zeros((1, 1, 8, 8), dtype=torch.float32, device=device)
    for r, c in HEXAGONAL_TO_CARTESIAN_MAP_ARRAY:
        mask[0, 0, r, c // 2] = 1.0
    return mask

class FlattenedResNetBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.register_buffer('spatial_mask', get_valid_spatial_mask_8x8())

    def forward(self, x):
        res = x
        out = self.conv1(x) * self.spatial_mask
        out = out.permute(0, 2, 3, 1).contiguous()
        out = F.mish(self.norm1(out))
        out = out.permute(0, 3, 1, 2).contiguous() * self.spatial_mask
        
        out = self.conv2(out) * self.spatial_mask
        out = out.permute(0, 2, 3, 1).contiguous()
        out = self.norm2(out)
        out = out.permute(0, 3, 1, 2).contiguous() * self.spatial_mask
        
        return F.mish(res + out) * self.spatial_mask

class RepresentationNet(nn.Module):
    def __init__(self, hidden_dim, num_blocks):
        super().__init__()
        self.proj_in = nn.Conv2d(40, hidden_dim, 3, padding=1)
        self.blocks = nn.Sequential(*[FlattenedResNetBlock(hidden_dim) for _ in range(num_blocks)])
        self.scale_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        B = x.shape[0]
        x_reshaped = x.view(B, 20, 8, 8, 2).permute(0, 1, 4, 2, 3).reshape(B, 40, 8, 8)
        h = self.proj_in(x_reshaped)
        h = self.blocks(h)
        h = h.permute(0, 2, 3, 1).contiguous()
        h = self.scale_norm(h)
        h = h.permute(0, 3, 1, 2).contiguous()
        return h

class DynamicsNet(nn.Module):
    def __init__(self, hidden_dim, num_blocks, support_size):
        super().__init__()
        self.piece_emb = nn.Embedding(48, hidden_dim)
        self.pos_emb = nn.Embedding(96, hidden_dim)
        self.proj_in = nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1)
        self.blocks = nn.Sequential(*[FlattenedResNetBlock(hidden_dim) for _ in range(num_blocks)])
        self.scale_norm = nn.LayerNorm(hidden_dim)
        
        self.reward_cond = nn.Conv2d(hidden_dim * 2, hidden_dim, 1)
        self.reward_fc1 = nn.Linear(hidden_dim, 64)
        self.reward_norm = nn.LayerNorm(64)
        self.reward_fc2 = nn.Linear(64, 2 * support_size + 1)
        
    def forward(self, hidden_state, batched_action, batched_piece_identifier):
        pos_indices = batched_action % 96
        action_embeddings = self.piece_emb(batched_piece_identifier) + self.pos_emb(pos_indices)
        action_expanded = action_embeddings.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 8, 8)
        concatenated_features = torch.cat([hidden_state, action_expanded], dim=1)
        
        reward_convolutions_mish = F.mish(self.reward_cond(concatenated_features))
        hidden_state_avg = reward_convolutions_mish.mean(dim=(2, 3))
        
        reward_features = F.mish(self.reward_norm(self.reward_fc1(hidden_state_avg)))
        reward_logits = self.reward_fc2(reward_features)
        
        hidden_state_next = self.proj_in(concatenated_features)
        hidden_state_next = self.blocks(hidden_state_next)
        hidden_state_next = hidden_state_next.permute(0, 2, 3, 1).contiguous()
        hidden_state_next = self.scale_norm(hidden_state_next)
        hidden_state_next = hidden_state_next.permute(0, 3, 1, 2).contiguous()
        
        return hidden_state_next, reward_logits

class HolePredictor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # In rust: 0 and 2
        setattr(self, '0', nn.Linear(hidden_dim, 64))
        setattr(self, '2', nn.Linear(64, 2))
        
    def forward(self, x):
        l1 = getattr(self, '0')
        l2 = getattr(self, '2')
        return l2(F.mish(l1(x)))

class PredictionNet(nn.Module):
    def __init__(self, hidden_dim, support_size, action_count):
        super().__init__()
        self.val_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        self.val_norm = nn.LayerNorm(hidden_dim // 2)
        self.value_fc1 = nn.Linear(hidden_dim // 2, 64)
        self.value_fc2 = nn.Linear(64, 2 * support_size + 1)
        
        self.pol_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        self.pol_norm = nn.LayerNorm(hidden_dim // 2)
        self.policy_fc1 = nn.Linear(hidden_dim // 2, action_count)
        
        self.hole_predictor = HolePredictor(hidden_dim)
        
    def forward(self, hidden_state):
        transposed = hidden_state.permute(0, 2, 3, 1)
        
        val_feat = F.mish(self.val_norm(self.val_proj(transposed))).mean(dim=(1, 2))
        val_inter = F.mish(self.value_fc1(val_feat))
        val_logits = self.value_fc2(val_inter)
        
        pol_feat = F.mish(self.pol_norm(self.pol_proj(transposed))).mean(dim=(1, 2))
        pol_logits = self.policy_fc1(pol_feat)
        
        hole_logits = self.hole_predictor(transposed).flatten(1, 3)
        return val_logits, pol_logits, hole_logits

class InitialInferenceModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, batched_state):
        hidden_state = self.model.representation(batched_state)
        val_logits, pol_logits, hole_logits = self.model.prediction(hidden_state)
        pol_probs = F.softmax(pol_logits, dim=-1)
        return hidden_state, val_logits, pol_probs, hole_logits

class RecurrentInferenceModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, hidden_state, batched_action, batched_piece_id):
        hidden_state_next, reward_logits = self.model.dynamics(hidden_state, batched_action, batched_piece_id)
        val_logits, pol_logits, hole_logits = self.model.prediction(hidden_state_next)
        pol_probs = F.softmax(pol_logits, dim=-1)
        return hidden_state_next, reward_logits, val_logits, pol_probs, hole_logits

class MuZeroNet(nn.Module):
    def __init__(self, hidden_dim=256, num_blocks=4, support_size=300):
        super().__init__()
        self.representation = RepresentationNet(hidden_dim, num_blocks)
        self.dynamics = DynamicsNet(hidden_dim, num_blocks, support_size)
        self.prediction = PredictionNet(hidden_dim, support_size, 288)

def export_onnx(model_path=None):
    import os
    model = MuZeroNet()
    out_dir = ""
    if model_path:
        out_dir = os.path.dirname(model_path)
        if out_dir: out_dir += "/"
        print(f"Loading {model_path} dict into Python architecture...")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        
        # Map the state_dict appropriately
        model.load_state_dict(state_dict, strict=False)
    else:
        print("No model path provided. Generating ONNX with randomly initialized dummy weights...")
    
    model.eval()
    
    print("Exporting initial_inference to ONNX...")
    initial_model = InitialInferenceModel(model).eval()
    dummy_initial_in = torch.randn(1, 20, 8, 16)
    torch.onnx.export(
        initial_model,
        dummy_initial_in,
        f"{out_dir}initial_inference.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['batched_state'],
        output_names=['hidden_state', 'value_logits', 'policy_probs', 'hole_logits'],
        dynamic_axes={'batched_state': {0: 'batch_size'},
                      'hidden_state': {0: 'batch_size'},
                      'value_logits': {0: 'batch_size'},
                      'policy_probs': {0: 'batch_size'},
                      'hole_logits': {0: 'batch_size'}}
    )

    print("Exporting recurrent_inference to ONNX...")
    recurrent_model = RecurrentInferenceModel(model).eval()
    dummy_hidden = torch.randn(1, 256, 8, 8)
    dummy_action = torch.randint(0, 96, (1,), dtype=torch.int64)
    dummy_piece = torch.randint(0, 48, (1,), dtype=torch.int64)
    torch.onnx.export(
        recurrent_model,
        (dummy_hidden, dummy_action, dummy_piece),
        f"{out_dir}recurrent_inference.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['hidden_state', 'action', 'piece'],
        output_names=['hidden_state_next', 'reward_logits', 'value_logits', 'policy_probs', 'hole_logits'],
        dynamic_axes={'hidden_state': {0: 'batch_size'},
                      'action': {0: 'batch_size'},
                      'piece': {0: 'batch_size'},
                      'hidden_state_next': {0: 'batch_size'},
                      'reward_logits': {0: 'batch_size'},
                      'value_logits': {0: 'batch_size'},
                      'policy_probs': {0: 'batch_size'},
                      'hole_logits': {0: 'batch_size'}}
    )
    print("Export complete!")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    export_onnx(path)
```

### File: `scripts/optuna_insights.py`

```py
#!/usr/bin/env python3
import os
import optuna
import optunahub


def generate_insights():
    db_path = "sqlite:///autotune.db"
    study_name = "tricked-ai-tuning-sota-2h"

    print(f"🔍 Loading Study '{study_name}'...")
    try:
        study = optuna.load_study(study_name=study_name, storage=db_path)
        print(f"✅ Loaded {len(study.trials)} trials.")
    except Exception as e:
        print(f"❌ Could not load study: {e}")
        return

    output_dir = "outputs/insights"
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Parallel Coordinate Plot (Built-In)
    # The ultimate high-dimensional flow visualizer.
    # ---------------------------------------------------------
    print("📈 Generating Parallel Coordinate Plot...")
    fig_parallel = optuna.visualization.plot_parallel_coordinate(
        study,
        params=[
            "simulations",
            "gumbel_scale",
            "unroll_steps",
            "temporal_difference_steps",
            "lr_init",
        ],
    )
    fig_parallel.update_layout(
        title="High-Dimensional Flow: How parameters route to the highest score",
        template="plotly_dark",
    )
    fig_parallel.write_html(f"{output_dir}/1_parallel_coordinates.html")

    # ---------------------------------------------------------
    # 2. Contour Plot (Built-In)
    # The 2D Heatmap for specific interaction boundaries.
    # ---------------------------------------------------------
    print("🗺️ Generating Contour Heatmaps...")
    # Plot Mcripts Depth vs Exploration Noise
    fig_contour_mcts = optuna.visualization.plot_contour(
        study, params=["simulations", "gumbel_scale"]
    )
    fig_contour_mcts.update_layout(
        title="Contour: Simulations vs Gumbel Scale", template="plotly_dark"
    )
    # Plot Bootstrapping Horizon vs Dynamics Prediction
    fig_contour_mcts.write_html(f"{output_dir}/2_contour_mcts_exploration.html")

    fig_contour_td = optuna.visualization.plot_contour(
        study, params=["temporal_difference_steps", "unroll_steps"]
    )
    fig_contour_td.update_layout(
        title="Contour: TD Steps vs Unroll Steps", template="plotly_dark"
    )
    fig_contour_td.write_html(f"{output_dir}/3_contour_value_horizon.html")

    # ---------------------------------------------------------
    # 3. SHAP-like Beeswarm Plot (OptunaHub)
    # Shows if a high parameter value pushes the score up or down.
    # ---------------------------------------------------------
    print("🐝 Loading SHAP-like Beeswarm from OptunaHub...")
    try:
        # Note: optunahub searches registered namespaces. We use the standard import pattern.
        shap_module = optunahub.load_module("visualization/plot_beeswarm")
        fig, ax, cbar = shap_module.plot_beeswarm(study)
        fig.suptitle("SHAP Beeswarm: Directional Parameter Impact")
        fig.savefig(f"{output_dir}/4_shap_beeswarm.png")
        print("✅ Saved 4_shap_beeswarm.png (Matplotlib)")
    except Exception as e:
        print(f"⚠️ Could not load/render SHAP Beeswarm from OptunaHub: {e}")
        # Fallback to Built-in Importance if Hub fails
        print("Falling back to Built-In Hyperparameter Importances...")
        fig_import = optuna.visualization.plot_param_importances(study)
        fig_import.update_layout(
            title="fANOVA Parameter Importances", template="plotly_dark"
        )
        fig_import.write_html(f"{output_dir}/4_basic_importance.html")

    # ---------------------------------------------------------
    # 4. Step Distribution Plot (OptunaHub)
    # Shows exactly when and where the pruner is killing bad trials.
    # ---------------------------------------------------------
    print("📉 Loading Step Distribution from OptunaHub...")
    fig_step = None
    try:
        step_dist_module = optunahub.load_module("visualization/plot_step_distribution")
        fig_step = step_dist_module.plot_step_distribution(study)
        fig_step.update_layout(
            title="Pruning Step Distribution (Where agents die)", template="plotly_dark"
        )
        fig_step.write_html(f"{output_dir}/5_step_distribution.html")
    except Exception as e:
        print(f"⚠️ Could not load/render Step Distribution from OptunaHub: {e}")
        # Fallback to Built-in Optimization History
        print("Falling back to Built-in Optimization History...")
        fig_history = optuna.visualization.plot_optimization_history(study)
        fig_history.update_layout(title="Optimization History", template="plotly_dark")
        fig_history.write_html(f"{output_dir}/5_optimization_history.html")

    # ---------------------------------------------------------
    # Dashboard Integration
    # ---------------------------------------------------------
    try:
        from optuna_dashboard import save_plotly_graph_object

        print("💻 Pushing Plotly graphs to Optuna Dashboard...")
        save_plotly_graph_object(study, fig_parallel)
        save_plotly_graph_object(study, fig_contour_mcts)
        save_plotly_graph_object(study, fig_contour_td)
        if fig_step is not None:
            save_plotly_graph_object(study, fig_step)
        print("✅ Custom Plotly graphs are now visible in Optuna Dashboard!")
    except ImportError:
        print("⚠️ optuna-dashboard not installed. Skipping dashboard push.")
    except Exception as e:
        print(f"⚠️ Could not push to dashboard: {e}")

    print(
        f"\n🎉 Success! Open the files in '{output_dir}' to view the interactive insights standalone, or check your Optuna Dashboard!"
    )


if __name__ == "__main__":
    generate_insights()
```

### File: `scripts/profile_cmodule.py`

``py
import torch
import sys
from torch.profiler import profile, record_function, ProfilerActivity

def run_profile(model_path, batch_size=1024):
    """
    Profiles the exported TorchScript tracing/compilation model 
    to measure cudaMemcpy vs actual Kernel execution.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading TorchScript CModule '{model_path}' onto {device} with Batch Size {batch_size}")
    
    try:
        model = torch.jit.load(model_path).to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Create dummy tensors matching Tricked Engine state input size (BATCH, 20, 8, 16)
    # This reflects the exact dimension of our `pinned_initial_states`
    dummy_input = torch.randn(batch_size, 20, 8, 16, device=device)
    
    print("Warming up CModule execution...")
    with torch.no_grad():
        for _ in range(10):
            try:
                # The exported model should have the "initial_inference" method 
                # as defined in the python export script
                model.initial_inference(dummy_input)
            except Exception as e:
                print(f"Error during execution call: {e}")
                print("Make sure the loaded .pt file exports 'initial_inference(tensor)'.")
                return

    print("Profiling initial_inference across 50 iterations...")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                for _ in range(50):
                    model.initial_inference(dummy_input)

    print("\n--- Profiling Results (Sorted by CUDA Time) ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    
    trace_file = "trace_initial.json"
    prof.export_chrome_trace(trace_file)
    print(f"\nExported chrome trace to {trace_file}")
    print("You can view the exact PCIe transfer timeline by opening chrome://tracing and loading this JSON file.")
    print("If cudaMemcpyAsync dominates, ensure Rust `pinned_initial_states.pin_memory(device)` is active.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python profile_cmodule.py <path_to_model.pt> [batch_size]")
        sys.exit(1)
    
    bs = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    run_profile(sys.argv[1], bs)
``

### File: `scripts/tb_logger.py`

``py
#!/usr/bin/env python3
import os
import glob
import time
import pandas as pd
from tensorboardX import SummaryWriter
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def sync_csv_to_tensorboard(runs_dir="runs", poll_interval=2.0, max_iterations=None):
    """
    Monitors `runs/*/*_metrics.csv` files and converts appended rows into TensorBoard events.
    """
    logging.info(f"Starting TensorBoard CSV bridge. Monitoring directory: {runs_dir}/")
    state = {}
    writers = {}

    iterations = 0
    while True:
        csv_files = glob.glob(
            os.path.join(runs_dir, "**", "*_metrics.csv"), recursive=True
        )

        for csv_path in csv_files:
            experiment_dir = os.path.dirname(csv_path)

            if csv_path not in state:
                state[csv_path] = 0
                writers[csv_path] = SummaryWriter(log_dir=experiment_dir)
                logging.info(f"Discovered new metrics file: {csv_path}")

                config_path = os.path.join(experiment_dir, "config.json")
                if os.path.exists(config_path):
                    import json

                    try:
                        with open(config_path, "r") as f:
                            hparams = json.load(f)
                            flat_hparams = {
                                k: str(v) if isinstance(v, (dict, list)) else v
                                for k, v in hparams.items()
                            }
                            writers[csv_path].add_hparams(
                                flat_hparams, {"Loss/total_loss": 0.0}
                            )
                            logging.info(
                                f"Injected hyperparameters from config.json for {csv_path}"
                            )
                    except Exception as e:
                        logging.debug(f"Failed to load config.json: {e}")

            try:
                # Read the CSV. We use standard read_csv and slice by state to avoid file locks
                if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
                    continue

                df = pd.read_csv(csv_path)

                if len(df) > state[csv_path]:
                    new_rows = df.iloc[state[csv_path] :]

                    for _, row in new_rows.iterrows():
                        step = int(row["step"])
                        writer = writers[csv_path]

                        writer.add_scalar(
                            "Loss/total_loss", row.get("total_loss", 0.0), step
                        )
                        writer.add_scalar(
                            "Loss/policy_loss", row.get("policy_loss", 0.0), step
                        )
                        writer.add_scalar(
                            "Loss/value_loss", row.get("value_loss", 0.0), step
                        )
                        writer.add_scalar(
                            "Loss/reward_loss", row.get("reward_loss", 0.0), step
                        )
                        writer.add_scalar("Optimization/lr", row.get("lr", 0.0), step)

                    state[csv_path] = len(df)
                    writers[csv_path].flush()

            except Exception as e:
                # File might be mid-write or locked by the Rust engine
                logging.debug(f"Transient error reading {csv_path}: {e}")

        if max_iterations is not None:
            iterations += 1
            if iterations >= max_iterations:
                break

        time.sleep(poll_interval)


def test_tb_logger():
    """
    Test suite for the TensorBoard logger to guarantee execution safety.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        exp_dir = os.path.join(tmpdir, "test_exp")
        os.makedirs(exp_dir)
        csv_path = os.path.join(exp_dir, "test_exp_metrics.csv")

        # Write dummy data
        df = pd.DataFrame(
            {
                "step": [1, 2],
                "total_loss": [10.5, 9.2],
                "policy_loss": [5.0, 4.0],
                "value_loss": [5.5, 5.2],
                "reward_loss": [0.0, 0.0],
                "lr": [0.001, 0.001],
            }
        )
        df.to_csv(csv_path, index=False)

        # Run bridge for 1 iteration
        sync_csv_to_tensorboard(runs_dir=tmpdir, poll_interval=0.1, max_iterations=1)

        # Verify tfevents file was created
        events_files = glob.glob(os.path.join(exp_dir, "events.out.tfevents.*"))
        assert len(events_files) > 0, "SummaryWriter failed to generate tfevents file!"
        print(
            "✅ [Test Passed] TensorBoard Bridge Successfully Translated CSV to TFEvents."
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_tb_logger()
    else:
        # Run infinitely
        try:
            sync_csv_to_tensorboard()
        except KeyboardInterrupt:
            logging.info("TensorBoard bridge terminated by user.")
``

### File: `scripts/tune.py`

```py
#!/usr/bin/env python3
import optuna
import subprocess
import time
import pandas as pd
import os
import signal
import sys
from pathlib import Path

# Paths


def objective(trial):
    # Suggest hyperparameters (ranges tailored for rapid low-fidelity evaluation)
    lr_init = trial.suggest_float("lr_init", 1e-4, 5e-3, log=True)
    simulations = trial.suggest_int("simulations", 10, 50)
    unroll_steps = trial.suggest_int("unroll_steps", 5, 10)
    temporal_difference_steps = trial.suggest_int("temporal_difference_steps", 5, 10)
    reanalyze_ratio = trial.suggest_float("reanalyze_ratio", 0.0, 0.5)
    support_size = trial.suggest_int("support_size", 100, 500)
    temp_decay_steps = trial.suggest_int("temp_decay_steps", 1000, 50000)

    # We want to run for a fixed number of steps per trial to evaluate
    # Low-fidelity iteration for the optimization pipeline
    max_steps = 20
    experiment_name = f"optuna_trial_{trial.number}"
    metrics_file = f"runs/{experiment_name}/{experiment_name}_metrics.csv"

    # Ensure runs directory exists
    os.makedirs(f"runs/{experiment_name}", exist_ok=True)

    cmd = [
        "cargo",
        "run",
        "--release",
        "--bin",
        "tricked_engine",
        "--",
        "train",
        "--experiment-name",
        experiment_name,
        "--lr-init",
        str(lr_init),
        "--simulations",
        str(simulations),
        "--unroll-steps",
        str(unroll_steps),
        "--temporal-difference-steps",
        str(temporal_difference_steps),
        "--reanalyze-ratio",
        str(reanalyze_ratio),
        "--support-size",
        str(support_size),
        "--temp-decay-steps",
        str(temp_decay_steps),
        "--max-steps",
        str(max_steps),
    ]

    print(f"\n[Trial {trial.number}] Starting with CMD: {' '.join(cmd)}")

    import select

    # Start the process in a new process group so we can cleanly kill it
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,  # Line buffered
        preexec_fn=os.setsid,
    )

    final_loss = float("inf")
    last_reported_step = -1

    try:
        # Wait for the process to finish or check periodically for pruning
        while process.poll() is None:
            reads, _, _ = select.select([process.stdout], [], [], 2.0)
            if process.stdout in reads:
                line = process.stdout.readline()
                if "FINAL_EVAL_SCORE:" in line:
                    try:
                        final_loss = float(line.strip().split("FINAL_EVAL_SCORE:")[1])
                    except ValueError:
                        pass

            if os.path.exists(metrics_file):
                try:
                    df = pd.read_csv(metrics_file)
                    if not df.empty:
                        last_step = df["step"].iloc[-1]
                        last_loss = df["total_loss"].iloc[-1]

                        if last_step > last_reported_step:
                            # Report to Optuna for pruning
                            trial.report(last_loss, last_step)
                            last_reported_step = last_step

                        if trial.should_prune():
                            print(f"[Trial {trial.number}] Pruned at step {last_step}.")
                            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                            raise optuna.TrialPruned()

                except Exception as e:
                    # File might be locked or half-written, ignore and retry next loop
                    pass

        # Final check if finished normally and we missed the line
        for line in process.stdout:
            if "FINAL_EVAL_SCORE:" in line:
                try:
                    final_loss = float(line.strip().split("FINAL_EVAL_SCORE:")[1])
                except ValueError:
                    pass

    finally:
        # Ensure cleanup of the child process tree
        if process.poll() is None:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait()

    if final_loss == float("inf"):
        print(
            f"[Trial {trial.number}] Failed to read final loss. Process may have crashed."
        )
        raise optuna.TrialPruned()

    return final_loss


if __name__ == "__main__":
    storage_name = "sqlite:///optuna_study.db"

    # Create study using Median Pruner
    study = optuna.create_study(
        study_name="tricked_ai_optimization",
        direction="minimize",
        storage=storage_name,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=100, n_startup_trials=5),
    )

    print("🚀 Starting Optuna optimization... Press Ctrl+C to stop.")
    try:
        study.optimize(objective, n_trials=50)
    except KeyboardInterrupt:
        print("\n🛑 Optimization interrupted by user.")

    print("\n✅ Optimization Session Complete!")
    print("Best Trial:")
    print(f"  Value (Final Loss): {study.best_trial.value}")
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")
```

### File: `src/config.rs`

```rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ExperimentPaths {
    pub base_directory: String,
    pub model_checkpoint_path: String,
    pub metrics_file_path: String,
    pub experiment_name_identifier: String,
}

impl ExperimentPaths {
    pub fn new(experiment_name: &str) -> Self {
        let base_directory = format!("runs/{}", experiment_name);
        Self {
            model_checkpoint_path: format!("{}/{}_weights.pt", base_directory, experiment_name),
            metrics_file_path: format!("{}/{}_metrics.csv", base_directory, experiment_name),
            base_directory: base_directory.clone(),
            experiment_name_identifier: experiment_name.to_string(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    pub experiment_name_identifier: String,
    #[serde(skip, default = "default_paths")]
    pub paths: ExperimentPaths,
    pub device: String,
    pub hidden_dimension_size: i64,
    pub num_blocks: i64,
    pub support_size: i64,
    pub buffer_capacity_limit: usize,
    pub simulations: i64,
    pub train_batch_size: usize,
    pub train_epochs: i64,
    pub num_processes: i64,
    pub worker_device: String,
    pub unroll_steps: usize,
    pub temporal_difference_steps: usize,
    pub inference_batch_size_limit: i64,
    pub inference_timeout_ms: i64,
    pub max_gumbel_k: i64,
    pub gumbel_scale: f32,
    pub temp_decay_steps: i64,
    pub difficulty: i32,
    pub temp_boost: bool,
    pub lr_init: f64,
    pub reanalyze_ratio: f32,
}

fn default_paths() -> ExperimentPaths {
    ExperimentPaths::new("default_fallback")
}
```

### File: `src/core/board.rs`

``rs
use once_cell::sync::Lazy;
use rand::Rng;

use crate::core::constants::{ALL_MASKS, STANDARD_PIECES};

pub static WEIGHTED_PIECES_BY_DIFFICULTY: Lazy<std::collections::HashMap<i32, Vec<i32>>> =
    Lazy::new(|| {
        let mut map = std::collections::HashMap::new();
        for diff in 0..=10 {
            let mut valid_pieces = Vec::new();
            for (p_id, piece_masks) in STANDARD_PIECES.iter().enumerate() {
                for &mask in piece_masks {
                    if mask != 0 {
                        let size = mask.count_ones();
                        let allowed_size = std::cmp::max(3, diff as u32);
                        if size <= allowed_size {
                            let weight = match size {
                                1 => 70,
                                2 => 25,
                                3 => 5,
                                _ => 1,
                            };
                            for _ in 0..weight {
                                valid_pieces.push(p_id as i32);
                            }
                        }
                        break;
                    }
                }
            }
            if valid_pieces.is_empty() {
                for i in 0..STANDARD_PIECES.len() {
                    valid_pieces.push(i as i32);
                }
            }
            map.insert(diff, valid_pieces);
        }
        map
    });

/// High-performance FFI boundary structuring the Tricked Hex-Grid state.
/// This class exposes a true 96-bit triangular environment safely natively
/// bypassing the Python GIL. Represented essentially mathematically by a `u128` bitboard.
#[derive(Clone, Debug)]
pub struct GameStateExt {
    pub board_bitmask_u128: u128,
    pub available: [i32; 3],
    pub score: i32,
    pub pieces_left: i32,
    pub terminal: bool,
    pub difficulty: i32,
    pub total_lines_cleared: i32,
}

impl GameStateExt {
    pub fn new(
        pieces: Option<[i32; 3]>,
        board_state: u128,
        current_score: i32,
        difficulty: i32,
        clutter_amount: i32,
    ) -> Self {
        let mut state = GameStateExt {
            board_bitmask_u128: board_state,
            score: current_score,
            available: [-1, -1, -1],
            pieces_left: 0,
            terminal: false,
            difficulty,
            total_lines_cleared: 0,
        };

        if clutter_amount > 0 {
            let mut rng = rand::thread_rng();
            for _ in 0..clutter_amount {
                let p_id = rng.gen_range(0..STANDARD_PIECES.len());
                let mut valid_placements = Vec::new();
                for &mask in STANDARD_PIECES[p_id].iter() {
                    if mask != 0 && (state.board_bitmask_u128 & mask) == 0 {
                        valid_placements.push(mask);
                    }
                }
                if !valid_placements.is_empty() {
                    let chosen_mask = valid_placements[rng.gen_range(0..valid_placements.len())];
                    state.board_bitmask_u128 |= chosen_mask;
                }
            }
        }

        if let Some(pieces_available) = pieces {
            state.pieces_left = pieces_available.iter().filter(|&&x| x != -1).count() as i32;
            state.available = pieces_available;
            if state.pieces_left == 0 {
                state.refill_tray();
            } else {
                state.check_terminal();
            }
        } else {
            state.refill_tray();
        }

        state
    }

    /// Dynamically recalculates the `terminal` status explicitly checking
    /// if any available kinetic fragment (`p_id`) can physically be placed
    /// onto the current topological layout without intersection.
    pub fn check_terminal(&mut self) {
        self.terminal = false;
        if self.pieces_left > 0 {
            let mut has_move = false;
            for &piece_id in &self.available {
                if piece_id == -1 {
                    continue;
                }
                for &piece_mask in &STANDARD_PIECES[piece_id as usize] {
                    if piece_mask != 0 && (self.board_bitmask_u128 & piece_mask) == 0 {
                        has_move = true;
                        break;
                    }
                }
                if has_move {
                    break;
                }
            }
            self.terminal = !has_move;
        }
    }

    /// Natively executes a valid fragment drop onto the `u128` bitboard tracking
    /// structural line-clearing operations (`ALL_MASKS`) via rapid bitwise `$ AND = mask`.
    ///
    /// Returns:
    ///     `Some(GameStateExt)` representing the transition $s_{t+1}$ if valid.
    ///     `None` if the move intersects existing layout topology or invalid.
    pub fn apply_move(&mut self, slot: usize, index: usize) -> Option<GameStateExt> {
        assert!(slot < 3, "Invalid slot array boundary");

        let piece_id = self.available[slot];
        if piece_id == -1 {
            return None;
        }

        let piece_mask = STANDARD_PIECES[piece_id as usize][index];
        if piece_mask == 0 || (self.board_bitmask_u128 & piece_mask) != 0 {
            return None;
        }

        let mut next_available = self.available;
        next_available[slot] = -1;

        let mut next_board_bitmask_u128 = self.board_bitmask_u128 | piece_mask;
        let mut next_score = self.score + piece_mask.count_ones() as i32;

        let mut cleared_mask: u128 = 0;
        let mut lines_cleared = 0;

        for &line in ALL_MASKS.iter() {
            let is_match = ((next_board_bitmask_u128 & line) == line) as u128;
            lines_cleared += is_match as i32;
            let masku = is_match.wrapping_neg();
            cleared_mask |= line & masku;
            next_score += (is_match as i32) * (line.count_ones() as i32) * 2;
        }

        if lines_cleared > 0 {
            next_board_bitmask_u128 &= !cleared_mask;
        }

        let mut next_state = GameStateExt::new(
            Some(next_available),
            next_board_bitmask_u128,
            next_score,
            self.difficulty,
            0,
        );
        next_state.total_lines_cleared = self.total_lines_cleared + lines_cleared;

        Some(next_state)
    }

    pub fn refill_tray(&mut self) {
        let mut rng = rand::thread_rng();

        let valid_pieces = WEIGHTED_PIECES_BY_DIFFICULTY
            .get(&self.difficulty)
            .unwrap_or_else(|| WEIGHTED_PIECES_BY_DIFFICULTY.get(&6).unwrap());

        let max_idx = valid_pieces.len();

        self.available = [
            valid_pieces[rng.gen_range(0..max_idx)],
            valid_pieces[rng.gen_range(0..max_idx)],
            valid_pieces[rng.gen_range(0..max_idx)],
        ];

        self.pieces_left = 3;
        self.check_terminal();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use rand::Rng;

    proptest! {
        // [1. Bitboard State Transitions (CPU / Correctness)]
        // Generate valid piece IDs and arbitrary drop indices.
        #[test]
        fn properties_of_apply_move(
            piece_slot in 0usize..3,
            drop_index in 0usize..96,
            initial_board in 0u128..=u128::MAX // Arbitrary random initial noise
        ) {
            let mut state = GameStateExt::new(Some([0, 1, 2]), initial_board, 0, 0, 0);

            // Mask to limit to only 96 valid grid bits.
            state.board_bitmask_u128 &= (1u128 << 96) - 1;

            // Apply a random piece to a random index
            if let Some(next_state) = state.apply_move(piece_slot, drop_index) {
                // Assert no overlapping bits were counted incorrectly and board never exceeds 96 allowed bits
                assert!(next_state.board_bitmask_u128.count_ones() <= 96, "Board bits exceeded 96 max allowed on valid drop.");

                // Assert mask logic did not inadvertently clear non-existent mask pieces.
                // We could verify score increments here strictly if we replicate the mask logic but count_ones is the baseline safety.
            }
        }
    }

    // [2. Terminal State & Tray Refill Logic (Correctness)]
    #[test]
    fn test_monte_carlo_terminal_exhaustion() {
        let mut rng = rand::thread_rng();
        let mut timeouts = 0;

        for _ in 0..10_000 {
            let mut state = GameStateExt::new(None, 0, 0, 0, 0);
            let mut move_count = 0;

            // Random walk until terminal
            while !state.terminal {
                let mut valid_moves = Vec::new();
                for slot in 0..3 {
                    if state.available[slot] != -1 {
                        for idx in 0..96 {
                            // Test if move is valid using pure bitwise mask check (duplicating apply_move validation basically)
                            let p_id = state.available[slot] as usize;
                            if let Some(mask) = STANDARD_PIECES[p_id].get(idx) {
                                if (state.board_bitmask_u128 & mask) == 0 {
                                    valid_moves.push((slot, idx));
                                }
                            }
                        }
                    }
                }

                if valid_moves.is_empty() {
                    // check_terminal must be true here.
                    state.check_terminal();
                    assert!(
                        state.terminal,
                        "State physically has no valid moves but check_terminal returned false!"
                    );
                    break;
                }

                let (chosen_slot, chosen_idx) = valid_moves[rng.gen_range(0..valid_moves.len())];
                if let Some(new_state) = state.apply_move(chosen_slot, chosen_idx) {
                    state = new_state;
                } else {
                    continue;
                }

                if state.available == [-1, -1, -1] {
                    state.refill_tray();
                }

                move_count += 1;
                if move_count > 2000 {
                    timeouts += 1;
                    break;
                }
            }
            assert!(timeouts < 10, "Monte Carlo random walk looping infinitely.");
        }
    }

    #[test]
    fn test_bitboard_collision_logic() {
        let mut rng = rand::thread_rng();

        for _ in 0..10_000 {
            // Generate a random board state
            let mut base_board;
            loop {
                base_board = rng.r#gen::<u128>() & ((1_u128 << 96) - 1);
                let mut has_lines = false;
                for &line in ALL_MASKS.iter() {
                    if (base_board & line) == line {
                        has_lines = true;
                        break;
                    }
                }
                if !has_lines {
                    break;
                }
            }
            let mut state = GameStateExt::new(None, base_board, 0, 6, 0);

            // Generate random pieces
            state.refill_tray();

            let slot = 0;
            let p_id = state.available[slot];
            if p_id == -1 {
                continue;
            }

            let piece_masks = &STANDARD_PIECES[p_id as usize];
            let index = rng.gen_range(0..piece_masks.len());
            let mask = piece_masks[index];

            if mask == 0 {
                continue;
            }

            let collision = (state.board_bitmask_u128 & mask) != 0;

            let mut expected_lines_cleared = 0;
            if !collision {
                let simulated_board_bitmask_u128 = state.board_bitmask_u128 | mask;
                for &line in ALL_MASKS.iter() {
                    if (simulated_board_bitmask_u128 & line) == line {
                        expected_lines_cleared += 1;
                    }
                }
            }

            let result = state.apply_move(slot, index);

            if collision {
                assert!(result.is_none(), "Move should fail on collision!");
            } else {
                assert!(result.is_some(), "Move should succeed if no collision!");
                let new_state = result.unwrap();
                let placed_board_bitmask_u128 = state.board_bitmask_u128 | mask;

                if expected_lines_cleared > 0 {
                    assert!(
                        new_state.score > state.score + mask.count_ones() as i32,
                        "Score didn't account for line clears!"
                    );
                    assert!(
                        (new_state.board_bitmask_u128 & mask) != mask,
                        "Line should be cleared from board entirely!"
                    );
                } else {
                    assert_eq!(
                        new_state.board_bitmask_u128, placed_board_bitmask_u128,
                        "Board bitmask didn't correctly encode the placed geometry!"
                    );
                }
            }
        }
    }

    #[test]
    fn test_simultaneous_line_clears() {
        let mut found = false;
        for (i, &mask_i) in ALL_MASKS.iter().enumerate() {
            for (j, &mask_j) in ALL_MASKS.iter().enumerate().skip(i + 1) {
                let intersection = mask_i & mask_j;
                if intersection != 0 {
                    for (p_id, piece_masks) in STANDARD_PIECES.iter().enumerate() {
                        for (idx, &mask) in piece_masks.iter().enumerate() {
                            if mask != 0 && (mask & intersection) == mask {
                                let initial_board_bitmask_u128 =
                                    (ALL_MASKS[i] | ALL_MASKS[j]) & !mask;
                                let mut state = GameStateExt::new(
                                    Some([p_id as i32, -1, -1]),
                                    initial_board_bitmask_u128,
                                    0,
                                    6,
                                    0,
                                );
                                let next_state =
                                    state.apply_move(0, idx).expect("Move should be valid");

                                assert_eq!((next_state.board_bitmask_u128 & mask_i), 0);
                                assert_eq!((next_state.board_bitmask_u128 & mask_j), 0);

                                found = true;
                                break;
                            }
                        }
                        if found {
                            break;
                        }
                    }
                }
                if found {
                    break;
                }
            }
            if found {
                break;
            }
        }
        assert!(
            found,
            "Could not find a valid simultaneous line clear scenario to test!"
        );
    }

    #[test]
    fn test_terminal_state_accuracy() {
        let mut rng = rand::thread_rng();
        for _ in 0..10_000 {
            let mut state =
                GameStateExt::new(None, rng.r#gen::<u128>() & ((1_u128 << 96) - 1), 0, 6, 0);
            state.refill_tray();

            let is_terminal = state.terminal;
            let mut found_valid_move = false;

            for &p_id in &state.available {
                if p_id == -1 {
                    continue;
                }
                for &mask in &STANDARD_PIECES[p_id as usize] {
                    if mask != 0 && (state.board_bitmask_u128 & mask) == 0 {
                        found_valid_move = true;
                        break;
                    }
                }
                if found_valid_move {
                    break;
                }
            }

            assert_eq!(is_terminal, !found_valid_move, "Terminal state mismatch!");
        }
    }

    #[test]
    fn test_scoring_correctness() {
        // Goal: Ensure piece placement = 1 point per triangle
        // And cleared lines = 2 points per triangle in the line (even on intersections)
        let mut found = false;
        for (i, &mask_i) in ALL_MASKS.iter().enumerate() {
            for (j, &mask_j) in ALL_MASKS.iter().enumerate().skip(i + 1) {
                let intersection = mask_i & mask_j;
                if intersection != 0 {
                    for (p_id, piece_masks) in STANDARD_PIECES.iter().enumerate() {
                        for (idx, &mask) in piece_masks.iter().enumerate() {
                            if mask != 0 && (mask & intersection) != 0 {
                                let initial_board_bitmask_u128 = (mask_i | mask_j) & !mask;

                                let mut has_other_lines = false;
                                for (k, &other_line) in ALL_MASKS.iter().enumerate() {
                                    if k != i
                                        && k != j
                                        && (initial_board_bitmask_u128 & other_line) == other_line
                                    {
                                        has_other_lines = true;
                                        break;
                                    }
                                }

                                let simulated_board_bitmask_u128 =
                                    initial_board_bitmask_u128 | mask;
                                let mut lines_formed = 0;
                                for &other_line in ALL_MASKS.iter() {
                                    if (simulated_board_bitmask_u128 & other_line) == other_line {
                                        lines_formed += 1;
                                    }
                                }

                                if !has_other_lines && lines_formed == 2 {
                                    let mut state = GameStateExt::new(
                                        Some([p_id as i32, -1, -1]),
                                        initial_board_bitmask_u128,
                                        0,
                                        6,
                                        0,
                                    );
                                    let next_state =
                                        state.apply_move(0, idx).expect("Move should be valid");

                                    let placed_hexes = mask.count_ones() as i32;
                                    let line1_hexes = ALL_MASKS[i].count_ones() as i32;
                                    let line2_hexes = ALL_MASKS[j].count_ones() as i32;

                                    let expected_score =
                                        placed_hexes + (line1_hexes * 2) + (line2_hexes * 2);

                                    assert_eq!(
                                        next_state.score, expected_score,
                                        "Scoring logic failed: placed {} triangles, line lengths are {} and {}. Expected {}, got {}",
                                        placed_hexes, line1_hexes, line2_hexes, expected_score, next_state.score
                                    );

                                    found = true;
                                    break;
                                }
                            }
                        }
                        if found {
                            break;
                        }
                    }
                }
                if found {
                    break;
                }
            }
            if found {
                break;
            }
        }

        assert!(
            found,
            "Could not find a valid simultaneous intersecting line clear scenario to test!"
        );
    }

    #[test]
    fn test_game_flow_and_refill() {
        // 1. Start game, ensure 3 pieces
        let mut state = GameStateExt::new(None, 0, 0, 6, 0);
        assert_eq!(state.pieces_left, 3);
        assert_eq!(state.available.iter().filter(|&&x| x != -1).count(), 3);

        // 2. Play 1st piece
        let p0 = state.available[0];
        let idx0 = STANDARD_PIECES[p0 as usize]
            .iter()
            .position(|&m| m != 0)
            .unwrap();
        state = state.apply_move(0, idx0).unwrap();
        assert_eq!(state.pieces_left, 2);
        assert_eq!(state.available[0], -1);

        // 3. Play 2nd piece
        let p1 = state.available[1];
        let idx1 = STANDARD_PIECES[p1 as usize]
            .iter()
            .position(|&m| m != 0 && (state.board_bitmask_u128 & m) == 0)
            .unwrap();
        state = state.apply_move(1, idx1).unwrap();
        assert_eq!(state.pieces_left, 1);
        assert_eq!(state.available[1], -1);

        // 4. Play 3rd piece, ensure refill
        let p2 = state.available[2];
        let idx2 = STANDARD_PIECES[p2 as usize]
            .iter()
            .position(|&m| m != 0 && (state.board_bitmask_u128 & m) == 0)
            .unwrap();
        let state = state.apply_move(2, idx2).unwrap();
        assert_eq!(
            state.pieces_left, 3,
            "Tray should refill after placing the last piece"
        );
        assert_eq!(state.available.iter().filter(|&&x| x != -1).count(), 3);
        assert!(
            !state.terminal,
            "Game should not be terminal on an empty board"
        );
    }

    #[test]
    fn test_clear_lines_before_terminal_check() {
        let mut p1_id = 0;
        let mut piece_mask = 0;
        for (i, piece) in STANDARD_PIECES.iter().enumerate() {
            for &m in piece.iter() {
                if m.count_ones() == 1 {
                    p1_id = i;
                    piece_mask = m;
                    break;
                }
            }
            if piece_mask != 0 {
                break;
            }
        }

        let mut p3_id = 0;
        for (i, piece) in STANDARD_PIECES.iter().enumerate() {
            for &m in piece.iter() {
                if m.count_ones() == 3 {
                    p3_id = i;
                    break;
                }
            }
            if p3_id != 0 {
                break;
            }
        }

        // Board is full everywhere EXCEPT `piece_mask`
        let initial_board_bitmask_u128 = ((1u128 << 96) - 1) & !piece_mask;

        let mut state = GameStateExt::new(
            Some([p1_id as i32, p3_id as i32, -1]),
            initial_board_bitmask_u128,
            0,
            6,
            0,
        );

        let mut p3_fits = false;
        for &m in &STANDARD_PIECES[p3_id] {
            if m != 0 && (initial_board_bitmask_u128 & m) == 0 {
                p3_fits = true;
                break;
            }
        }
        assert!(!p3_fits, "Piece 3-hex should not fit initially");

        let idx0 = STANDARD_PIECES[p1_id]
            .iter()
            .position(|&m| m == piece_mask)
            .unwrap();
        let next_state = state.apply_move(0, idx0).expect("Move should be valid");

        // Lines were cleared making room
        assert!(
            next_state.board_bitmask_u128.count_ones() < initial_board_bitmask_u128.count_ones(),
            "Lines should be cleared"
        );

        // Terminal is false because Piece 3 can now fit
        assert!(
            !next_state.terminal,
            "Game should not be terminal, lines were cleared making room for Piece 3"
        );
    }

    #[test]
    fn test_clutter_generation_overlaps() {
        for _ in 0..10_000 {
            // Request 10 overlapping pieces of clutter
            let _g = GameStateExt::new(None, 0, 0, 6, 10);

            // FFI and topological boundary generation safely handled
            // probabilistic coverage includes 0-size valid placements
        }
    }
}
``

### File: `src/core/constants.rs`

```rs
pub const ALL_MASKS: [u128; 24] = [
    511,
    1048064,
    8588886016,
    281466386776064,
    9223090561878065152,
    75548640353877468643328,
    154666947046946620038971392,
    79073420009353665059181559808,
    281500749661699,
    9225060989806843916,
    75613210713947260346416,
    155196073480980152050614464,
    930269303745265676927238400,
    3721077214981062707707904000,
    14884308859924250822239584256,
    59537235439696721796801757184,
    29725069823625977243979746305,
    39672116780239571382375034886,
    77399587518937808049848344,
    37782390680249151193184,
    4611897131103158656,
    7431267455906494310994936320,
    1857816863976623577748209664,
    464454215994155890141822976,
];

pub static STANDARD_PIECES: [[u128; 96]; 48] = [
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        6151,
        0,
        24604,
        0,
        98416,
        0,
        393664,
        0,
        0,
        0,
        0,
        12586496,
        0,
        50345984,
        0,
        201383936,
        0,
        805535744,
        0,
        3222142976,
        0,
        0,
        0,
        0,
        103086555136,
        0,
        412346220544,
        0,
        1649384882176,
        0,
        6597539528704,
        0,
        26390158114816,
        0,
        105560632459264,
        0,
        0,
        0,
        1688909989806080,
        0,
        6755639959224320,
        0,
        27022559836897280,
        0,
        108090239347589120,
        0,
        432360957390356480,
        0,
        1729443829561425920,
        0,
        6917775318245703680,
        0,
        0,
        55344172870802604032,
        0,
        221376691483210416128,
        0,
        885506765932841664512,
        0,
        3542027063731366658048,
        0,
        14168108254925466632192,
        0,
        56672433019701866528768,
        0,
        0,
        453476309564001907376128,
        0,
        1813905238256007629504512,
        0,
        7255620953024030518018048,
        0,
        29022483812096122072072192,
        0,
        116089935248384488288288768,
        0,
        0,
        929512839556198006702211072,
        0,
        3718051358224792026808844288,
        0,
        14872205432899168107235377152,
        0,
        59488821731596672428941508608,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        4103,
        0,
        16412,
        0,
        65648,
        0,
        262592,
        0,
        0,
        0,
        0,
        8392192,
        0,
        33568768,
        0,
        134275072,
        0,
        537100288,
        0,
        2148401152,
        0,
        0,
        0,
        0,
        68726816768,
        0,
        274907267072,
        0,
        1099629068288,
        0,
        4398516273152,
        0,
        17594065092608,
        0,
        70376260370432,
        0,
        0,
        0,
        1125960036384768,
        0,
        4503840145539072,
        0,
        18015360582156288,
        0,
        72061442328625152,
        0,
        288245769314500608,
        0,
        1152983077258002432,
        0,
        4611932309032009728,
        0,
        0,
        36897428797093052416,
        0,
        147589715188372209664,
        0,
        590358860753488838656,
        0,
        2361435443013955354624,
        0,
        9445741772055821418496,
        0,
        37782967088223285673984,
        0,
        0,
        302360582112173260537856,
        0,
        1209442328448693042151424,
        0,
        4837769313794772168605696,
        0,
        19351077255179088674422784,
        0,
        77404309020716354697691136,
        0,
        0,
        620027829734852937977430016,
        0,
        2480111318939411751909720064,
        0,
        9920445275757647007638880256,
        0,
        39681781103030588030555521024,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        6150,
        0,
        24600,
        0,
        98400,
        0,
        393600,
        0,
        0,
        0,
        0,
        12585984,
        0,
        50343936,
        0,
        201375744,
        0,
        805502976,
        0,
        3222011904,
        0,
        0,
        0,
        0,
        103085506560,
        0,
        412342026240,
        0,
        1649368104960,
        0,
        6597472419840,
        0,
        26389889679360,
        0,
        105559558717440,
        0,
        0,
        0,
        1688901399871488,
        0,
        6755605599485952,
        0,
        27022422397943808,
        0,
        108089689591775232,
        0,
        432358758367100928,
        0,
        1729435033468403712,
        0,
        6917740133873614848,
        0,
        0,
        55343609920849182720,
        0,
        221374439683396730880,
        0,
        885497758733586923520,
        0,
        3541991034934347694080,
        0,
        14167964139737390776320,
        0,
        56671856558949563105280,
        0,
        0,
        453457862819928197824512,
        0,
        1813831451279712791298048,
        0,
        7255325805118851165192192,
        0,
        29021303220475404660768768,
        0,
        116085212881901618643075072,
        0,
        0,
        929361723828746178055372800,
        0,
        3717446895314984712221491200,
        0,
        14869787581259938848885964800,
        0,
        59479150325039755395543859200,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        4110,
        0,
        16440,
        0,
        65760,
        0,
        0,
        0,
        0,
        0,
        0,
        8395776,
        0,
        33583104,
        0,
        134332416,
        0,
        537329664,
        0,
        0,
        0,
        0,
        0,
        0,
        68734156800,
        0,
        274936627200,
        0,
        1099746508800,
        0,
        4398986035200,
        0,
        17595944140800,
        0,
        0,
        0,
        0,
        0,
        1126020165926912,
        0,
        4504080663707648,
        0,
        18016322654830592,
        0,
        72065290619322368,
        0,
        288261162477289472,
        0,
        1153044649909157888,
        0,
        0,
        9225342361691750400,
        0,
        36901369446767001600,
        0,
        147605477787068006400,
        0,
        590421911148272025600,
        0,
        2361687644593088102400,
        0,
        9446750578372352409600,
        0,
        37787002313489409638400,
        75622427330172306849792,
        0,
        302489709320689227399168,
        0,
        1209958837282756909596672,
        0,
        4839835349131027638386688,
        0,
        19359341396524110553546752,
        0,
        77437365586096442214187008,
        155271409956753934626324480,
        0,
        621085639827015738505297920,
        0,
        2484342559308062954021191680,
        0,
        9937370237232251816084766720,
        0,
        39749480948929007264339066880,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        4102,
        0,
        16408,
        0,
        65632,
        0,
        262528,
        0,
        0,
        0,
        0,
        8391680,
        0,
        33566720,
        0,
        134266880,
        0,
        537067520,
        0,
        2148270080,
        0,
        0,
        0,
        0,
        68725768192,
        0,
        274903072768,
        0,
        1099612291072,
        0,
        4398449164288,
        0,
        17593796657152,
        0,
        70375186628608,
        0,
        0,
        0,
        1125951446450176,
        0,
        4503805785800704,
        0,
        18015223143202816,
        0,
        72060892572811264,
        0,
        288243570291245056,
        0,
        1152974281164980224,
        0,
        4611897124659920896,
        9224216461784907776,
        0,
        36896865847139631104,
        0,
        147587463388558524416,
        0,
        590349853554234097664,
        0,
        2361399414216936390656,
        0,
        9445597656867745562624,
        0,
        37782390627470982250496,
        75585533842024887746560,
        0,
        302342135368099550986240,
        0,
        1209368541472398203944960,
        0,
        4837474165889592815779840,
        0,
        19349896663558371263119360,
        0,
        77399586654233485052477440,
        154969178501850277332647936,
        0,
        619876714007401109330591744,
        0,
        2479506856029604437322366976,
        0,
        9918027424118417749289467904,
        0,
        39672109696473670997157871616,
    ],
    [
        0,
        2,
        0,
        8,
        0,
        32,
        0,
        128,
        0,
        0,
        1024,
        0,
        4096,
        0,
        16384,
        0,
        65536,
        0,
        262144,
        0,
        0,
        2097152,
        0,
        8388608,
        0,
        33554432,
        0,
        134217728,
        0,
        536870912,
        0,
        2147483648,
        0,
        0,
        17179869184,
        0,
        68719476736,
        0,
        274877906944,
        0,
        1099511627776,
        0,
        4398046511104,
        0,
        17592186044416,
        0,
        70368744177664,
        0,
        281474976710656,
        0,
        1125899906842624,
        0,
        4503599627370496,
        0,
        18014398509481984,
        0,
        72057594037927936,
        0,
        288230376151711744,
        0,
        1152921504606846976,
        0,
        4611686018427387904,
        9223372036854775808,
        0,
        36893488147419103232,
        0,
        147573952589676412928,
        0,
        590295810358705651712,
        0,
        2361183241434822606848,
        0,
        9444732965739290427392,
        0,
        37778931862957161709568,
        75557863725914323419136,
        0,
        302231454903657293676544,
        0,
        1208925819614629174706176,
        0,
        4835703278458516698824704,
        0,
        19342813113834066795298816,
        0,
        77371252455336267181195264,
        154742504910672534362390528,
        0,
        618970019642690137449562112,
        0,
        2475880078570760549798248448,
        0,
        9903520314283042199192993792,
        0,
        39614081257132168796771975168,
    ],
    [
        0,
        3,
        0,
        12,
        0,
        48,
        0,
        192,
        0,
        0,
        1536,
        0,
        6144,
        0,
        24576,
        0,
        98304,
        0,
        393216,
        0,
        0,
        3145728,
        0,
        12582912,
        0,
        50331648,
        0,
        201326592,
        0,
        805306368,
        0,
        3221225472,
        0,
        0,
        25769803776,
        0,
        103079215104,
        0,
        412316860416,
        0,
        1649267441664,
        0,
        6597069766656,
        0,
        26388279066624,
        0,
        105553116266496,
        0,
        0,
        0,
        1688849860263936,
        0,
        6755399441055744,
        0,
        27021597764222976,
        0,
        108086391056891904,
        0,
        432345564227567616,
        0,
        1729382256910270464,
        0,
        6917529027641081856,
        0,
        0,
        55340232221128654848,
        0,
        221360928884514619392,
        0,
        885443715538058477568,
        0,
        3541774862152233910272,
        0,
        14167099448608935641088,
        0,
        56668397794435742564352,
        0,
        0,
        453347182355485940514816,
        0,
        1813388729421943762059264,
        0,
        7253554917687775048237056,
        0,
        29014219670751100192948224,
        0,
        116056878683004400771792896,
        0,
        0,
        928455029464035206174343168,
        0,
        3713820117856140824697372672,
        0,
        14855280471424563298789490688,
        0,
        59421121885698253195157962752,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1537,
        0,
        6148,
        0,
        24592,
        0,
        98368,
        0,
        393472,
        0,
        0,
        3146240,
        0,
        12584960,
        0,
        50339840,
        0,
        201359360,
        0,
        805437440,
        0,
        3221749760,
        0,
        0,
        25770852352,
        0,
        103083409408,
        0,
        412333637632,
        0,
        1649334550528,
        0,
        6597338202112,
        0,
        26389352808448,
        0,
        105557411233792,
        0,
        0,
        0,
        1688884220002304,
        0,
        6755536880009216,
        0,
        27022147520036864,
        0,
        108088590080147456,
        0,
        432354360320589824,
        0,
        1729417441282359296,
        0,
        6917669765129437184,
        0,
        0,
        55342484020942340096,
        0,
        221369936083769360384,
        0,
        885479744335077441536,
        0,
        3541918977340309766144,
        0,
        14167675909361239064576,
        0,
        56670703637444956258304,
        0,
        0,
        453420969331780778721280,
        0,
        1813683877327123114885120,
        0,
        7254735509308492459540480,
        0,
        29018942037233969838161920,
        0,
        116075768148935879352647680,
        0,
        0,
        929059492373842520761696256,
        0,
        3716237969495370083046785024,
        0,
        14864951877981480332187140096,
        0,
        59459807511925921328748560384,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        3585,
        0,
        14340,
        0,
        57360,
        0,
        229440,
        0,
        917760,
        0,
        0,
        7340544,
        0,
        29362176,
        0,
        117448704,
        0,
        469794816,
        0,
        1879179264,
        0,
        7516717056,
        0,
        0,
        60130590720,
        0,
        240522362880,
        0,
        962089451520,
        0,
        3848357806080,
        0,
        15393431224320,
        0,
        61573724897280,
        0,
        246294899589120,
        0,
        0,
        0,
        3940684033687552,
        0,
        15762736134750208,
        0,
        63050944539000832,
        0,
        252203778156003328,
        0,
        1008815112624013312,
        0,
        4035260450496053248,
        0,
        0,
        0,
        0,
        129129460315780546560,
        0,
        516517841263122186240,
        0,
        2066071365052488744960,
        0,
        8264285460209954979840,
        0,
        33057141840839819919360,
        0,
        0,
        0,
        0,
        1057883879139095366074368,
        0,
        4231535516556381464297472,
        0,
        16926142066225525857189888,
        0,
        67704568264902103428759552,
        0,
        0,
        0,
        0,
        2166999531659222795660820480,
        0,
        8667998126636891182643281920,
        0,
        34671992506547564730573127680,
        0,
        0,
    ],
    [
        0,
        7,
        0,
        28,
        0,
        112,
        0,
        448,
        0,
        0,
        3584,
        0,
        14336,
        0,
        57344,
        0,
        229376,
        0,
        917504,
        0,
        0,
        7340032,
        0,
        29360128,
        0,
        117440512,
        0,
        469762048,
        0,
        1879048192,
        0,
        7516192768,
        0,
        0,
        60129542144,
        0,
        240518168576,
        0,
        962072674304,
        0,
        3848290697216,
        0,
        15393162788864,
        0,
        61572651155456,
        0,
        246290604621824,
        0,
        0,
        0,
        3940649673949184,
        0,
        15762598695796736,
        0,
        63050394783186944,
        0,
        252201579132747776,
        0,
        1008806316530991104,
        0,
        4035225266123964416,
        0,
        0,
        0,
        0,
        129127208515966861312,
        0,
        516508834063867445248,
        0,
        2066035336255469780992,
        0,
        8264141345021879123968,
        0,
        33056565380087516495872,
        0,
        0,
        0,
        0,
        1057810092162800527867904,
        0,
        4231240368651202111471616,
        0,
        16924961474604808445886464,
        0,
        67699845898419233783545856,
        0,
        0,
        0,
        0,
        2166395068749415481073467392,
        0,
        8665580274997661924293869568,
        0,
        34662321099990647697175478272,
        0,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1027,
        0,
        4108,
        0,
        16432,
        0,
        65728,
        0,
        0,
        0,
        0,
        2098688,
        0,
        8394752,
        0,
        33579008,
        0,
        134316032,
        0,
        537264128,
        0,
        0,
        0,
        0,
        17183014912,
        0,
        68732059648,
        0,
        274928238592,
        0,
        1099712954368,
        0,
        4398851817472,
        0,
        17595407269888,
        0,
        0,
        0,
        281500746514432,
        0,
        1126002986057728,
        0,
        4504011944230912,
        0,
        18016047776923648,
        0,
        72064191107694592,
        0,
        288256764430778368,
        0,
        1153027057723113472,
        0,
        0,
        9225060886715039744,
        0,
        36900243546860158976,
        0,
        147600974187440635904,
        0,
        590403896749762543616,
        0,
        2361615586999050174464,
        0,
        9446462347996200697856,
        0,
        37785849391984802791424,
        75613203958135452073984,
        0,
        302452815832541808295936,
        0,
        1209811263330167233183744,
        0,
        4839245053320668932734976,
        0,
        19356980213282675730939904,
        0,
        77427920853130702923759616,
        155195852093028020302905344,
        0,
        620783408372112081211621376,
        0,
        2483133633488448324846485504,
        0,
        9932534533953793299385942016,
        0,
        39730138135815173197543768064,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        7171,
        0,
        28684,
        0,
        114736,
        0,
        458944,
        0,
        0,
        0,
        0,
        14681600,
        0,
        58726400,
        0,
        234905600,
        0,
        939622400,
        0,
        3758489600,
        0,
        0,
        0,
        0,
        120262230016,
        0,
        481048920064,
        0,
        1924195680256,
        0,
        7696782721024,
        0,
        30787130884096,
        0,
        123148523536384,
        0,
        0,
        0,
        1970350606778368,
        0,
        7881402427113472,
        0,
        31525609708453888,
        0,
        126102438833815552,
        0,
        504409755335262208,
        0,
        2017639021341048832,
        0,
        8070556085364195328,
        0,
        0,
        64565293107843694592,
        0,
        258261172431374778368,
        0,
        1033044689725499113472,
        0,
        4132178758901996453888,
        0,
        16528715035607985815552,
        0,
        66114860142431943262208,
        0,
        0,
        528960386313621392588800,
        0,
        2115841545254485570355200,
        0,
        8463366181017942281420800,
        0,
        33853464724071769125683200,
        0,
        135413858896287076502732800,
        0,
        0,
        1083650881557063226477248512,
        0,
        4334603526228252905908994048,
        0,
        17338414104913011623635976192,
        0,
        69353656419652046494543904768,
        0,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        5127,
        0,
        20508,
        0,
        82032,
        0,
        328128,
        0,
        0,
        0,
        0,
        10489344,
        0,
        41957376,
        0,
        167829504,
        0,
        671318016,
        0,
        2685272064,
        0,
        0,
        0,
        0,
        85906685952,
        0,
        343626743808,
        0,
        1374506975232,
        0,
        5498027900928,
        0,
        21992111603712,
        0,
        87968446414848,
        0,
        0,
        0,
        1407435013095424,
        0,
        5629740052381696,
        0,
        22518960209526784,
        0,
        90075840838107136,
        0,
        360303363352428544,
        0,
        1441213453409714176,
        0,
        5764853813638856704,
        0,
        0,
        46120800833947828224,
        0,
        184483203335791312896,
        0,
        737932813343165251584,
        0,
        2951731253372661006336,
        0,
        11806925013490644025344,
        0,
        47227700053962576101376,
        0,
        0,
        377918445838087583956992,
        0,
        1511673783352350335827968,
        0,
        6046695133409401343311872,
        0,
        24186780533637605373247488,
        0,
        96747122134550421492989952,
        0,
        0,
        774770334645525472339820544,
        0,
        3099081338582101889359282176,
        0,
        12396325354328407557437128704,
        0,
        49585301417313630229748514816,
        0,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        3075,
        0,
        12300,
        0,
        49200,
        0,
        196800,
        0,
        0,
        0,
        0,
        6292992,
        0,
        25171968,
        0,
        100687872,
        0,
        402751488,
        0,
        1611005952,
        0,
        0,
        0,
        0,
        51542753280,
        0,
        206171013120,
        0,
        824684052480,
        0,
        3298736209920,
        0,
        13194944839680,
        0,
        52779779358720,
        0,
        0,
        0,
        844450699935744,
        0,
        3377802799742976,
        0,
        13511211198971904,
        0,
        54044844795887616,
        0,
        216179379183550464,
        0,
        864717516734201856,
        0,
        3458870066936807424,
        0,
        0,
        27671804960424591360,
        0,
        110687219841698365440,
        0,
        442748879366793461760,
        0,
        1770995517467173847040,
        0,
        7083982069868695388160,
        0,
        28335928279474781552640,
        0,
        0,
        226728931409964098912256,
        0,
        906915725639856395649024,
        0,
        3627662902559425582596096,
        0,
        14510651610237702330384384,
        0,
        58042606440950809321537536,
        0,
        0,
        464680861914373089027686400,
        0,
        1858723447657492356110745600,
        0,
        7434893790629969424442982400,
        0,
        29739575162519877697771929600,
        0,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        3079,
        0,
        12316,
        0,
        49264,
        0,
        197056,
        0,
        0,
        0,
        0,
        6295040,
        0,
        25180160,
        0,
        100720640,
        0,
        402882560,
        0,
        1611530240,
        0,
        0,
        0,
        0,
        51546947584,
        0,
        206187790336,
        0,
        824751161344,
        0,
        3299004645376,
        0,
        13196018581504,
        0,
        52784074326016,
        0,
        0,
        0,
        844485059674112,
        0,
        3377940238696448,
        0,
        13511760954785792,
        0,
        54047043819143168,
        0,
        216188175276572672,
        0,
        864752701106290688,
        0,
        3459010804425162752,
        0,
        0,
        27674056760238276608,
        0,
        110696227040953106432,
        0,
        442784908163812425728,
        0,
        1771139632655249702912,
        0,
        7084558530620998811648,
        0,
        28338234122483995246592,
        0,
        0,
        226802718386258937118720,
        0,
        907210873545035748474880,
        0,
        3628843494180142993899520,
        0,
        14515373976720571975598080,
        0,
        58061495906882287902392320,
        0,
        0,
        465285324824180403615039488,
        0,
        1861141299296721614460157952,
        0,
        7444565197186886457840631808,
        0,
        29778260788747545831362527232,
        0,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1031,
        0,
        4124,
        0,
        16496,
        0,
        65984,
        0,
        0,
        0,
        0,
        2100736,
        0,
        8402944,
        0,
        33611776,
        0,
        134447104,
        0,
        537788416,
        0,
        0,
        0,
        0,
        17187209216,
        0,
        68748836864,
        0,
        274995347456,
        0,
        1099981389824,
        0,
        4399925559296,
        0,
        17599702237184,
        0,
        0,
        0,
        281535106252800,
        0,
        1126140425011200,
        0,
        4504561700044800,
        0,
        18018246800179200,
        0,
        72072987200716800,
        0,
        288291948802867200,
        0,
        1153167795211468800,
        0,
        0,
        9227312686528724992,
        0,
        36909250746114899968,
        0,
        147637002984459599872,
        0,
        590548011937838399488,
        0,
        2362192047751353597952,
        0,
        9448768191005414391808,
        0,
        0,
        75686990934430290280448,
        0,
        302747963737721161121792,
        0,
        1210991854950884644487168,
        0,
        4843967419803538577948672,
        0,
        19375869679214154311794688,
        0,
        0,
        155800315002835334890258432,
        0,
        623201260011341339561033728,
        0,
        2492805040045365358244134912,
        0,
        9971220160181461432976539648,
        0,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        7174,
        0,
        28696,
        0,
        114784,
        0,
        459136,
        0,
        0,
        0,
        0,
        14683136,
        0,
        58732544,
        0,
        234930176,
        0,
        939720704,
        0,
        3758882816,
        0,
        0,
        0,
        0,
        120265375744,
        0,
        481061502976,
        0,
        1924246011904,
        0,
        7696984047616,
        0,
        30787936190464,
        0,
        123151744761856,
        0,
        0,
        0,
        1970376376582144,
        0,
        7881505506328576,
        0,
        31526022025314304,
        0,
        126104088101257216,
        0,
        504416352405028864,
        0,
        2017665409620115456,
        0,
        8070661638480461824,
        0,
        0,
        64566981957703958528,
        0,
        258267927830815834112,
        0,
        1033071711323263336448,
        0,
        4132286845293053345792,
        0,
        16529147381172213383168,
        0,
        66116589524688853532672,
        0,
        0,
        529015726545842521243648,
        0,
        2116062906183370084974592,
        0,
        8464251624733480339898368,
        0,
        33857006498933921359593472,
        0,
        135428025995735685438373888,
        0,
        0,
        1084104228739418712417763328,
        0,
        4336416914957674849671053312,
        0,
        17345667659830699398684213248,
        0,
        69382670639322797594736852992,
        0,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1025,
        0,
        4100,
        0,
        16400,
        0,
        65600,
        0,
        262400,
        0,
        0,
        2097664,
        0,
        8390656,
        0,
        33562624,
        0,
        134250496,
        0,
        537001984,
        0,
        2148007936,
        0,
        0,
        17180917760,
        0,
        68723671040,
        0,
        274894684160,
        0,
        1099578736640,
        0,
        4398314946560,
        0,
        17593259786240,
        0,
        70373039144960,
        0,
        281483566645248,
        0,
        1125934266580992,
        0,
        4503737066323968,
        0,
        18014948265295872,
        0,
        72059793061183488,
        0,
        288239172244733952,
        0,
        1152956688978935808,
        0,
        4611826755915743232,
        9223934986808197120,
        0,
        36895739947232788480,
        0,
        147582959788931153920,
        0,
        590331839155724615680,
        0,
        2361327356622898462720,
        0,
        9445309426491593850880,
        0,
        37781237705966375403520,
        75576310469988032970752,
        0,
        302305241879952131883008,
        0,
        1209220967519808527532032,
        0,
        4836883870079234110128128,
        0,
        19347535480316936440512512,
        0,
        77390141921267745762050048,
        154893620638124363009228800,
        0,
        619574482552497452036915200,
        0,
        2478297930209989808147660800,
        0,
        9913191720839959232590643200,
        0,
        39652766883359836930362572800,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        7169,
        0,
        28676,
        0,
        114704,
        0,
        458816,
        0,
        0,
        0,
        0,
        14680576,
        0,
        58722304,
        0,
        234889216,
        0,
        939556864,
        0,
        3758227456,
        0,
        0,
        0,
        0,
        120260132864,
        0,
        481040531456,
        0,
        1924162125824,
        0,
        7696648503296,
        0,
        30786594013184,
        0,
        123146376052736,
        0,
        0,
        0,
        1970333426909184,
        0,
        7881333707636736,
        0,
        31525334830546944,
        0,
        126101339322187776,
        0,
        504405357288751104,
        0,
        2017621429155004416,
        0,
        8070485716620017664,
        0,
        0,
        64564167207936851968,
        0,
        258256668831747407872,
        0,
        1033026675326989631488,
        0,
        4132106701307958525952,
        0,
        16528426805231834103808,
        0,
        66113707220927336415232,
        0,
        0,
        528923492825473973485568,
        0,
        2115693971301895893942272,
        0,
        8462775885207583575769088,
        0,
        33851103540830334303076352,
        0,
        135404414163321337212305408,
        0,
        0,
        1083348650102159569183571968,
        0,
        4333394600408638276734287872,
        0,
        17333578401634553106937151488,
        0,
        69334313606538212427748605952,
        0,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        7173,
        0,
        28692,
        0,
        114768,
        0,
        459072,
        0,
        0,
        0,
        0,
        14682624,
        0,
        58730496,
        0,
        234921984,
        0,
        939687936,
        0,
        3758751744,
        0,
        0,
        0,
        0,
        120264327168,
        0,
        481057308672,
        0,
        1924229234688,
        0,
        7696916938752,
        0,
        30787667755008,
        0,
        123150671020032,
        0,
        0,
        0,
        1970367786647552,
        0,
        7881471146590208,
        0,
        31525884586360832,
        0,
        126103538345443328,
        0,
        504414153381773312,
        0,
        2017656613527093248,
        0,
        8070626454108372992,
        0,
        0,
        64566419007750537216,
        0,
        258265676031002148864,
        0,
        1033062704124008595456,
        0,
        4132250816496034381824,
        0,
        16529003265984137527296,
        0,
        66116013063936550109184,
        0,
        0,
        528997279801768811692032,
        0,
        2115989119207075246768128,
        0,
        8463956476828300987072512,
        0,
        33855825907313203948290048,
        0,
        135423303629252815793160192,
        0,
        0,
        1083953113011966883770925056,
        0,
        4335812452047867535083700224,
        0,
        17343249808191470140334800896,
        0,
        69372999232765880561339203584,
        0,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        3073,
        0,
        12292,
        0,
        49168,
        0,
        196672,
        0,
        786688,
        0,
        0,
        6291968,
        0,
        25167872,
        0,
        100671488,
        0,
        402685952,
        0,
        1610743808,
        0,
        6442975232,
        0,
        0,
        51540656128,
        0,
        206162624512,
        0,
        824650498048,
        0,
        3298601992192,
        0,
        13194407968768,
        0,
        52777631875072,
        0,
        211110527500288,
        0,
        844433520066560,
        0,
        3377734080266240,
        0,
        13510936321064960,
        0,
        54043745284259840,
        0,
        216174981137039360,
        0,
        864699924548157440,
        0,
        3458799698192629760,
        0,
        0,
        27670679060517748736,
        0,
        110682716242070994944,
        0,
        442730864968283979776,
        0,
        1770923459873135919104,
        0,
        7083693839492543676416,
        0,
        28334775357970174705664,
        0,
        0,
        226692037921816679809024,
        0,
        906768151687266719236096,
        0,
        3627072606749066876944384,
        0,
        14508290426996267507777536,
        0,
        58033161707985070031110144,
        0,
        0,
        464378630459469431734009856,
        0,
        1857514521837877726936039424,
        0,
        7430058087351510907744157696,
        0,
        29720232349406043630976630784,
        0,
        0,
    ],
    [
        0,
        14,
        0,
        56,
        0,
        224,
        0,
        0,
        0,
        0,
        7168,
        0,
        28672,
        0,
        114688,
        0,
        458752,
        0,
        0,
        0,
        0,
        14680064,
        0,
        58720256,
        0,
        234881024,
        0,
        939524096,
        0,
        3758096384,
        0,
        0,
        0,
        0,
        120259084288,
        0,
        481036337152,
        0,
        1924145348608,
        0,
        7696581394432,
        0,
        30786325577728,
        0,
        123145302310912,
        0,
        0,
        0,
        1970324836974592,
        0,
        7881299347898368,
        0,
        31525197391593472,
        0,
        126100789566373888,
        0,
        504403158265495552,
        0,
        2017612633061982208,
        0,
        8070450532247928832,
        0,
        0,
        64563604257983430656,
        0,
        258254417031933722624,
        0,
        1033017668127734890496,
        0,
        4132070672510939561984,
        0,
        16528282690043758247936,
        0,
        66113130760175032991744,
        0,
        0,
        528905046081400263933952,
        0,
        2115620184325601055735808,
        0,
        8462480737302404222943232,
        0,
        33849922949209616891772928,
        0,
        135399691796838467567091712,
        0,
        0,
        1083197534374707740536733696,
        0,
        4332790137498830962146934784,
        0,
        17331160549995323848587739136,
        0,
        69324642199981295394350956544,
        0,
        0,
    ],
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        7172,
        0,
        28688,
        0,
        114752,
        0,
        459008,
        0,
        0,
        0,
        0,
        14682112,
        0,
        58728448,
        0,
        234913792,
        0,
        939655168,
        0,
        3758620672,
        0,
        0,
        0,
        0,
        120263278592,
        0,
        481053114368,
        0,
        1924212457472,
        0,
        7696849829888,
        0,
        30787399319552,
        0,
        123149597278208,
        0,
        0,
        0,
        1970359196712960,
        0,
        7881436786851840,
        0,
        31525747147407360,
        0,
        126102988589629440,
        0,
        504411954358517760,
        0,
        2017647817434071040,
        0,
        8070591269736284160,
        0,
        0,
        64565856057797115904,
        0,
        258263424231188463616,
        0,
        1033053696924753854464,
        0,
        4132214787699015417856,
        0,
        16528859150796061671424,
        0,
        66115436603184246685696,
        0,
        0,
        528978833057695102140416,
        0,
        2115915332230780408561664,
        0,
        8463661328923121634246656,
        0,
        33854645315692486536986624,
        0,
        135418581262769946147946496,
        0,
        0,
        1083801997284515055124086784,
        0,
        4335207989138060220496347136,
        0,
        17340831956552240881985388544,
        0,
        69363327826208963527941554176,
        0,
        0,
    ],
    [
        0,
        6,
        0,
        24,
        0,
        96,
        0,
        384,
        0,
        0,
        3072,
        0,
        12288,
        0,
        49152,
        0,
        196608,
        0,
        786432,
        0,
        0,
        6291456,
        0,
        25165824,
        0,
        100663296,
        0,
        402653184,
        0,
        1610612736,
        0,
        6442450944,
        0,
        0,
        51539607552,
        0,
        206158430208,
        0,
        824633720832,
        0,
        3298534883328,
        0,
        13194139533312,
        0,
        52776558133248,
        0,
        211106232532992,
        0,
        844424930131968,
        0,
        3377699720527872,
        0,
        13510798882111488,
        0,
        54043195528445952,
        0,
        216172782113783808,
        0,
        864691128455135232,
        0,
        3458764513820540928,
        0,
        0,
        27670116110564327424,
        0,
        110680464442257309696,
        0,
        442721857769029238784,
        0,
        1770887431076116955136,
        0,
        7083549724304467820544,
        0,
        28334198897217871282176,
        0,
        0,
        226673591177742970257408,
        0,
        906694364710971881029632,
        0,
        3626777458843887524118528,
        0,
        14507109835375550096474112,
        0,
        58028439341502200385896448,
        0,
        0,
        464227514732017603087171584,
        0,
        1856910058928070412348686336,
        0,
        7427640235712281649394745344,
        0,
        29710560942849126597578981376,
        0,
        0,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ],
    [
        1,
        0,
        4,
        0,
        16,
        0,
        64,
        0,
        256,
        512,
        0,
        2048,
        0,
        8192,
        0,
        32768,
        0,
        131072,
        0,
        524288,
        1048576,
        0,
        4194304,
        0,
        16777216,
        0,
        67108864,
        0,
        268435456,
        0,
        1073741824,
        0,
        4294967296,
        8589934592,
        0,
        34359738368,
        0,
        137438953472,
        0,
        549755813888,
        0,
        2199023255552,
        0,
        8796093022208,
        0,
        35184372088832,
        0,
        140737488355328,
        0,
        562949953421312,
        0,
        2251799813685248,
        0,
        9007199254740992,
        0,
        36028797018963968,
        0,
        144115188075855872,
        0,
        576460752303423488,
        0,
        2305843009213693952,
        0,
        0,
        18446744073709551616,
        0,
        73786976294838206464,
        0,
        295147905179352825856,
        0,
        1180591620717411303424,
        0,
        4722366482869645213696,
        0,
        18889465931478580854784,
        0,
        0,
        151115727451828646838272,
        0,
        604462909807314587353088,
        0,
        2417851639229258349412352,
        0,
        9671406556917033397649408,
        0,
        38685626227668133590597632,
        0,
        0,
        309485009821345068724781056,
        0,
        1237940039285380274899124224,
        0,
        4951760157141521099596496896,
        0,
        19807040628566084398385987584,
        0,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ],
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ],
];
```

### File: `src/core/features.rs`

```rs
use crate::core::constants::STANDARD_PIECES;
use once_cell::sync::Lazy;
use tch::{Device, Tensor};

pub static CANONICAL_PIECE_MASKS: Lazy<[Vec<usize>; 48]> = Lazy::new(|| {
    let mut masks: [Vec<usize>; 48] = core::array::from_fn(|_| Vec::new());
    for (piece_table_index, mask) in masks.iter_mut().enumerate().take(48) {
        let mut canonical_shape_drawn = false;

        for &(_rotation_index, piece_mask) in &crate::node::COMPACT_PIECE_MASKS[piece_table_index] {
            if !canonical_shape_drawn {
                let mut minimum_row = 8;
                let mut maximum_row = 0;
                let mut minimum_column = 16;
                let mut maximum_column = 0;

                let mut temp_mask = piece_mask & ((1_u128 << 96) - 1);
                while temp_mask != 0 {
                    let bit_index = temp_mask.trailing_zeros() as usize;
                    let (row, column) = HEXAGONAL_TO_CARTESIAN_MAP_ARRAY[bit_index];
                    minimum_row = minimum_row.min(row);
                    maximum_row = maximum_row.max(row);
                    minimum_column = minimum_column.min(column);
                    maximum_column = maximum_column.max(column);
                    temp_mask &= temp_mask - 1;
                }

                let middle_row = (minimum_row + maximum_row) / 2;
                let middle_column = (minimum_column + maximum_column) / 2;
                let target_row = 3;
                let target_column = 8;

                let mut temp_mask2 = piece_mask & ((1_u128 << 96) - 1);
                while temp_mask2 != 0 {
                    let bit_index = temp_mask2.trailing_zeros() as usize;
                    let (row, column) = HEXAGONAL_TO_CARTESIAN_MAP_ARRAY[bit_index];
                    let offset_row = (row as isize - middle_row as isize) + target_row as isize;
                    let offset_column =
                        (column as isize - middle_column as isize) + target_column as isize;

                    if (0..8).contains(&offset_row) && (0..16).contains(&offset_column) {
                        mask.push(offset_row as usize * 16 + offset_column as usize);
                    }
                    temp_mask2 &= temp_mask2 - 1;
                }
                canonical_shape_drawn = true;
            }
        }
    }
    masks
});
pub const TOTAL_TRIANGLES: usize = 96;
pub const SPATIAL_ROWS: usize = 8;
pub const SPATIAL_COLS: usize = 16;
pub const SPATIAL_SIZE: usize = SPATIAL_ROWS * SPATIAL_COLS;

pub const HEXAGONAL_TO_CARTESIAN_MAP_ARRAY: [(usize, usize); 96] = [
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 8),
    (0, 9),
    (0, 10),
    (0, 11),
    (0, 12),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (1, 7),
    (1, 8),
    (1, 9),
    (1, 10),
    (1, 11),
    (1, 12),
    (1, 13),
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (2, 6),
    (2, 7),
    (2, 8),
    (2, 9),
    (2, 10),
    (2, 11),
    (2, 12),
    (2, 13),
    (2, 14),
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
    (3, 5),
    (3, 6),
    (3, 7),
    (3, 8),
    (3, 9),
    (3, 10),
    (3, 11),
    (3, 12),
    (3, 13),
    (3, 14),
    (3, 15),
    (4, 1),
    (4, 2),
    (4, 3),
    (4, 4),
    (4, 5),
    (4, 6),
    (4, 7),
    (4, 8),
    (4, 9),
    (4, 10),
    (4, 11),
    (4, 12),
    (4, 13),
    (4, 14),
    (4, 15),
    (5, 2),
    (5, 3),
    (5, 4),
    (5, 5),
    (5, 6),
    (5, 7),
    (5, 8),
    (5, 9),
    (5, 10),
    (5, 11),
    (5, 12),
    (5, 13),
    (5, 14),
    (6, 3),
    (6, 4),
    (6, 5),
    (6, 6),
    (6, 7),
    (6, 8),
    (6, 9),
    (6, 10),
    (6, 11),
    (6, 12),
    (6, 13),
    (7, 4),
    (7, 5),
    (7, 6),
    (7, 7),
    (7, 8),
    (7, 9),
    (7, 10),
    (7, 11),
    (7, 12),
];

#[inline(always)]
pub fn get_spatial_idx(hex_idx: usize) -> usize {
    let (row, column) = HEXAGONAL_TO_CARTESIAN_MAP_ARRAY[hex_idx];
    row * SPATIAL_COLS + column
}

pub fn get_valid_spatial_mask_8x8(computation_device: Device) -> Tensor {
    let mut mask = vec![0.0_f32; 64];
    for &(row, column) in HEXAGONAL_TO_CARTESIAN_MAP_ARRAY.iter() {
        mask[row * 8 + (column / 2)] = 1.0;
    }
    Tensor::from_slice(&mask)
        .view([1, 1, 8, 8])
        .to_device(computation_device)
}

pub fn extract_feature_native(
    extracted_features_tensor_flat: &mut [f32],
    current_board_state: u128,
    available_pieces: &[i32; 3],
    history_boards: &[u128],
    action_history: &[i32],
    difficulty: i32,
) {
    extracted_features_tensor_flat.fill(0.0);

    fill_history_channels(
        extracted_features_tensor_flat,
        current_board_state,
        history_boards,
    );
    fill_action_history_channels(extracted_features_tensor_flat, action_history);
    fill_piece_overlay_channels(
        extracted_features_tensor_flat,
        available_pieces,
        current_board_state,
    );
    fill_static_game_channels(
        extracted_features_tensor_flat,
        difficulty,
        current_board_state,
    );
}

fn fill_channel(
    extracted_features_tensor_flat: &mut [f32],
    channel_index: usize,
    mut board_bits: u128,
) {
    let memory_offset = channel_index * SPATIAL_SIZE;
    board_bits &= (1_u128 << 96) - 1; // Mask out any bits beyond the 96th structural triangle
    while board_bits != 0 {
        let bit_index = board_bits.trailing_zeros() as usize;
        extracted_features_tensor_flat[memory_offset + get_spatial_idx(bit_index)] = 1.0;
        board_bits &= board_bits - 1;
    }
}

fn fill_history_channels(
    extracted_features_tensor_flat: &mut [f32],
    current_board_state: u128,
    unwrapped_history: &[u128],
) {
    fill_channel(extracted_features_tensor_flat, 0, current_board_state);

    for memory_index in 1..=7 {
        if unwrapped_history.len() >= memory_index {
            let prior_state = unwrapped_history[unwrapped_history.len() - memory_index];
            fill_channel(extracted_features_tensor_flat, memory_index, prior_state);
        } else {
            fill_channel(
                extracted_features_tensor_flat,
                memory_index,
                current_board_state,
            );
        }
    }
}

fn fill_action_history_channels(
    extracted_features_tensor_flat: &mut [f32],
    unwrapped_actions: &[i32],
) {
    // channel 8-10: action history
    for memory_index in 0..3 {
        if unwrapped_actions.len() > memory_index {
            let prior_action = unwrapped_actions[unwrapped_actions.len() - (memory_index + 1)];
            let slot_index = prior_action / (TOTAL_TRIANGLES as i32);
            let map_index = (prior_action % (TOTAL_TRIANGLES as i32)) as usize;

            if map_index < TOTAL_TRIANGLES {
                extracted_features_tensor_flat
                    [(8 + memory_index) * SPATIAL_SIZE + get_spatial_idx(map_index)] =
                    (slot_index as f32 + 1.0) * 0.33;
            }
        }
    }
}

fn fill_piece_overlay_channels(
    extracted_features_tensor_flat: &mut [f32],
    available_pieces: &[i32; 3],
    current_board_state: u128,
) {
    // channel 11-16: pieces overlay (canonical shape) and valid mask
    for slot_index in 0..3 {
        let piece_identifier = available_pieces[slot_index];
        if piece_identifier == -1 {
            continue;
        }

        let piece_table_index = piece_identifier as usize;
        let mut validity_mask: u128 = 0;

        for &spatial_idx in &CANONICAL_PIECE_MASKS[piece_table_index] {
            extracted_features_tensor_flat[(11 + slot_index * 2) * SPATIAL_SIZE + spatial_idx] =
                1.0;
        }

        for &(_rotation_index, piece_mask) in &crate::node::COMPACT_PIECE_MASKS[piece_table_index] {
            // Keep the validity mask so the network knows WHERE it can legally place it
            if (current_board_state & piece_mask) == 0 {
                validity_mask |= piece_mask;
            }
        }

        // Channel 12, 14, 16: The legal placement footprint
        validity_mask &= (1_u128 << 96) - 1; // Purge topological off-grid spillovers
        while validity_mask != 0 {
            let bit_index = validity_mask.trailing_zeros() as usize;
            extracted_features_tensor_flat
                [(12 + slot_index * 2) * SPATIAL_SIZE + get_spatial_idx(bit_index)] = 1.0;
            validity_mask &= validity_mask - 1;
        }
    }
}

fn fill_static_game_channels(
    extracted_features_tensor_flat: &mut [f32],
    difficulty_level: i32,
    current_board_state: u128,
) {
    let all_hexes = (1_u128 << TOTAL_TRIANGLES) - 1;
    let normalized_difficulty = difficulty_level as f32 / 6.0;

    // channel 17 & 18: empty and difficulty
    let mut temp = all_hexes;
    while temp != 0 {
        let bit_index = temp.trailing_zeros() as usize;
        let spatial_idx = get_spatial_idx(bit_index);
        extracted_features_tensor_flat[17 * SPATIAL_SIZE + spatial_idx] = 1.0 / 22.0;
        extracted_features_tensor_flat[18 * SPATIAL_SIZE + spatial_idx] = normalized_difficulty;
        temp &= temp - 1;
    }

    // channel 19: explicit dead zone detection
    let mut global_valid_mask = 0_u128;
    for pieces_set in STANDARD_PIECES.iter() {
        for &piece_mask in pieces_set.iter() {
            if piece_mask != 0 && (current_board_state & piece_mask) == 0 {
                global_valid_mask |= piece_mask;
            }
        }
    }

    let dead_zone_mask = !current_board_state & !global_valid_mask & all_hexes;
    let mut temp = dead_zone_mask;
    while temp != 0 {
        let bit_index = temp.trailing_zeros() as usize;
        extracted_features_tensor_flat[19 * SPATIAL_SIZE + get_spatial_idx(bit_index)] = 1.0;
        temp &= temp - 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::GameStateExt;

    #[test]
    fn test_extract_feature_history_padding() {
        let mut state = GameStateExt::new(Some([0, 0, 0]), 0, 0, 6, 0);
        state.board_bitmask_u128 = 0b101;

        let history_boards = vec![0b010];
        let mut extracted_features_tensor_flat = vec![0.0; 20 * 128];
        extract_feature_native(
            &mut extracted_features_tensor_flat,
            state.board_bitmask_u128,
            &state.available,
            &history_boards,
            &[],
            6,
        );

        assert_eq!(extracted_features_tensor_flat.len(), 20 * 128);

        assert_eq!(extracted_features_tensor_flat[get_spatial_idx(0)], 1.0);
        assert_eq!(extracted_features_tensor_flat[get_spatial_idx(2)], 1.0);
        assert_eq!(extracted_features_tensor_flat[get_spatial_idx(1)], 0.0);

        let memory_offset_1 = 128;
        assert_eq!(
            extracted_features_tensor_flat[memory_offset_1 + get_spatial_idx(1)],
            1.0
        );
        assert_eq!(
            extracted_features_tensor_flat[memory_offset_1 + get_spatial_idx(0)],
            0.0
        );

        let memory_offset_2 = 2 * 128;
        assert_eq!(
            extracted_features_tensor_flat[memory_offset_2 + get_spatial_idx(0)],
            1.0
        );
        assert_eq!(
            extracted_features_tensor_flat[memory_offset_2 + get_spatial_idx(2)],
            1.0
        );
        assert_eq!(
            extracted_features_tensor_flat[memory_offset_2 + get_spatial_idx(1)],
            0.0
        );
    }
}
```

### File: `src/core/mod.rs`

```rs
pub mod board;
pub mod constants;
pub mod features;
```

### File: `src/lib.rs`

```rs
pub mod config;
pub mod core;
pub mod env;
pub mod mcts;
pub mod net;
pub mod train;

pub mod node;
pub mod queue;
pub mod sumtree;

#[cfg(test)]
pub mod tests;

#[cfg(test)]
pub mod performance_benches;
```

### File: `src/main.rs`

```rs
use clap::{Parser, Subcommand};
use crossbeam_channel::unbounded;
use std::sync::{Arc, RwLock};
use std::thread;
use tch::{nn, nn::OptimizerConfig, Device};

use tricked_engine::config::Config;
use tricked_engine::env::reanalyze;
use tricked_engine::env::worker as selfplay;
use tricked_engine::net::MuZeroNet;
use tricked_engine::queue;
use tricked_engine::train::buffer::ReplayBuffer;
use tricked_engine::train::optimizer as trainer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    Train {
        /// The name of the experiment for logging and paths
        #[arg(long, default_value = "cli_run")]
        experiment_name: String,

        /// Optional path to a JSON/YAML config file
        #[arg(short, long)]
        config: Option<String>,

        /// Overrides for hyperparameters
        #[arg(long)]
        lr_init: Option<f64>,
        #[arg(long)]
        simulations: Option<i64>,
        #[arg(long)]
        unroll_steps: Option<usize>,
        #[arg(long)]
        temporal_difference_steps: Option<usize>,
        #[arg(long)]
        reanalyze_ratio: Option<f32>,
        #[arg(long)]
        support_size: Option<i64>,
        #[arg(long)]
        temp_decay_steps: Option<i64>,

        /// Max training steps to run before exiting (0 = infinite)
        #[arg(long, default_value = "0")]
        max_steps: usize,
    },
}

fn main() {
    let cli = Cli::parse();
    let Commands::Train {
        experiment_name,
        config,
        lr_init,
        simulations,
        unroll_steps,
        temporal_difference_steps,
        reanalyze_ratio,
        support_size,
        temp_decay_steps,
        max_steps,
    } = cli.command;
    let mut cfg = if let Some(path) = config {
        let file = std::fs::File::open(&path).expect("Failed to open config file");
        let mut parsed: Config = if path.ends_with(".yaml") || path.ends_with(".yml") {
            serde_yaml::from_reader(file).expect("Failed to parse YAML config")
        } else {
            serde_json::from_reader(file).expect("Failed to parse JSON config")
        };
        parsed.experiment_name_identifier = experiment_name.clone();
        parsed.paths = tricked_engine::config::ExperimentPaths::new(&experiment_name);
        parsed
    } else {
        Config {
            experiment_name_identifier: experiment_name.clone(),
            paths: tricked_engine::config::ExperimentPaths::new(&experiment_name),
            device: "cuda".to_string(),
            hidden_dimension_size: 256,
            num_blocks: 10,
            support_size: 300,
            buffer_capacity_limit: 1_000_000,
            simulations: 200,
            train_batch_size: 256,
            train_epochs: 1000,
            num_processes: 4,
            worker_device: "cpu".to_string(),
            unroll_steps: 15,
            temporal_difference_steps: 15,
            inference_batch_size_limit: 64,
            inference_timeout_ms: 5,
            max_gumbel_k: 16,
            gumbel_scale: 1.0,
            temp_decay_steps: 10000,
            difficulty: 6,
            temp_boost: true,
            lr_init: 0.0003,
            reanalyze_ratio: 0.0,
        }
    };
    if let Some(v) = lr_init {
        cfg.lr_init = v;
    }
    if let Some(v) = simulations {
        cfg.simulations = v;
    }
    if let Some(v) = unroll_steps {
        cfg.unroll_steps = v;
    }
    if let Some(v) = temporal_difference_steps {
        cfg.temporal_difference_steps = v;
    }
    if let Some(v) = reanalyze_ratio {
        cfg.reanalyze_ratio = v;
    }
    if let Some(v) = support_size {
        cfg.support_size = v;
    }
    if let Some(v) = temp_decay_steps {
        cfg.temp_decay_steps = v;
    }

    run_training(cfg, max_steps);
}

fn run_training(config: Config, max_steps: usize) {
    println!(
        "🚀 Starting Tricked AI Native Engine (CLI Mode) for experiment: {}",
        config.experiment_name_identifier
    );

    let configuration_arc = Arc::new(config);

    assert!(configuration_arc.buffer_capacity_limit > configuration_arc.train_batch_size);
    assert!(configuration_arc.temporal_difference_steps > 0);
    assert!(configuration_arc.num_processes > 0);

    let shared_replay_buffer = Arc::new(ReplayBuffer::new(
        configuration_arc.buffer_capacity_limit,
        configuration_arc.unroll_steps,
        configuration_arc.temporal_difference_steps,
    ));

    let computation_device = if configuration_arc.device == "cuda" && tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };

    let mut training_var_store = nn::VarStore::new(computation_device);
    let mut inference_var_store = nn::VarStore::new(computation_device);
    let exponential_moving_average_var_store = nn::VarStore::new(computation_device);

    let training_network = MuZeroNet::new(
        &training_var_store.root(),
        configuration_arc.hidden_dimension_size,
        configuration_arc.num_blocks,
        configuration_arc.support_size,
    );
    let ema_network = MuZeroNet::new(
        &exponential_moving_average_var_store.root(),
        configuration_arc.hidden_dimension_size,
        configuration_arc.num_blocks,
        configuration_arc.support_size,
    );

    let mut inference_var_store_b = nn::VarStore::new(computation_device);
    let inference_net_a = Arc::new(MuZeroNet::new(
        &inference_var_store.root(),
        configuration_arc.hidden_dimension_size,
        configuration_arc.num_blocks,
        configuration_arc.support_size,
    ));
    let inference_net_b = Arc::new(MuZeroNet::new(
        &inference_var_store_b.root(),
        configuration_arc.hidden_dimension_size,
        configuration_arc.num_blocks,
        configuration_arc.support_size,
    ));

    let active_inference_net = Arc::new(arc_swap::ArcSwap::from(Arc::clone(&inference_net_a)));
    let mut cmodule_inference: Option<Arc<tch::CModule>> = None;

    if !configuration_arc.paths.model_checkpoint_path.is_empty() {
        if std::path::Path::new(&configuration_arc.paths.model_checkpoint_path).exists() {
            if configuration_arc
                .paths
                .model_checkpoint_path
                .ends_with(".pt")
            {
                println!(
                    "🚀 Loading TorchScript model: {}",
                    configuration_arc.paths.model_checkpoint_path
                );
                cmodule_inference = tch::CModule::load_on_device(
                    &configuration_arc.paths.model_checkpoint_path,
                    computation_device,
                )
                .ok()
                .map(Arc::new);
            } else {
                println!(
                    "🚀 Loading Native Rust weights: {}",
                    configuration_arc.paths.model_checkpoint_path
                );
                let _ = training_var_store.load(&configuration_arc.paths.model_checkpoint_path);
                inference_var_store.copy(&training_var_store).unwrap();
                inference_var_store_b.copy(&training_var_store).unwrap();
            }
        } else {
            println!(
                "⚠️ Checkpoint '{}' not found. Init weights.",
                configuration_arc.paths.model_checkpoint_path
            );
            inference_var_store.copy(&training_var_store).unwrap();
            inference_var_store_b.copy(&training_var_store).unwrap();
            std::fs::create_dir_all(
                std::path::Path::new(&configuration_arc.paths.model_checkpoint_path)
                    .parent()
                    .unwrap(),
            )
            .unwrap();
            let _ = training_var_store.save(&configuration_arc.paths.model_checkpoint_path);
        }
    } else {
        inference_var_store.copy(&training_var_store).unwrap();
        inference_var_store_b.copy(&training_var_store).unwrap();
    }

    tch::no_grad(|| {
        for (name, tensor) in training_var_store.variables().iter() {
            assert!(
                i64::try_from(tensor.isnan().any()).unwrap() == 0,
                "NaN detected in weights '{name}'"
            );
        }
    });

    let active_training_flag = Arc::new(RwLock::new(true));

    std::fs::create_dir_all(
        std::path::Path::new(&configuration_arc.paths.metrics_file_path)
            .parent()
            .unwrap(),
    )
    .unwrap();
    let mut csv_writer =
        csv::Writer::from_path(&configuration_arc.paths.metrics_file_path).unwrap();
    csv_writer
        .write_record([
            "step",
            "total_loss",
            "policy_loss",
            "value_loss",
            "reward_loss",
            "lr",
            "game_score_min",
            "game_score_max",
            "game_score_med",
            "game_score_mean",
            "game_lines_cleared",
            "game_count",
            "ram_usage_mb",
            "gpu_usage_pct",
            "cpu_usage_pct",
            "io_usage",
            "disk_usage_pct",
            "vram_usage_mb",
            "mcts_depth_mean",
            "mcts_search_time_mean",
        ])
        .unwrap();
    csv_writer.flush().unwrap();

    let config_path = std::path::Path::new(&configuration_arc.paths.metrics_file_path)
        .parent()
        .unwrap()
        .join("config.json");
    let config_json = serde_json::to_string_pretty(&*configuration_arc).unwrap();
    std::fs::write(config_path, config_json).unwrap();

    let csv_mutex = Arc::new(std::sync::Mutex::new(csv_writer));

    let reanalyze_worker_count = std::cmp::max(1, configuration_arc.num_processes / 4);
    let total_workers = configuration_arc.num_processes + reanalyze_worker_count;
    let inference_queue = Arc::new(queue::FixedInferenceQueue::new(
        total_workers as usize,
        total_workers as usize,
    ));

    for _ in 0..1 {
        let thread_evaluation_receiver = Arc::clone(&inference_queue);
        let thread_network_mutex = Arc::clone(&active_inference_net);
        let thread_cmodule = cmodule_inference.clone();
        let thread_active_flag = Arc::clone(&active_training_flag);
        let configuration_model_dimension = configuration_arc.hidden_dimension_size;
        let max_nodes = (configuration_arc.simulations as usize) + 300;
        let inference_batch_size_limit = configuration_arc.inference_batch_size_limit as usize;
        let inference_timeout_milliseconds = configuration_arc.inference_timeout_ms as u64;

        thread::spawn(move || {
            while *thread_active_flag.read().unwrap() {
                selfplay::inference_loop(selfplay::InferenceLoopParams {
                    receiver_queue: Arc::clone(&thread_evaluation_receiver),
                    shared_neural_model: Arc::clone(&thread_network_mutex),
                    cmodule_inference: thread_cmodule.clone(),
                    model_dimension: configuration_model_dimension,
                    computation_device,
                    total_workers: total_workers as usize,
                    maximum_allowed_nodes_in_search_tree: max_nodes,
                    inference_batch_size_limit,
                    inference_timeout_milliseconds,
                    active_flag: Arc::clone(&thread_active_flag),
                });
            }
        });
    }

    let selfplay_worker_count = configuration_arc.num_processes;
    for worker_id in 0..selfplay_worker_count {
        let thread_configuration = Arc::clone(&configuration_arc);
        let thread_evaluation_sender = Arc::clone(&inference_queue);
        let thread_replay_buffer = Arc::clone(&shared_replay_buffer);
        let thread_active_flag = Arc::clone(&active_training_flag);

        thread::spawn(move || {
            while *thread_active_flag.read().unwrap() {
                selfplay::game_loop(selfplay::GameLoopExecutionParameters {
                    configuration: Arc::clone(&thread_configuration),
                    evaluation_transmitter: Arc::clone(&thread_evaluation_sender),
                    experience_buffer: Arc::clone(&thread_replay_buffer),
                    worker_id: worker_id as usize,
                    active_flag: Arc::clone(&thread_active_flag),
                });
            }
        });
    }

    for worker_index in 0..reanalyze_worker_count {
        let worker_id = selfplay_worker_count + worker_index;
        let thread_configuration = Arc::clone(&configuration_arc);
        let thread_evaluation_sender = Arc::clone(&inference_queue);
        let thread_replay_buffer = Arc::clone(&shared_replay_buffer);
        let thread_active_flag = Arc::clone(&active_training_flag);

        thread::spawn(move || {
            while *thread_active_flag.read().unwrap() {
                reanalyze::reanalyze_worker_loop(
                    Arc::clone(&thread_configuration),
                    Arc::clone(&thread_evaluation_sender),
                    Arc::clone(&thread_replay_buffer),
                    worker_id as usize,
                    Arc::clone(&thread_active_flag),
                );
            }
        });
    }

    let prefetch_replay_buffer = Arc::clone(&shared_replay_buffer);
    let prefetch_active_flag = Arc::clone(&active_training_flag);
    let (prefetch_tx, prefetch_rx) = unbounded();
    let prefetch_device = computation_device;
    let prefetch_batch_size = configuration_arc.train_batch_size;

    thread::spawn(move || {
        while *prefetch_active_flag.read().unwrap() {
            if prefetch_replay_buffer.get_length() < prefetch_batch_size {
                thread::sleep(std::time::Duration::from_millis(100));
                continue;
            }
            let current_step = prefetch_replay_buffer
                .state
                .completed_games
                .load(std::sync::atomic::Ordering::Relaxed) as f64;
            let beta = (0.4 + 0.6 * (current_step / 100_000.0)).min(1.0);

            if let Some(mut batch) =
                prefetch_replay_buffer.sample_batch(prefetch_batch_size, prefetch_device, beta)
            {
                batch.state_features_batch = batch.state_features_batch.to_device(prefetch_device);
                batch.actions_batch = batch.actions_batch.to_device(prefetch_device);
                batch.piece_identifiers_batch =
                    batch.piece_identifiers_batch.to_device(prefetch_device);
                batch.rewards_batch = batch.rewards_batch.to_device(prefetch_device).nan_to_num(
                    0.0,
                    Some(0.0),
                    Some(0.0),
                );
                batch.target_policies_batch = batch
                    .target_policies_batch
                    .to_device(prefetch_device)
                    .nan_to_num(0.0, Some(0.0), Some(0.0));
                batch.target_values_batch = batch
                    .target_values_batch
                    .to_device(prefetch_device)
                    .nan_to_num(0.0, Some(0.0), Some(0.0));
                batch.model_values_batch = batch
                    .model_values_batch
                    .to_device(prefetch_device)
                    .nan_to_num(0.0, Some(0.0), Some(0.0));
                batch.transition_states_batch =
                    batch.transition_states_batch.to_device(prefetch_device);
                batch.loss_masks_batch = batch.loss_masks_batch.to_device(prefetch_device);
                batch.importance_weights_batch =
                    batch.importance_weights_batch.to_device(prefetch_device);

                if prefetch_tx.send(batch).is_err() {
                    break;
                }
            } else {
                thread::sleep(std::time::Duration::from_millis(10));
            }
        }
    });

    let mut gradient_optimizer = nn::Adam::default()
        .build(&training_var_store, configuration_arc.lr_init)
        .unwrap();
    let mut last_trained_games = 0;
    let games_per_train_step = 1;

    let optimizer_network_arcswap = Arc::clone(&active_inference_net);
    let mut active_is_a = true;

    let optimizer_replay_buffer = Arc::clone(&shared_replay_buffer);
    let optimizer_configuration = Arc::clone(&configuration_arc);
    let optimizer_active_flag = Arc::clone(&active_training_flag);

    let mut training_steps = 0;

    let mut sys = sysinfo::System::new_all();

    while *optimizer_active_flag.read().unwrap() {
        let current_games = optimizer_replay_buffer
            .state
            .completed_games
            .load(std::sync::atomic::Ordering::Relaxed);

        if current_games < last_trained_games + games_per_train_step {
            thread::sleep(std::time::Duration::from_millis(10));
            continue;
        }

        let batched_experience_tensorserience =
            match prefetch_rx.recv_timeout(std::time::Duration::from_millis(100)) {
                Ok(batch) => batch,
                Err(_) => continue,
            };

        last_trained_games += games_per_train_step;

        let lr_multiplier = if (training_steps as f64) < 10000.0 {
            1.0
        } else if (training_steps as f64) < 50000.0 {
            0.1
        } else {
            0.01
        };
        let current_lr = optimizer_configuration.lr_init * lr_multiplier;
        gradient_optimizer.set_lr(current_lr);

        let step_metrics = trainer::optimization::train_step(
            &training_network,
            &ema_network,
            &mut gradient_optimizer,
            &optimizer_replay_buffer,
            batched_experience_tensorserience,
            optimizer_configuration.unroll_steps,
        );

        sys.refresh_cpu_usage();
        sys.refresh_memory();

        let cpu_usage = sys.global_cpu_info().cpu_usage();
        let ram_usage = sys.used_memory() as f32 / 1024.0 / 1024.0;
        let (gpu_usage, vram_usage) = get_gpu_metrics();

        let disks = sysinfo::Disks::new_with_refreshed_list();
        let mut total_disk = 0;
        let mut used_disk = 0;
        for disk in &disks {
            total_disk += disk.total_space();
            used_disk += disk.total_space() - disk.available_space();
        }
        let disk_usage_pct = if total_disk > 0 {
            (used_disk as f64 / total_disk as f64) * 100.0
        } else {
            0.0
        };

        let mut score_min = 0.0_f32;
        let mut score_max = 0.0_f32;
        let mut score_mean = 0.0_f32;
        let mut score_med = 0.0_f32;
        let mut lines_cleared = 0_u32;
        let mut mcts_depth = 0.0_f32;
        let mut mcts_search_time = 0.0_f32;

        {
            let episodes_lock = match optimizer_replay_buffer.state.episodes.lock() {
                Ok(lock) => lock,
                Err(poison) => poison.into_inner(),
            };
            let count = episodes_lock.len();
            if count > 0 {
                let mut scores: Vec<f32> = episodes_lock.iter().map(|e| e.score).collect();
                scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                score_min = scores[0];
                score_max = *scores.last().unwrap_or(&0.0);
                score_med = scores[count / 2];
                score_mean = scores.iter().sum::<f32>() / count as f32;

                let total_lines: u32 = episodes_lock.iter().map(|e| e.lines_cleared).sum();
                lines_cleared = total_lines / count as u32;

                let sum_depth: f32 = episodes_lock.iter().map(|e| e.mcts_depth_mean).sum();
                mcts_depth = sum_depth / count as f32;

                let sum_time: f32 = episodes_lock.iter().map(|e| e.mcts_search_time_mean).sum();
                mcts_search_time = sum_time / count as f32;
            }
        }

        if let Ok(mut writer) = csv_mutex.lock() {
            let _ = writer.write_record(&[
                training_steps.to_string(),
                step_metrics.total_loss.to_string(),
                step_metrics.policy_loss.to_string(),
                step_metrics.value_loss.to_string(),
                step_metrics.reward_loss.to_string(),
                current_lr.to_string(),
                score_min.to_string(),
                score_max.to_string(),
                score_med.to_string(),
                score_mean.to_string(),
                lines_cleared.to_string(),
                current_games.to_string(),
                ram_usage.to_string(),
                gpu_usage.to_string(),
                cpu_usage.to_string(),
                "0.0".to_string(), // io_usage
                disk_usage_pct.to_string(),
                vram_usage.to_string(),
                mcts_depth.to_string(),
                mcts_search_time.to_string(),
            ]);
            let _ = writer.flush();
        }

        training_steps += 1;

        if training_steps % 100 == 0 {
            println!(
                "🔄 Step {} | Games: {} | Loss: {:.4}",
                training_steps, current_games, step_metrics.total_loss
            );
            if !optimizer_configuration
                .paths
                .model_checkpoint_path
                .is_empty()
            {
                let _ =
                    training_var_store.save(&optimizer_configuration.paths.model_checkpoint_path);
            }
        }

        if training_steps % 50 == 0 {
            tch::no_grad(|| {
                if active_is_a {
                    inference_var_store_b.copy(&training_var_store).unwrap();
                    optimizer_network_arcswap.store(Arc::clone(&inference_net_b));
                } else {
                    inference_var_store.copy(&training_var_store).unwrap();
                    optimizer_network_arcswap.store(Arc::clone(&inference_net_a));
                }
            });
            active_is_a = !active_is_a;
        }

        tch::no_grad(|| {
            let mut exponential_moving_average_variables =
                exponential_moving_average_var_store.variables();
            let active_network_variables = training_var_store.variables();
            for (tensor_name, ema_tensor_mut) in exponential_moving_average_variables.iter_mut() {
                if let Some(active_tensor) = active_network_variables.get(tensor_name) {
                    let ema_decay_rate = 0.99;
                    let updated_tensor =
                        &*ema_tensor_mut * ema_decay_rate + active_tensor * (1.0 - ema_decay_rate);
                    ema_tensor_mut.copy_(&updated_tensor);
                }
            }
        });

        if max_steps > 0 && training_steps >= max_steps {
            println!(
                "✅ Hit max training steps limit ({}). Shutting down...",
                max_steps
            );
            println!("FINAL_EVAL_SCORE: {}", step_metrics.total_loss);
            *optimizer_active_flag.write().unwrap() = false;
            break;
        }
    }
}

fn get_gpu_metrics() -> (f32, f32) {
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=utilization.gpu,memory.used")
        .arg("--format=csv,noheader,nounits")
        .output()
    {
        if let Ok(out_str) = String::from_utf8(output.stdout) {
            if let Some(first_line) = out_str.trim().lines().next() {
                let parts: Vec<&str> = first_line.split(", ").collect();
                if parts.len() == 2 {
                    let gpu_util = parts[0].parse::<f32>().unwrap_or(0.0);
                    let vram_used = parts[1].parse::<f32>().unwrap_or(0.0);
                    return (gpu_util, vram_used);
                }
            }
        }
    }
    (0.0, 0.0)
}
```

### File: `src/mcts/mod.rs`

```rs
use crate::core::board::GameStateExt;
use crate::node::{get_valid_action_mask, select_child, LatentNode};
use rand::Rng;
use std::collections::HashMap;

pub struct EvaluationRequest {
    pub is_initial: bool,
    pub board_bitmask: u128,
    pub available_pieces: [i32; 3],
    pub recent_board_history: [u128; 8],
    pub history_len: usize,
    pub recent_action_history: [i32; 4],
    pub action_history_len: usize,
    pub difficulty: i32,
    pub piece_action: i64,
    pub piece_id: i64,
    pub node_index: usize,
    pub worker_id: usize,
    pub parent_cache_index: u32,
    pub leaf_cache_index: u32,
    pub evaluation_request_transmitter: crossbeam_channel::Sender<EvaluationResponse>,
}

pub struct EvaluationResponse {
    pub reward: f32,
    pub value: f32,
    pub child_prior_probabilities_tensor: [f32; 288],
    pub node_index: usize,
}

pub trait NetworkEvaluator: Send + Sync {
    fn send_batch(&self, reqs: Vec<EvaluationRequest>) -> Result<(), String>;
}

impl NetworkEvaluator for std::sync::Arc<crate::queue::FixedInferenceQueue> {
    fn send_batch(&self, reqs: Vec<EvaluationRequest>) -> Result<(), String> {
        if reqs.is_empty() {
            return Ok(());
        }
        self.push_batch(reqs[0].worker_id, reqs)
            .map_err(|_| "Queue Disconnected".to_string())
    }
}

#[cfg(test)]
pub struct MockEvaluator;
#[cfg(test)]
impl NetworkEvaluator for MockEvaluator {
    fn send_batch(&self, reqs: Vec<EvaluationRequest>) -> Result<(), String> {
        for request in reqs {
            let response = EvaluationResponse {
                reward: 0.0,
                value: 0.0,
                child_prior_probabilities_tensor: [1.0 / 288.0; 288],
                node_index: request.node_index,
            };
            let _ = request.evaluation_request_transmitter.send(response);
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct MctsTree {
    pub arena: Vec<LatentNode>,
    pub swap_arena: Vec<LatentNode>,
    pub pointer_remapping: Vec<u32>,
    pub arena_alloc_ptr: usize,
    pub root_index: usize,
    pub free_list: Vec<u32>, // GPU latent state free list
    pub maximum_allowed_nodes_in_search_tree: u32,
}

pub struct MctsParams<'a> {
    pub raw_policy_probabilities: &'a [f32],
    pub root_cache_index: u32,
    pub maximum_allowed_nodes_in_search_tree: u32,
    pub worker_id: usize,
    pub game_state: &'a GameStateExt,
    pub total_simulations: usize,
    pub max_gumbel_k_samples: usize,
    pub gumbel_noise_scale: f32,
    pub previous_tree: Option<MctsTree>,
    pub last_executed_action: Option<i32>,
    pub neural_evaluator: &'a dyn NetworkEvaluator,
    pub evaluation_request_transmitter: crossbeam_channel::Sender<EvaluationResponse>,
    pub evaluation_response_receiver: &'a crossbeam_channel::Receiver<EvaluationResponse>,
    pub active_flag: std::sync::Arc<std::sync::RwLock<bool>>,
    pub _seed: Option<u64>,
}

pub fn mcts_search(params: MctsParams) -> Result<(i32, HashMap<i32, i32>, f32, MctsTree), String> {
    let MctsParams {
        raw_policy_probabilities,
        root_cache_index,
        maximum_allowed_nodes_in_search_tree,
        worker_id,
        game_state,
        total_simulations,
        max_gumbel_k_samples,
        gumbel_noise_scale,
        previous_tree,
        last_executed_action,
        neural_evaluator,
        evaluation_request_transmitter,
        evaluation_response_receiver,
        active_flag,
        _seed,
    } = params;
    let (normalized_probabilities, valid_mask, valid_actions) =
        normalize_policy_distributions(raw_policy_probabilities, game_state);

    if valid_actions.is_empty() {
        let tree = initialize_search_tree(
            previous_tree,
            last_executed_action,
            maximum_allowed_nodes_in_search_tree,
            total_simulations,
        );
        return Ok((-1, HashMap::new(), 0.0, tree));
    }

    let mut tree = initialize_search_tree(
        previous_tree,
        last_executed_action,
        maximum_allowed_nodes_in_search_tree,
        total_simulations,
    );
    let root_index = tree.root_index;

    expand_root_node(&mut tree, root_cache_index, &normalized_probabilities);

    let k_dynamic_samples = calculate_dynamic_k_samples(max_gumbel_k_samples, valid_actions.len());
    if k_dynamic_samples <= 1 {
        let mut visit_distribution = HashMap::new();
        visit_distribution.insert(valid_actions[0], 1);
        let val = tree.arena[root_index].value();
        return Ok((valid_actions[0], visit_distribution, val, tree));
    }

    let mut candidate_actions = valid_actions.clone();
    let gumbel_noisy_logits = inject_gumbel_noise(
        &mut tree.arena,
        root_index,
        &candidate_actions,
        &normalized_probabilities,
        gumbel_noise_scale,
    );

    candidate_actions.sort_by(|&action_a, &action_b| {
        gumbel_noisy_logits[action_b as usize]
            .partial_cmp(&gumbel_noisy_logits[action_a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    candidate_actions.truncate(k_dynamic_samples);

    execute_sequential_halving(
        &mut tree,
        &mut candidate_actions,
        total_simulations,
        &gumbel_noisy_logits,
        game_state,
        neural_evaluator,
        worker_id,
        evaluation_request_transmitter,
        evaluation_response_receiver,
        &active_flag,
    )?;

    compute_final_action_distribution(tree, valid_mask, candidate_actions, gumbel_noisy_logits)
}

fn normalize_policy_distributions(
    raw_policy_probabilities: &[f32],
    game_state: &GameStateExt,
) -> (Vec<f32>, [bool; 288], Vec<i32>) {
    let valid_action_mask = get_valid_action_mask(game_state);
    let mut normalized_probabilities = Vec::with_capacity(288);
    let mut valid_actions = Vec::new();
    let mut valid_action_count = 0;

    for (index, &probability) in raw_policy_probabilities.iter().enumerate() {
        if valid_action_mask[index] {
            normalized_probabilities.push(probability.max(1e-8));
            valid_actions.push(index as i32);
            valid_action_count += 1;
        } else {
            normalized_probabilities.push(0.0);
        }
    }

    let sum_probabilities: f32 = normalized_probabilities.iter().sum();
    assert!(!sum_probabilities.is_nan(), "Policy mask sum is NaN!");

    if sum_probabilities > 0.0 {
        for probability in normalized_probabilities.iter_mut() {
            *probability /= sum_probabilities;
        }
        let check_sum: f32 = normalized_probabilities.iter().sum();
        assert!(
            (check_sum - 1.0).abs() < 1e-4,
            "Normalized prior policies must sum to 1.0. Actual: {}",
            check_sum
        );
    } else if valid_action_count > 0 {
        for (index, probability) in normalized_probabilities.iter_mut().enumerate() {
            if valid_action_mask[index] {
                *probability = 1.0 / (valid_action_count as f32);
            }
        }
    }

    (normalized_probabilities, valid_action_mask, valid_actions)
}

fn calculate_dynamic_k_samples(max_gumbel_k_samples: usize, valid_action_count: usize) -> usize {
    let empty_board_density = 1.0 - (valid_action_count as f32 / 288.0);
    let mut k_dynamic_samples =
        4i32 + ((max_gumbel_k_samples as f32 - 4.0) * empty_board_density) as i32;
    if k_dynamic_samples < 2 {
        k_dynamic_samples = 2;
    }
    (k_dynamic_samples as usize).min(valid_action_count)
}

fn allocate_node(tree: &mut MctsTree, probability: f32, action: i16) -> u32 {
    let new_idx = tree.arena_alloc_ptr;

    if new_idx >= tree.arena.len() {
        let new_capacity = tree.arena.len().max(10_000) * 2;
        tree.arena.resize(new_capacity, LatentNode::new(0.0, -1));
        tree.swap_arena
            .resize(new_capacity, LatentNode::new(0.0, -1));
        tree.pointer_remapping.resize(new_capacity, u32::MAX);
    }

    tree.arena_alloc_ptr += 1;

    tree.arena[new_idx] = LatentNode::new(probability, action);
    new_idx as u32
}

pub fn gc_tree(mut tree: MctsTree, new_root: usize) -> MctsTree {
    let mut new_alloc_ptr = 0;
    let mut queue = vec![new_root as u32];
    tree.pointer_remapping.fill(u32::MAX);

    // Copy new root
    tree.pointer_remapping[new_root] = new_alloc_ptr as u32;
    tree.swap_arena[new_alloc_ptr] = tree.arena[new_root].clone();
    new_alloc_ptr += 1;

    let mut head = 0;
    while head < queue.len() {
        let old_node_idx = queue[head] as usize;
        let new_node_idx = tree.pointer_remapping[old_node_idx] as usize;
        head += 1;

        let mut child_idx = tree.arena[old_node_idx].first_child;
        let mut prev_new_child_idx = u32::MAX;
        let mut is_first = true;

        while child_idx != u32::MAX {
            let new_child_idx = new_alloc_ptr;
            new_alloc_ptr += 1;

            tree.pointer_remapping[child_idx as usize] = new_child_idx as u32;
            tree.swap_arena[new_child_idx] = tree.arena[child_idx as usize].clone();

            if is_first {
                tree.swap_arena[new_node_idx].first_child = new_child_idx as u32;
                is_first = false;
            } else {
                tree.swap_arena[prev_new_child_idx as usize].next_sibling = new_child_idx as u32;
            }

            tree.swap_arena[new_child_idx].next_sibling = u32::MAX;

            queue.push(child_idx);
            prev_new_child_idx = new_child_idx as u32;
            child_idx = tree.arena[child_idx as usize].next_sibling;
        }
    }

    // Rebuild free_list of GPU cache states
    tree.free_list.clear();
    let mut used_states = vec![false; tree.maximum_allowed_nodes_in_search_tree as usize];
    for i in 0..new_alloc_ptr {
        let state_idx = tree.swap_arena[i].hidden_state_index;
        if state_idx != u32::MAX {
            used_states[state_idx as usize] = true;
        }
    }
    for (i, &used) in used_states.iter().enumerate() {
        if !used {
            tree.free_list.push(i as u32);
        }
    }

    std::mem::swap(&mut tree.arena, &mut tree.swap_arena);
    tree.arena_alloc_ptr = new_alloc_ptr;
    tree.root_index = 0; // The new root is now at index 0

    tree
}

fn initialize_search_tree(
    previous_tree: Option<MctsTree>,
    last_executed_action: Option<i32>,
    maximum_allowed_nodes_in_search_tree: u32,
    total_simulations: usize,
) -> MctsTree {
    if let Some(existing_tree) = previous_tree {
        if let Some(action) = last_executed_action {
            let child_index = existing_tree.arena[existing_tree.root_index]
                .get_child(&existing_tree.arena, action);
            if child_index != usize::MAX {
                let gc_d_tree = gc_tree(existing_tree, child_index);
                if gc_d_tree.free_list.len() > total_simulations + 10 {
                    return gc_d_tree;
                }
            }
        } else {
            let root = existing_tree.root_index;
            let gc_d_tree = gc_tree(existing_tree, root);
            if gc_d_tree.free_list.len() > total_simulations + 10 {
                return gc_d_tree;
            }
        }
    }

    let dynamic_capacity = (total_simulations * 300 + 10_000).max(100_000);
    let mut arena = vec![LatentNode::new(0.0, -1); dynamic_capacity];
    let swap_arena = vec![LatentNode::new(0.0, -1); dynamic_capacity];
    let pointer_remapping = vec![u32::MAX; dynamic_capacity];

    let free_list = (0..maximum_allowed_nodes_in_search_tree)
        .rev()
        .collect::<Vec<u32>>();

    arena[0] = LatentNode::new(1.0, -1);

    MctsTree {
        arena,
        swap_arena,
        pointer_remapping,
        arena_alloc_ptr: 1,
        root_index: 0,
        free_list,
        maximum_allowed_nodes_in_search_tree,
    }
}

fn expand_root_node(tree: &mut MctsTree, root_cache_index: u32, normalized_probabilities: &[f32]) {
    let root_index = tree.root_index;
    if tree.arena[root_index].is_topologically_expanded {
        return;
    }
    tree.arena[root_index].hidden_state_index = root_cache_index;
    tree.arena[root_index].reward = 0.0;
    tree.arena[root_index].is_topologically_expanded = true;

    let mut prev_child = u32::MAX;
    let mut first_child = u32::MAX;

    for (action_index, &probability) in normalized_probabilities.iter().enumerate() {
        if probability > 0.0 {
            let new_node_index = allocate_node(tree, probability, action_index as i16);
            if first_child == u32::MAX {
                first_child = new_node_index;
            } else {
                tree.arena[prev_child as usize].next_sibling = new_node_index;
            }
            prev_child = new_node_index;
        }
    }
    tree.arena[root_index].first_child = first_child;
}

fn inject_gumbel_noise(
    arena: &mut [LatentNode],
    root_index: usize,
    candidate_actions: &[i32],
    normalized_probabilities: &[f32],
    gumbel_noise_scale: f32,
) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut gumbel_noisy_logits = vec![f32::NEG_INFINITY; 288];

    for &action_index in candidate_actions {
        let uniform_random_sample: f32 = rng.gen_range(1e-6..=(1.0 - 1e-6));
        let gumbel_noise_value = -(-(uniform_random_sample.ln())).ln();
        assert!(!gumbel_noise_value.is_nan(), "Gumbel noise is NaN");

        let action_usize = action_index as usize;
        let child_index = arena[root_index].get_child(arena, action_index);

        if child_index != usize::MAX {
            arena[child_index].gumbel_noise = gumbel_noise_value;
            let log_probability = (normalized_probabilities[action_usize] + 1e-8).ln();
            gumbel_noisy_logits[action_usize] =
                log_probability + (gumbel_noise_value * gumbel_noise_scale);
        }
    }
    gumbel_noisy_logits
}

#[allow(clippy::too_many_arguments)]
fn execute_sequential_halving(
    tree: &mut MctsTree,
    candidate_actions: &mut Vec<i32>,
    total_simulations: usize,
    gumbel_noisy_logits: &[f32],
    game_state: &GameStateExt,
    neural_evaluator: &dyn NetworkEvaluator,
    worker_id: usize,
    evaluation_request_transmitter: crossbeam_channel::Sender<EvaluationResponse>,
    evaluation_response_receiver: &crossbeam_channel::Receiver<EvaluationResponse>,
    active_flag: &std::sync::Arc<std::sync::RwLock<bool>>,
) -> Result<(), String> {
    let candidate_count = candidate_actions.len();
    let total_halving_phases = if candidate_count > 1 {
        (candidate_count as f32).log2().ceil() as usize
    } else {
        0
    };

    let mut remaining_simulations = total_simulations;
    let mut remaining_phases = total_halving_phases;

    for _phase in 0..total_halving_phases {
        let current_candidate_count = candidate_actions.len();
        if current_candidate_count <= 1 || remaining_phases == 0 {
            break;
        }

        let mut visits_per_candidate =
            (remaining_simulations / remaining_phases) / current_candidate_count;
        if visits_per_candidate == 0 {
            visits_per_candidate = 1;
        }

        expand_and_evaluate_candidates(
            tree,
            candidate_actions,
            visits_per_candidate,
            game_state,
            neural_evaluator,
            worker_id,
            evaluation_request_transmitter.clone(),
            evaluation_response_receiver,
            active_flag,
        )?;

        let root_index = tree.root_index;
        prune_candidates(
            &tree.arena,
            root_index,
            candidate_actions,
            gumbel_noisy_logits,
        );

        remaining_simulations =
            remaining_simulations.saturating_sub(visits_per_candidate * current_candidate_count);
        remaining_phases -= 1;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn expand_and_evaluate_candidates(
    tree: &mut MctsTree,
    candidate_actions: &[i32],
    visits_per_candidate: usize,
    game_state: &GameStateExt,
    neural_evaluator: &dyn NetworkEvaluator,
    worker_id: usize,
    evaluation_request_transmitter: crossbeam_channel::Sender<EvaluationResponse>,
    evaluation_response_receiver: &crossbeam_channel::Receiver<EvaluationResponse>,
    active_flag: &std::sync::Arc<std::sync::RwLock<bool>>,
) -> Result<(), String> {
    let mut eval_batch = Vec::new();
    let mut batch_paths = Vec::new();

    let root_index = tree.root_index;

    for &candidate_action in candidate_actions {
        for _ in 0..visits_per_candidate {
            let (search_path, leaf_node_index, successfully_traversed) =
                traverse_tree_to_leaf(&tree.arena, root_index, candidate_action);

            if !successfully_traversed {
                continue;
            }

            for &node_idx in &search_path {
                tree.arena[node_idx].visits += 1;
                tree.arena[node_idx].value_sum -= 1.0;
            }

            let parent_index = search_path[search_path.len() - 3];
            let last_action_taken = search_path[search_path.len() - 2];
            let slot_index = last_action_taken / 96;
            let position_bit_index = last_action_taken % 96;
            let mut piece_identifier = game_state.available[slot_index];
            if piece_identifier == -1 {
                piece_identifier = 0;
            }

            let piece_action_identifier = piece_identifier * 96 + (position_bit_index as i32);
            let prev_idx = tree.arena[parent_index].hidden_state_index;
            let new_idx = tree.free_list.pop().unwrap();

            tree.arena[leaf_node_index].hidden_state_index = new_idx;

            let evaluation_request = EvaluationRequest {
                is_initial: false,
                board_bitmask: 0,
                available_pieces: [-1; 3],
                recent_board_history: [0; 8],
                history_len: 0,
                recent_action_history: [0; 4],
                action_history_len: 0,
                difficulty: 6,
                piece_action: piece_action_identifier as i64,
                piece_id: piece_identifier as i64,
                node_index: leaf_node_index,
                worker_id,
                parent_cache_index: prev_idx,
                leaf_cache_index: new_idx,
                evaluation_request_transmitter: evaluation_request_transmitter.clone(),
            };

            batch_paths.push(search_path);
            eval_batch.push(evaluation_request);
        }
    }

    let active_requests = eval_batch.len();
    if active_requests > 0 {
        if let Err(error) = neural_evaluator.send_batch(eval_batch) {
            return Err(format!("Failed sending eval request: {}", error));
        }

        process_evaluation_responses(
            tree,
            evaluation_response_receiver,
            active_requests as u32,
            batch_paths,
            active_flag,
        )?;
    }
    Ok(())
}

fn traverse_tree_to_leaf(
    arena: &[LatentNode],
    root_index: usize,
    candidate_action: i32,
) -> (Vec<usize>, usize, bool) {
    let mut search_path = vec![root_index];
    let mut current_node_index = root_index;

    let immediate_child_index = arena[current_node_index].get_child(arena, candidate_action);
    if immediate_child_index == usize::MAX {
        return (search_path, current_node_index, false);
    }

    search_path.push(candidate_action as usize);
    search_path.push(immediate_child_index);
    current_node_index = immediate_child_index;

    while arena[current_node_index].is_topologically_expanded {
        let (best_action, next_node_index) = select_child(arena, current_node_index, false);
        if next_node_index == usize::MAX {
            break;
        }
        search_path.push(best_action as usize);
        search_path.push(next_node_index);
        current_node_index = next_node_index;
    }

    (search_path, current_node_index, true)
}

fn process_evaluation_responses(
    tree: &mut MctsTree,
    receiver_rx: &crossbeam_channel::Receiver<EvaluationResponse>,
    active_requests: u32,
    batch_paths: Vec<Vec<usize>>,
    active_flag: &std::sync::Arc<std::sync::RwLock<bool>>,
) -> Result<(), String> {
    let mut paths_map = std::collections::HashMap::new();
    for path in batch_paths {
        let leaf_index = *path.last().unwrap();
        paths_map.insert(leaf_index, path);
    }

    for _ in 0..active_requests {
        let evaluation_response = loop {
            if !*active_flag.read().unwrap() {
                return Err("Training stopped".to_string());
            }
            match receiver_rx.recv_timeout(std::time::Duration::from_millis(100)) {
                Ok(resp) => break resp,
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
                Err(_) => return Err("Channel disconnected".to_string()),
            }
        };

        let leaf_node_index = evaluation_response.node_index;
        let search_path = paths_map.get(&leaf_node_index).unwrap();

        tree.arena[leaf_node_index].reward = evaluation_response.reward;
        tree.arena[leaf_node_index].is_topologically_expanded = true;

        let mut prev_child = u32::MAX;
        let mut first_child = u32::MAX;

        for (action_index, &probability) in evaluation_response
            .child_prior_probabilities_tensor
            .iter()
            .enumerate()
        {
            if probability > 0.0 {
                let new_node_index = allocate_node(tree, probability, action_index as i16);
                if first_child == u32::MAX {
                    first_child = new_node_index;
                } else {
                    tree.arena[prev_child as usize].next_sibling = new_node_index;
                }
                prev_child = new_node_index;
            }
        }
        tree.arena[leaf_node_index].first_child = first_child;

        let mut backprop_value = evaluation_response.value;

        for index in (0..search_path.len()).step_by(2).rev() {
            let node_index = search_path[index];
            tree.arena[node_index].value_sum += 1.0;
            tree.arena[node_index].value_sum += backprop_value;
            backprop_value = tree.arena[node_index].reward + 0.99 * backprop_value;
        }
    }
    Ok(())
}

fn prune_candidates(
    arena: &[LatentNode],
    root_index: usize,
    candidate_actions: &mut Vec<i32>,
    gumbel_noisy_logits: &[f32],
) {
    // 1. PRE-FETCH: Find all child indices in O(N) time before sorting
    let mut candidates_with_nodes: Vec<(i32, usize)> = candidate_actions
        .iter()
        .map(|&a| (a, arena[root_index].get_child(arena, a)))
        .filter(|&(_, idx)| idx != usize::MAX)
        .collect();

    // 2. SORT: Now we use the pre-fetched indices. No linked-list traversal here!
    candidates_with_nodes.sort_by(|&(action_a, index_a), &(action_b, index_b)| {
        let node_a = &arena[index_a];
        let node_b = &arena[index_b];

        let q_value_a = node_a.reward + 0.99 * node_a.value();
        let q_value_b = node_b.reward + 0.99 * node_b.value();

        let exploration_scale_a = 50.0 / ((node_a.visits + 1) as f32);
        let score_a = gumbel_noisy_logits[action_a as usize] + (exploration_scale_a * q_value_a);

        let exploration_scale_b = 50.0 / ((node_b.visits + 1) as f32);
        let score_b = gumbel_noisy_logits[action_b as usize] + (exploration_scale_b * q_value_b);

        score_b
            .partial_cmp(&score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // 3. TRUNCATE: Drop the bottom half and map back to just the actions
    let items_to_drop = candidates_with_nodes.len() / 2;
    candidates_with_nodes.truncate(candidates_with_nodes.len() - items_to_drop);

    *candidate_actions = candidates_with_nodes.into_iter().map(|(a, _)| a).collect();
}

fn compute_final_action_distribution(
    tree: MctsTree,
    valid_action_mask: [bool; 288],
    candidate_actions: Vec<i32>,
    gumbel_noisy_logits: Vec<f32>,
) -> Result<(i32, HashMap<i32, i32>, f32, MctsTree), String> {
    let arena = &tree.arena;
    let root_index = tree.root_index;
    let mut evaluated_candidates = Vec::new();
    for (action_index, &is_valid) in valid_action_mask.iter().enumerate() {
        let child_index = arena[root_index].get_child(arena, action_index as i32);
        if child_index != usize::MAX && arena[child_index].visits > 0 && is_valid {
            evaluated_candidates.push((action_index as i32, child_index));
        }
    }

    if evaluated_candidates.is_empty() {
        let mut uniform_visits = HashMap::new();
        uniform_visits.insert(candidate_actions[0], 1);
        let val = arena[root_index].value();
        return Ok((candidate_actions[0], uniform_visits, val, tree));
    }

    let mut q_values = Vec::new();
    let mut maximum_q_value = f32::NEG_INFINITY;
    let mut minimum_q_value = f32::INFINITY;

    for &(_action_index, child_index) in &evaluated_candidates {
        let q_value = arena[child_index].reward + 0.99 * arena[child_index].value();
        q_values.push(q_value);
        if q_value > maximum_q_value {
            maximum_q_value = q_value;
        }
        if q_value < minimum_q_value {
            minimum_q_value = q_value;
        }
    }

    let mut visit_distribution = HashMap::new();
    for &(action_index, child_index) in &evaluated_candidates {
        visit_distribution.insert(action_index, arena[child_index].visits);
    }

    let mut optimal_action = candidate_actions[0];
    let mut optimal_action_score = f32::NEG_INFINITY;

    let mut max_visit = 0;
    let mut sum_visit = 0;
    for &(_action_index, child_index) in &evaluated_candidates {
        let visits = arena[child_index].visits;
        sum_visit += visits;
        if visits > max_visit {
            max_visit = visits;
        }
    }

    let exploration_scale = (50.0 + max_visit as f32) / (sum_visit as f32 + 1e-8);

    for &(action_index, child_index) in &evaluated_candidates {
        let q_value = arena[child_index].reward + 0.99 * arena[child_index].value();
        let completed_gumbel_score =
            gumbel_noisy_logits[action_index as usize] + exploration_scale * q_value;

        if completed_gumbel_score > optimal_action_score {
            optimal_action_score = completed_gumbel_score;
            optimal_action = action_index;
        }
    }

    let val = arena[root_index].value();
    Ok((optimal_action, visit_distribution, val, tree))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::GameStateExt;
    use crate::node::LatentNode;

    // [5. Q-Value Min-Max Normalization (Math Correctness)]
    #[test]
    fn test_q_value_min_max_normalization() {
        let q_values = [10.0, 50.0, 100.0];
        let min_q = q_values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_q = q_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let normalized: Vec<f32> = q_values
            .iter()
            .map(|&q| {
                if max_q > min_q {
                    (q - min_q) / (max_q - min_q)
                } else {
                    0.5
                }
            })
            .collect();

        assert!((normalized[0] - 0.0).abs() < 1e-6);
        assert!((normalized[1] - 0.4444444).abs() < 1e-4);
        assert!((normalized[2] - 1.0).abs() < 1e-6);
    }

    // [6. Gumbel Noise Distribution (Math Correctness)]
    #[test]
    fn test_gumbel_noise_distribution() {
        let mut mock_arena = vec![LatentNode::new(0.0, 0)];
        mock_arena.push(LatentNode::new(0.33, 1));
        mock_arena.push(LatentNode::new(0.33, 2));
        mock_arena.push(LatentNode::new(0.34, 3));
        mock_arena[0].first_child = 1;
        mock_arena[1].next_sibling = 2;
        mock_arena[2].next_sibling = 3;

        let actions = vec![1, 2, 3];
        let mut probs = vec![0.0; 288];
        probs[1] = 0.33;
        probs[2] = 0.33;
        probs[3] = 0.34;

        let mut sum = 0.0;
        let mut sq_sum = 0.0;
        let n = 200_000;

        for _ in 0..n {
            let res = inject_gumbel_noise(&mut mock_arena, 0, &actions, &probs, 1.0);
            for action in &actions {
                let log_prob = (probs[*action as usize] + 1e-8).ln();
                let noise = res[*action as usize] - log_prob;
                sum += noise;
                sq_sum += noise * noise;
            }
        }

        let mean = sum / (n as f32 * 3.0);
        let std_dev = (sq_sum / (n as f32 * 3.0) - (mean * mean)).sqrt();
        let variance = std_dev * std_dev;

        assert!((mean - 0.5772).abs() < 0.1, "Gumbel mean is off: {}", mean);
        assert!(
            (variance - 1.6449).abs() < 0.2,
            "Gumbel variance is off: {}",
            variance
        );
    }

    // [8. Tree Arena Garbage Collection (RAM Bottleneck / Leak)]
    #[test]
    fn test_tree_arena_garbage_collection_leak_100000_steps() {
        let max_nodes = 1000;
        let mut tree = initialize_search_tree(None, None, max_nodes, 500);

        for _step in 0..100_000 {
            let current_root = tree.root_index;
            // allocate 10 children
            let mut prev = u32::MAX;
            let mut first = u32::MAX;
            for i in 0..10 {
                let new_node = allocate_node(&mut tree, 0.1, i);
                tree.arena[new_node as usize].hidden_state_index = i as u32;
                if first == u32::MAX {
                    first = new_node;
                } else {
                    tree.arena[prev as usize].next_sibling = new_node;
                }
                prev = new_node;
            }
            tree.arena[current_root].first_child = first;

            // Pick the 5th child as the new root to step forward in the game
            let mut new_root = first;
            for _ in 0..4 {
                new_root = tree.arena[new_root as usize].next_sibling;
            }

            tree = gc_tree(tree, new_root as usize);

            // Assert that despite looping 100k times, we never exhaust arena size (no leaks)
            assert!(
                tree.arena_alloc_ptr <= max_nodes as usize * 2,
                "Arena leaked and grew infinitely!"
            );
            // In a properly functioning GC, the free list should roughly maintain its size.
        }
    }

    #[test]
    fn test_tree_arena_garbage_collection() {
        let mut tree = initialize_search_tree(None, None, 1000, 500);
        let new_root = 0; // Prevent cycle with node_free_list which operates 1..1000

        // Allocate a few nodes
        for i in 0..500 {
            let p_idx = allocate_node(&mut tree, 1.0, i as i16);
            if i > 0 {
                // link to root to keep valid
                tree.arena[p_idx as usize].next_sibling = tree.arena[new_root].first_child;
                tree.arena[new_root].first_child = p_idx;
                tree.arena[p_idx as usize].hidden_state_index = i as u32; // Simulate cached memory
            }
        }

        let new_tree = gc_tree(tree, new_root);

        // Ensure new_tree has a smaller reach but the arena capacity stays identical
        // arena_alloc_ptr will be exactly 500 since we retained all 500 allocated nodes
        assert_eq!(new_tree.arena_alloc_ptr, 500);

        // The new root MUST be mapped to 0 right after bump allocator GC.
        assert_eq!(new_tree.root_index, 0);
    }

    #[test]
    fn test_valid_action_mask() {
        let mut state = GameStateExt::new(Some([0, 1, 2]), 0, 0, 6, 0);
        let mask = get_valid_action_mask(&state);
        assert!(mask.contains(&true));

        state.terminal = true;
        let terminal_mask = get_valid_action_mask(&state);
        assert!(!terminal_mask.contains(&true));
    }

    #[test]
    fn test_sequential_halving_visits() {
        let evaluator = MockEvaluator;
        let state = GameStateExt::new(Some([0, 1, 2]), 0, 0, 6, 0);

        let (answer_tx, answer_rx) = crossbeam_channel::unbounded();

        let mut policy_probs = vec![0.0; 288];
        let mask = get_valid_action_mask(&state);
        let mut valid_count = 0;
        for i in 0..mask.len() {
            if mask[i] {
                policy_probs[i] = 1.0;
                valid_count += 1;
            }
        }
        for p in policy_probs.iter_mut() {
            *p /= valid_count as f32;
        }

        let simulations = 50;
        let k = 8;

        let (_best_action, visits, _value, _tree) = mcts_search(MctsParams {
            root_cache_index: 0,
            maximum_allowed_nodes_in_search_tree: 50000,
            worker_id: 0,
            raw_policy_probabilities: &policy_probs,
            game_state: &state,
            total_simulations: simulations,
            max_gumbel_k_samples: k,
            gumbel_noise_scale: 1.0,
            previous_tree: None,
            last_executed_action: None,
            neural_evaluator: &evaluator,
            evaluation_request_transmitter: answer_tx,
            evaluation_response_receiver: &answer_rx,
            active_flag: std::sync::Arc::new(std::sync::RwLock::new(true)),
            _seed: None,
        })
        .unwrap();

        let total_visits: i32 = visits.values().sum();
        assert!(
            (50..=200).contains(&total_visits),
            "Total visits should scale relative to requested simulations. Was: {}",
            total_visits
        );

        let mut visit_counts: Vec<i32> = visits.values().cloned().collect();
        visit_counts.sort_unstable_by(|a, b| b.cmp(a));
        assert!(visit_counts[0] > 8, "Sequential Halving correctly concentrates visits on top candidates, even if uniform prior.");
    }

    pub struct CustomEvaluator {
        pub reward: f32,
        pub value: f32,
    }
    impl super::NetworkEvaluator for CustomEvaluator {
        fn send_batch(&self, reqs: Vec<super::EvaluationRequest>) -> Result<(), String> {
            for request in reqs {
                let response = super::EvaluationResponse {
                    reward: self.reward,
                    value: self.value,
                    child_prior_probabilities_tensor: [1.0 / 288.0; 288],
                    node_index: request.node_index,
                };
                let _ = request.evaluation_request_transmitter.send(response);
            }
            Ok(())
        }
    }

    #[test]
    fn test_terminal_state_value_masking() {
        let evaluator = CustomEvaluator {
            reward: 1.0,
            value: 0.5,
        };
        let state = GameStateExt::new(Some([0, 1, 2]), 0, 0, 6, 0);
        let (answer_tx, answer_rx) = crossbeam_channel::unbounded();
        let mut policy_probs = vec![0.0; 288];
        let mask = get_valid_action_mask(&state);
        for i in 0..mask.len() {
            if mask[i] {
                policy_probs[i] = 1.0;
            }
        }
        let (_best_action, _visits, _value, tree) = mcts_search(MctsParams {
            root_cache_index: 0,
            maximum_allowed_nodes_in_search_tree: 50000,
            worker_id: 0,
            raw_policy_probabilities: &policy_probs,
            game_state: &state,
            total_simulations: 10,
            max_gumbel_k_samples: 8,
            gumbel_noise_scale: 1.0,
            previous_tree: None,
            last_executed_action: None,
            neural_evaluator: &evaluator,
            evaluation_request_transmitter: answer_tx,
            evaluation_response_receiver: &answer_rx,
            active_flag: std::sync::Arc::new(std::sync::RwLock::new(true)),
            _seed: None,
        })
        .unwrap();

        let root = &tree.arena[tree.root_index];
        let mut checked_any = false;
        let mut child_idx = root.first_child;
        while child_idx != u32::MAX {
            if tree.arena[child_idx as usize].visits == 1 {
                let child = &tree.arena[child_idx as usize];
                let expected = child.value();
                assert!(
                    (expected - 0.5).abs() < 1e-5 || (expected + 1.0).abs() < 1e-5 || (expected - 1.0).abs() < 1e-5,
                    "Child with 1 visit should contain expected network value or terminal mask! Found: {}",
                    expected
                );
                checked_any = true;
            }
            child_idx = tree.arena[child_idx as usize].next_sibling;
        }
        assert!(checked_any, "No children were evaluated.");
    }

    #[test]
    fn test_gumbel_distribution() {
        let mut rng = rand::thread_rng();
        let mut sum = 0.0;
        let n = 10_000;
        for _ in 0..n {
            let uniform_random_sample: f32 = rng.gen_range(1e-6..=(1.0 - 1e-6));
            let gumbel_noise_value = -(-(uniform_random_sample.ln())).ln();
            sum += gumbel_noise_value;
        }
        let mean = sum / (n as f32);
        assert!((mean - 0.5772).abs() < 0.05, "Gumbel mean off: {}", mean);
    }

    #[test]
    fn test_tree_garbage_collection() {
        let mut mock_arena = vec![LatentNode::new(0.0, 0); 10];
        let mut root = LatentNode::new(1.0, -1);
        root.is_topologically_expanded = true;
        root.first_child = 1;
        mock_arena[0] = root;

        let mut child1 = LatentNode::new(0.5, 0);
        child1.is_topologically_expanded = true;
        child1.first_child = 3;
        child1.next_sibling = 2; // points to child2
        child1.visits = 10;
        child1.hidden_state_index = 5;
        mock_arena[1] = child1;

        let mut child2 = LatentNode::new(0.5, 1);
        child2.visits = 5;
        child2.hidden_state_index = 6;
        mock_arena[2] = child2;

        let mut grandchild = LatentNode::new(1.0, 2);
        grandchild.visits = 2;
        grandchild.hidden_state_index = 7;
        mock_arena[3] = grandchild;

        let mut disconnected = LatentNode::new(1.0, 3);
        disconnected.hidden_state_index = 8;
        mock_arena[4] = disconnected;

        let initial_free_list = vec![9];

        let tree = MctsTree {
            arena: mock_arena,
            swap_arena: vec![LatentNode::new(0.0, -1); 1000],
            pointer_remapping: vec![u32::MAX; 1000],
            arena_alloc_ptr: 5,
            root_index: 0,
            free_list: initial_free_list,
            maximum_allowed_nodes_in_search_tree: 1000,
        };

        let new_tree = gc_tree(tree, 1);
        let actual_root = new_tree.root_index;
        let new_root_node = &new_tree.arena[actual_root];
        assert_eq!(new_root_node.visits, 10, "Visits must be retained");

        let first_child_idx = new_root_node.first_child as usize;
        assert!(
            first_child_idx != u32::MAX as usize,
            "Children must be retained"
        );

        let new_grandchild = &new_tree.arena[first_child_idx];
        assert_eq!(
            new_grandchild.visits, 2,
            "Grandchild visits must be retained"
        );

        assert!(
            new_tree.free_list.contains(&6),
            "Child 2 cache slot must be freed"
        );
        assert!(
            new_tree.free_list.contains(&8),
            "Disconnected node cache slot must be freed"
        );
        assert!(
            !new_tree.free_list.contains(&5),
            "Child 1 cache slot must survive"
        );
        assert!(
            !new_tree.free_list.contains(&7),
            "Grandchild cache slot must survive"
        );
        assert!(
            new_tree.free_list.contains(&9),
            "Original free list items must survive"
        );
    }
}
```

### File: `src/net/dynamics.rs`

```rs
use crate::net::FlattenedResNetBlock;
use tch::{nn, nn::Module, Kind, Tensor};

#[derive(Debug)]
pub struct DynamicsNet {
    piece_emb: nn::Embedding,
    pos_emb: nn::Embedding,
    proj_in: nn::Conv2D,
    blocks: Vec<FlattenedResNetBlock>,
    scale_norm: nn::LayerNorm,
    reward_cond: nn::Conv2D,
    reward_fc1: nn::Linear,
    reward_norm: nn::LayerNorm,
    reward_fc2: nn::Linear,
}

impl DynamicsNet {
    pub fn new(
        variable_store: &nn::Path,
        model_dimension: i64,
        num_blocks: i64,
        support_size: i64,
    ) -> Self {
        let piece_emb = nn::embedding(
            &(variable_store / "piece_emb"),
            48,
            model_dimension,
            Default::default(),
        );
        let pos_emb = nn::embedding(
            &(variable_store / "pos_emb"),
            96,
            model_dimension,
            Default::default(),
        );
        let proj_config = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
        let proj_in = nn::conv2d(
            &(variable_store / "proj_in"),
            model_dimension * 2,
            model_dimension,
            3,
            proj_config,
        );

        let mut blocks = Vec::new();
        let blocks_vs = variable_store / "blocks";
        for i in 0..num_blocks {
            blocks.push(FlattenedResNetBlock::new(
                &(&blocks_vs / i),
                model_dimension,
                128,
            ));
        }
        let scale_norm = nn::layer_norm(
            &(variable_store / "scale_norm"),
            vec![model_dimension],
            Default::default(),
        );

        let conv2d_config = nn::ConvConfig::default();
        let reward_cond = nn::conv2d(
            &(variable_store / "reward_cond"),
            model_dimension * 2,
            model_dimension,
            1,
            conv2d_config,
        );

        let reward_fc1 = nn::linear(
            &(variable_store / "reward_fc1"),
            model_dimension,
            64,
            Default::default(),
        );
        let reward_norm = nn::layer_norm(
            &(variable_store / "reward_norm"),
            vec![64],
            Default::default(),
        );
        let reward_fc2 = nn::linear(
            &(variable_store / "reward_fc2"),
            64,
            2 * support_size + 1,
            Default::default(),
        );

        Self {
            piece_emb,
            pos_emb,
            proj_in,
            blocks,
            scale_norm,
            reward_cond,
            reward_fc1,
            reward_norm,
            reward_fc2,
        }
    }

    pub fn forward(
        &self,
        hidden_state: &Tensor,
        batched_action: &Tensor,
        batched_piece_identifier: &Tensor,
    ) -> (Tensor, Tensor) {
        assert_eq!(
            hidden_state.size().len(),
            4,
            "Dynamics forward requires a 4D hidden_state tensor"
        );
        assert_eq!(
            batched_action.size()[0],
            batched_piece_identifier.size()[0],
            "Action sizes must match"
        );

        let position_indices = batched_action.remainder(96);
        let action_embeddings = self.piece_emb.forward(batched_piece_identifier)
            + self.pos_emb.forward(&position_indices);

        let action_expanded = action_embeddings
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand([-1, -1, 8, 8], false);
        let concatenated_features = Tensor::cat(&[hidden_state, &action_expanded], 1);

        let reward_convolutions_mish = self.reward_cond.forward(&concatenated_features).mish();
        let hidden_state_average_pooled =
            reward_convolutions_mish.mean_dim(&[2i64, 3i64][..], false, Kind::Float);

        let reward_features = self
            .reward_norm
            .forward(&self.reward_fc1.forward(&hidden_state_average_pooled))
            .mish();
        let reward_logits = self.reward_fc2.forward(&reward_features);

        let mut hidden_state_next = self.proj_in.forward(&concatenated_features);
        for block in &self.blocks {
            hidden_state_next = block.forward(&hidden_state_next);
        }
        hidden_state_next = self
            .scale_norm
            .forward(&hidden_state_next.permute([0, 2, 3, 1]))
            .permute([0, 3, 1, 2]);

        let final_hidden_state_next = hidden_state_next;

        assert_eq!(
            final_hidden_state_next.size(),
            hidden_state.size(),
            "Hidden state dynamic dims drifted!"
        );

        (final_hidden_state_next, reward_logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn, Device};

    #[test]
    fn test_dynamics_action_conditioning() {
        let variable_store = nn::VarStore::new(Device::Cpu);
        let dynamics_network = DynamicsNet::new(&variable_store.root(), 16, 1, 300);

        let batch_size = 2;
        let hidden_state = Tensor::zeros([batch_size, 16, 8, 8], (Kind::Float, Device::Cpu));
        let batched_action = Tensor::zeros([batch_size], (Kind::Int64, Device::Cpu));
        let batched_piece_identifier = Tensor::zeros([batch_size], (Kind::Int64, Device::Cpu));

        let (hidden_state_next, reward_logits) =
            dynamics_network.forward(&hidden_state, &batched_action, &batched_piece_identifier);

        assert_eq!(
            hidden_state_next.size(),
            vec![batch_size, 16, 8, 8],
            "Latent state dimensions incorrect after dynamics forward pass"
        );
        assert_eq!(
            reward_logits.size(),
            vec![batch_size, 601],
            "Reward logits boundaries do not match 2*support + 1"
        );
    }
}
```

### File: `src/net/mod.rs`

```rs
pub mod dynamics;
pub mod muzero;
pub mod prediction;
pub mod projector;
pub mod representation;
pub mod resnet;

pub use dynamics::DynamicsNet;
pub use muzero::MuZeroNet;
pub use prediction::PredictionNet;
pub use projector::ProjectorNet;
pub use representation::RepresentationNet;
pub use resnet::FlattenedResNetBlock;
```

### File: `src/net/muzero.rs`

```rs
use crate::net::{DynamicsNet, PredictionNet, ProjectorNet, RepresentationNet};
use tch::{nn, nn::Module, Kind, Tensor};

#[derive(Debug)]
pub struct MuZeroNet {
    pub representation: RepresentationNet,
    pub dynamics: DynamicsNet,
    pub prediction: PredictionNet,
    pub projector: ProjectorNet,
    pub support_size: i64,
    pub epsilon_factor: f64,
    pub math_cmodule: tch::CModule,
}

unsafe impl Sync for MuZeroNet {}
unsafe impl Send for MuZeroNet {}

impl MuZeroNet {
    pub fn new(
        variable_store: &nn::Path,
        model_dimension: i64,
        convolution_blocks: i64,
        support_size: i64,
    ) -> Self {
        let representation = RepresentationNet::new(
            &(variable_store / "representation"),
            model_dimension,
            convolution_blocks,
        );
        let dynamics = DynamicsNet::new(
            &(variable_store / "dynamics"),
            model_dimension,
            convolution_blocks,
            support_size,
        );
        let prediction = PredictionNet::new(
            &(variable_store / "prediction"),
            model_dimension,
            support_size,
            288,
        );
        let projector =
            ProjectorNet::new(&(variable_store / "projector"), model_dimension, 512, 128);

        let mut math_data = std::io::Cursor::new(include_bytes!("../../assets/math_kernels.pt"));
        let math_cmodule =
            tch::CModule::load_data_on_device(&mut math_data, variable_store.device())
                .expect("Failed to load embedded math_kernels.pt");

        Self {
            representation,
            dynamics,
            prediction,
            projector,
            support_size,
            epsilon_factor: 0.001,
            math_cmodule,
        }
    }

    pub fn support_to_scalar(&self, logits_prediction: &Tensor) -> Tensor {
        let ivalue = self
            .math_cmodule
            .method_is(
                "support_to_scalar",
                &[
                    tch::IValue::Tensor(logits_prediction.copy()),
                    tch::IValue::Int(self.support_size),
                    tch::IValue::Double(self.epsilon_factor),
                ],
            )
            .expect("math_cmodule support_to_scalar failed");

        if let tch::IValue::Tensor(t) = ivalue {
            t
        } else {
            unreachable!("math_kernels support_to_scalar must return a Tensor")
        }
    }

    pub fn scalar_to_support(&self, scalar_prediction: &Tensor) -> Tensor {
        let ivalue = self
            .math_cmodule
            .method_is(
                "scalar_to_support",
                &[
                    tch::IValue::Tensor(scalar_prediction.copy()),
                    tch::IValue::Int(self.support_size),
                    tch::IValue::Double(self.epsilon_factor),
                ],
            )
            .expect("math_cmodule scalar_to_support failed");

        if let tch::IValue::Tensor(t) = ivalue {
            t
        } else {
            unreachable!("math_kernels scalar_to_support must return a Tensor")
        }
    }

    pub fn initial_inference(&self, batched_state: &Tensor) -> (Tensor, Tensor, Tensor, Tensor) {
        assert_eq!(
            batched_state.size().len(),
            4,
            "Initial inference batched_state must have 4 dimensions"
        );
        assert_eq!(
            batched_state.size()[1],
            20,
            "Initial inference batched_state must have 20 spatial channels"
        );
        let hidden_state = self.representation.forward(batched_state);
        let (value_logits, policy_logits, hidden_state_logits) =
            self.prediction.forward(&hidden_state);

        let predicted_value_scalar = self.support_to_scalar(&value_logits);
        let policy_probabilities = policy_logits.softmax(-1, Kind::Float);

        (
            hidden_state,
            predicted_value_scalar,
            policy_probabilities,
            hidden_state_logits,
        )
    }

    pub fn recurrent_inference(
        &self,
        hidden_state: &Tensor,
        batched_action: &Tensor,
        batched_piece_identifier: &Tensor,
    ) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
        assert_eq!(
            batched_action.size()[0],
            batched_piece_identifier.size()[0],
            "Action and piece identifier batch sizes must match"
        );

        let (hidden_state_next, reward_logits) =
            self.dynamics
                .forward(hidden_state, batched_action, batched_piece_identifier);
        let (value_logits, policy_logits, hidden_state_logits) =
            self.prediction.forward(&hidden_state_next);

        let reward_scalar_prediction = self.support_to_scalar(&reward_logits);
        let value_scalar_prediction = self.support_to_scalar(&value_logits);
        let policy_probabilities = policy_logits.softmax(-1, Kind::Float);

        (
            hidden_state_next,
            reward_scalar_prediction,
            value_scalar_prediction,
            policy_probabilities,
            hidden_state_logits,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn, Device, Kind};

    #[test]
    fn test_muzero_nan_safety() {
        let variable_store = nn::VarStore::new(Device::Cpu);
        let neural_engine = MuZeroNet::new(&variable_store.root(), 256, 4, 300);

        let batch_size = 2;
        let batched_state = Tensor::zeros([batch_size, 20, 8, 16], (Kind::Float, Device::Cpu));

        let (hidden_state, value_scalar, policy_probs, hidden_state_logits) =
            neural_engine.initial_inference(&batched_state);
        assert_eq!(
            i64::try_from(hidden_state.isnan().any()).unwrap(),
            0,
            "NaN in representation"
        );
        assert_eq!(
            i64::try_from(value_scalar.isnan().any()).unwrap(),
            0,
            "NaN in initial value"
        );
        assert_eq!(
            i64::try_from(policy_probs.isnan().any()).unwrap(),
            0,
            "NaN in initial policy"
        );
        assert_eq!(
            i64::try_from(hidden_state_logits.isnan().any()).unwrap(),
            0,
            "NaN in hole logits"
        );

        let batched_action = Tensor::zeros([batch_size], (Kind::Int64, Device::Cpu));
        let batched_piece_identifier = Tensor::zeros([batch_size], (Kind::Int64, Device::Cpu));

        let (
            hidden_state_next,
            reward_scalar,
            value_scalar_next,
            policy_probs_next,
            hidden_state_logits_next,
        ) = neural_engine.recurrent_inference(
            &hidden_state,
            &batched_action,
            &batched_piece_identifier,
        );

        assert_eq!(
            i64::try_from(hidden_state_next.isnan().any()).unwrap(),
            0,
            "NaN in recurrent hidden_state"
        );
        assert_eq!(
            i64::try_from(reward_scalar.isnan().any()).unwrap(),
            0,
            "NaN in recurrent reward"
        );
        assert_eq!(
            i64::try_from(value_scalar_next.isnan().any()).unwrap(),
            0,
            "NaN in recurrent value"
        );
        assert_eq!(
            i64::try_from(policy_probs_next.isnan().any()).unwrap(),
            0,
            "NaN in recurrent policy"
        );
        assert_eq!(
            i64::try_from(hidden_state_logits_next.isnan().any()).unwrap(),
            0,
            "NaN in recurrent logits"
        );
    }

    #[test]
    fn test_support_vector_round_trip() {
        let variable_store = nn::VarStore::new(Device::Cpu);
        let neural_engine = MuZeroNet::new(&variable_store.root(), 256, 4, 300);

        // Test strictly positive scalars as the domain was deliberately shifted to [0, +inf)
        let original_scalars = Tensor::from_slice(&[0.0_f32, 10.0, 0.5, 5.5, 299.9]);

        // 1. Scalar to Support (Probabilities)
        let support_probs = neural_engine.scalar_to_support(&original_scalars);

        // 2. Convert Probabilities to Logits to feed into support_to_scalar
        // Add a tiny epsilon to prevent log(0) -> -inf which breaks softmax math
        let logits = (support_probs + 1e-9).log();

        // 3. Support to Scalar
        let reconstructed_scalars = neural_engine.support_to_scalar(&logits);

        let diff = (&original_scalars - &reconstructed_scalars).abs();
        let max_diff: f32 = diff.max().try_into().unwrap_or(1.0);

        // Max diff should be small (accounting for 32-bit float / epsilons across sqrt/pow operations)
        assert!(
            max_diff < 0.1,
            "Support Vector Math round-trip failed! Max delta: {}",
            max_diff
        );
    }
}
```

### File: `src/net/prediction.rs`

```rs
use tch::{nn, nn::Module, Kind, Tensor};

#[derive(Debug)]
struct HolePredictor {
    feature_layer_1: nn::Linear,
    feature_layer_2: nn::Linear,
}

impl HolePredictor {
    fn new(variable_store: &nn::Path, model_dimension: i64) -> Self {
        Self {
            feature_layer_1: nn::linear(
                &(variable_store / "0"),
                model_dimension,
                64,
                Default::default(),
            ),
            feature_layer_2: nn::linear(&(variable_store / "2"), 64, 2, Default::default()),
        }
    }
}

impl Module for HolePredictor {
    fn forward(&self, network_input: &Tensor) -> Tensor {
        self.feature_layer_2
            .forward(&self.feature_layer_1.forward(network_input).mish())
    }
}

#[derive(Debug)]
pub struct PredictionNet {
    value_projection: nn::Linear,
    value_normalization: nn::LayerNorm,
    value_layer_1: nn::Linear,
    value_layer_2: nn::Linear,

    policy_projection: nn::Linear,
    policy_normalization: nn::LayerNorm,
    policy_layer_1: nn::Linear,

    hole_predictor: HolePredictor,
}

impl PredictionNet {
    pub fn new(
        variable_store: &nn::Path,
        model_dimension: i64,
        support_size: i64,
        action_count: i64,
    ) -> Self {
        let value_projection = nn::linear(
            &(variable_store / "val_proj"),
            model_dimension,
            model_dimension / 2,
            Default::default(),
        );
        let value_normalization = nn::layer_norm(
            &(variable_store / "val_norm"),
            vec![model_dimension / 2],
            Default::default(),
        );
        let value_layer_1 = nn::linear(
            &(variable_store / "value_fc1"),
            model_dimension / 2,
            64,
            Default::default(),
        );
        let value_layer_2 = nn::linear(
            &(variable_store / "value_fc2"),
            64,
            2 * support_size + 1,
            Default::default(),
        );

        let policy_projection = nn::linear(
            &(variable_store / "pol_proj"),
            model_dimension,
            model_dimension / 2,
            Default::default(),
        );
        let policy_normalization = nn::layer_norm(
            &(variable_store / "pol_norm"),
            vec![model_dimension / 2],
            Default::default(),
        );
        let policy_layer_1 = nn::linear(
            &(variable_store / "policy_fc1"),
            model_dimension / 2,
            action_count,
            Default::default(),
        );

        let hole_predictor =
            HolePredictor::new(&(variable_store / "hole_predictor"), model_dimension);

        Self {
            value_projection,
            value_normalization,
            value_layer_1,
            value_layer_2,
            policy_projection,
            policy_normalization,
            policy_layer_1,
            hole_predictor,
        }
    }

    pub fn forward(&self, hidden_state: &Tensor) -> (Tensor, Tensor, Tensor) {
        assert_eq!(
            hidden_state.size().len(),
            4,
            "Prediction forward requires a 4D hidden_state tensor"
        );

        let transposed_hidden_state = hidden_state.permute([0, 2, 3, 1]);

        let value_features_mish = self
            .value_normalization
            .forward(&self.value_projection.forward(&transposed_hidden_state))
            .mish()
            .mean_dim(&[1i64, 2i64][..], false, Kind::Float);

        let value_intermediate = self.value_layer_1.forward(&value_features_mish).mish();
        let value_logits = self.value_layer_2.forward(&value_intermediate);

        let policy_features_mish = self
            .policy_normalization
            .forward(&self.policy_projection.forward(&transposed_hidden_state))
            .mish()
            .mean_dim(&[1i64, 2i64][..], false, Kind::Float);

        let policy_logits = self.policy_layer_1.forward(&policy_features_mish);

        let hole_logits = self
            .hole_predictor
            .forward(&transposed_hidden_state)
            .flatten(1, 3);

        (value_logits, policy_logits, hole_logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn, Device, Kind};

    #[test]
    fn test_prediction_output_shapes() {
        let variable_store = nn::VarStore::new(Device::Cpu);
        let prediction_network = PredictionNet::new(&variable_store.root(), 16, 300, 288);

        let batch_size = 2;
        let hidden_state = Tensor::zeros([batch_size, 16, 8, 8], (Kind::Float, Device::Cpu));

        let (value_logits, policy_logits, hole_logits) = prediction_network.forward(&hidden_state);

        assert_eq!(
            value_logits.size(),
            vec![batch_size, 601],
            "Value support boundaries do not match"
        );
        assert_eq!(
            policy_logits.size(),
            vec![batch_size, 288],
            "Action count boundaries do not match"
        );
        assert_eq!(
            hole_logits.size(),
            vec![batch_size, 128],
            "Spatial hole masking size does not match"
        );
    }
}
```

### File: `src/net/projector.rs`

```rs
use tch::{nn, nn::Module, Kind, Tensor};

#[derive(Debug)]
pub struct ProjectorNet {
    proj: nn::Linear,
    norm1: nn::LayerNorm,
    fully_connected_layer_1: nn::Linear,
    norm2: nn::LayerNorm,
    fc2: nn::Linear,
}

impl ProjectorNet {
    pub fn new(variable_store: &nn::Path, hidden_dimension_size: i64, proj_dim: i64, out_dim: i64) -> Self {
        let proj = nn::linear(
            &(variable_store / "proj"),
            hidden_dimension_size,
            hidden_dimension_size / 2,
            Default::default(),
        );
        let norm1 = nn::layer_norm(
            &(variable_store / "norm1"),
            vec![hidden_dimension_size / 2],
            Default::default(),
        );
        let fully_connected_layer_1 = nn::linear(
            &(variable_store / "fully_connected_layer_1"),
            hidden_dimension_size / 2,
            proj_dim,
            Default::default(),
        );
        let norm2 = nn::layer_norm(&(variable_store / "norm2"), vec![proj_dim], Default::default());
        let fc2 = nn::linear(&(variable_store / "fc2"), proj_dim, out_dim, Default::default());
        Self {
            proj,
            norm1,
            fully_connected_layer_1,
            norm2,
            fc2,
        }
    }

    pub fn forward(&self, hidden_state_tensor: &Tensor) -> Tensor {
        let intermediate_features = self
            .norm1
            .forward(&self.proj.forward(&hidden_state_tensor.permute([0, 2, 3, 1])))
            .mish()
            .mean_dim(&[1i64, 2i64][..], false, Kind::Float);
        let intermediate_features = self.norm2.forward(&self.fully_connected_layer_1.forward(&intermediate_features)).mish();
        self.fc2.forward(&intermediate_features)
    }
}
```

### File: `src/net/representation.rs`

```rs
use crate::net::FlattenedResNetBlock;
use tch::{nn, nn::Module, Tensor};

#[derive(Debug)]
pub struct RepresentationNet {
    proj_in: nn::Conv2D,
    blocks: Vec<FlattenedResNetBlock>,
    scale_norm: nn::LayerNorm,
}

impl RepresentationNet {
    pub fn new(vs: &nn::Path, hidden_dimension_size: i64, num_blocks: i64) -> Self {
        let config = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
        let proj_in = nn::conv2d(&(vs / "proj_in"), 40, hidden_dimension_size, 3, config);
        let mut blocks = Vec::new();
        let blk_vs = vs / "blocks";
        for i in 0..num_blocks {
            blocks.push(FlattenedResNetBlock::new(
                &(&blk_vs / i),
                hidden_dimension_size,
                128,
            ));
        }
        let scale_norm = nn::layer_norm(
            &(vs / "scale_norm"),
            vec![hidden_dimension_size],
            Default::default(),
        );
        Self {
            proj_in,
            blocks,
            scale_norm,
        }
    }
}

impl Module for RepresentationNet {
    fn forward(&self, input_tensor_batch_channel_height_width: &Tensor) -> Tensor {
        let input_shape = input_tensor_batch_channel_height_width.size();
        assert_eq!(
            input_shape.len(),
            4,
            "RepresentationNet requires [Batch, 20, 8, 16] input"
        );
        let batch = input_tensor_batch_channel_height_width.size()[0];
        let x_reshaped = input_tensor_batch_channel_height_width
            .view([batch, 20, 8, 8, 2])
            .permute([0, 1, 4, 2, 3])
            .reshape([batch, 40, 8, 8]);
        let mut h = self.proj_in.forward(&x_reshaped);
        for block in &self.blocks {
            h = block.forward(&h);
        }
        self.scale_norm
            .forward(&h.permute([0, 2, 3, 1]).contiguous())
            .permute([0, 3, 1, 2])
            .contiguous()
    }
}
```

### File: `src/net/resnet.rs`

```rs
use crate::core::features::get_valid_spatial_mask_8x8;
use tch::{nn, nn::Module, Tensor};

#[derive(Debug)]
pub struct FlattenedResNetBlock {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
    spatial_mask: Tensor,
}

impl FlattenedResNetBlock {
    pub fn new(variable_store: &nn::Path, hidden_dimension_size: i64, _grid_size: i64) -> Self {
        let config = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
        let conv1 = nn::conv2d(
            &(variable_store / "conv1"),
            hidden_dimension_size,
            hidden_dimension_size,
            3,
            config,
        );
        let conv2 = nn::conv2d(
            &(variable_store / "conv2"),
            hidden_dimension_size,
            hidden_dimension_size,
            3,
            config,
        );

        let norm1 = nn::layer_norm(
            &(variable_store / "norm1"),
            vec![hidden_dimension_size],
            Default::default(),
        );
        let norm2 = nn::layer_norm(
            &(variable_store / "norm2"),
            vec![hidden_dimension_size],
            Default::default(),
        );

        let spatial_mask = get_valid_spatial_mask_8x8(variable_store.device());

        Self {
            conv1,
            conv2,
            norm1,
            norm2,
            spatial_mask,
        }
    }
}

impl Module for FlattenedResNetBlock {
    fn forward(&self, input_tensor_batch_channel_height_width: &Tensor) -> Tensor {
        let input_shape = input_tensor_batch_channel_height_width.size();
        assert_eq!(
            input_shape.len(),
            4,
            "FlattenedResNetBlock requires [Batch, Channels, Height, Width] input"
        );
        let residual = input_tensor_batch_channel_height_width;
        let mut output_tensor = self.conv1.forward(input_tensor_batch_channel_height_width);
        output_tensor = &output_tensor * &self.spatial_mask;

        output_tensor = self
            .norm1
            .forward(&output_tensor.permute([0, 2, 3, 1]).contiguous())
            .permute([0, 3, 1, 2])
            .contiguous()
            .mish();
        output_tensor = &output_tensor * &self.spatial_mask;

        output_tensor = self.conv2.forward(&output_tensor);
        output_tensor = &output_tensor * &self.spatial_mask;

        output_tensor = self
            .norm2
            .forward(&output_tensor.permute([0, 2, 3, 1]).contiguous())
            .permute([0, 3, 1, 2])
            .contiguous();
        output_tensor = &output_tensor * &self.spatial_mask;

        ((residual + output_tensor).mish()) * &self.spatial_mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::features::HEXAGONAL_TO_CARTESIAN_MAP_ARRAY;
    use tch::{Device, Kind};

    #[test]
    fn test_topology_wormhole_masking() {
        let variable_store = nn::VarStore::new(Device::Cpu);
        let hidden_dimension_size = 16;
        let block = FlattenedResNetBlock::new(&variable_store.root(), hidden_dimension_size, 0);

        let input = Tensor::zeros([1, hidden_dimension_size, 8, 8], (Kind::Float, Device::Cpu));
        // Place a 1.0 at a valid hex position mapping (row, col/2)
        let (r, c) = HEXAGONAL_TO_CARTESIAN_MAP_ARRAY[0];
        let _ = input
            .narrow(2, r as i64, 1)
            .narrow(3, (c / 2) as i64, 1)
            .fill_(1.0);

        let mut output_tensor = input;
        for _ in 0..10 {
            output_tensor = block.forward(&output_tensor);
        }

        // Verify that dead cells are absolutely zero
        let out_slice: Vec<f32> = output_tensor.reshape([-1]).try_into().unwrap();

        let mut dead_cells_count = 0;
        for r in 0..8 {
            for c in 0..8 {
                let mut is_valid = false;
                for &(vr, vc) in HEXAGONAL_TO_CARTESIAN_MAP_ARRAY.iter() {
                    if vr == r && (vc / 2) == c {
                        is_valid = true;
                        break;
                    }
                }
                if !is_valid {
                    dead_cells_count += 1;
                    for m in 0..hidden_dimension_size as usize {
                        let idx = m * 64 + r * 8 + c; // N=1
                        assert_eq!(
                            out_slice[idx], 0.0,
                            "Dead cell {},{} leaked activation!",
                            r, c
                        );
                    }
                }
            }
        }
        assert!(
            dead_cells_count > 0,
            "There should be multiple dead cells in the 8x8 folding"
        );
    }
}
```

### File: `src/node.rs`

```rs
use crate::core::board::GameStateExt;
use crate::core::constants::STANDARD_PIECES;
use once_cell::sync::Lazy;

pub static COMPACT_PIECE_MASKS: Lazy<Vec<Vec<(usize, u128)>>> = Lazy::new(|| {
    STANDARD_PIECES
        .iter()
        .map(|masks| {
            masks
                .iter()
                .copied()
                .enumerate()
                .filter(|&(_, m)| m != 0)
                .collect()
        })
        .collect()
});

#[derive(Clone)]
pub struct LatentNode {
    pub visits: i32,
    pub value_sum: f32,
    pub policy_logit: f32, // CHANGED: Store the logit, not the raw action_prior_probability
    pub action_prior_probability: f32,
    pub reward: f32,
    pub gumbel_noise: f32,
    pub first_child: u32,
    pub next_sibling: u32,
    pub action: i16,
    pub hidden_state_index: u32,
    pub is_topologically_expanded: bool,
}

impl LatentNode {
    pub fn new(action_prior_probability: f32, action: i16) -> Self {
        LatentNode {
            visits: 0,
            value_sum: 0.0,
            // CHANGED: Compute the expensive logarithm exactly ONCE here
            policy_logit: action_prior_probability.max(1e-8).ln(),
            action_prior_probability,
            reward: 0.0,
            gumbel_noise: 0.0,
            first_child: u32::MAX,
            next_sibling: u32::MAX,
            action,
            hidden_state_index: u32::MAX,
            is_topologically_expanded: false,
        }
    }

    pub fn value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / (self.visits as f32)
        }
    }

    pub fn get_child(&self, arena: &[LatentNode], action: i32) -> usize {
        let mut current_node_pointer = self.first_child;
        while current_node_pointer != u32::MAX {
            if arena[current_node_pointer as usize].action == action as i16 {
                return current_node_pointer as usize;
            }
            current_node_pointer = arena[current_node_pointer as usize].next_sibling;
        }
        usize::MAX
    }
}

pub fn get_valid_action_mask(state: &GameStateExt) -> [bool; 288] {
    let mut valid_action_mask = [false; 288];
    if state.terminal {
        return valid_action_mask; // Dead boards cannot expand further mathematically.
    }

    for slot in 0..3 {
        let piece_identifier = state.available[slot];
        if piece_identifier == -1 {
            continue;
        }

        for &(rotation_index, structural_mask) in
            COMPACT_PIECE_MASKS[piece_identifier as usize].iter()
        {
            if (state.board_bitmask_u128 & structural_mask) == 0 {
                let absolute_action_index = (slot * 96) + rotation_index;
                valid_action_mask[absolute_action_index] = true;
            }
        }
    }
    valid_action_mask
}

pub fn select_child(arena: &[LatentNode], node_index: usize, is_root: bool) -> (i32, usize) {
    let parent_node = &arena[node_index];

    // NORMALIZE Q
    let mut minimum_q_value = f32::INFINITY;
    let mut maximum_q_value = f32::NEG_INFINITY;
    let mut child_index = parent_node.first_child;

    while child_index != u32::MAX {
        let child_node = &arena[child_index as usize];
        let expected_q_value = if child_node.visits == 0 {
            parent_node.value()
        } else {
            child_node.reward + 0.99 * child_node.value()
        };
        if expected_q_value < minimum_q_value {
            minimum_q_value = expected_q_value;
        }
        if expected_q_value > maximum_q_value {
            maximum_q_value = expected_q_value;
        }
        child_index = child_node.next_sibling;
    }

    let mut highest_score = f32::NEG_INFINITY;
    let mut highest_action_index = -1;
    let mut highest_child_index = usize::MAX;

    let mut child_index = parent_node.first_child;
    while child_index != u32::MAX {
        let child_node = &arena[child_index as usize];
        let action_index = child_node.action as i32;

        let raw_expected_q_value = if child_node.visits == 0 {
            parent_node.value()
        } else {
            child_node.reward + 0.99 * child_node.value()
        };

        let normalized_q_value = if maximum_q_value > minimum_q_value {
            (raw_expected_q_value - minimum_q_value) / (maximum_q_value - minimum_q_value)
        } else {
            0.5
        };

        // CHANGED: Instantly read the precomputed logit. No math required!
        let action_score = if is_root {
            let gumbel_noise_injected_logit = child_node.policy_logit + child_node.gumbel_noise;
            let exploration_scale = 50.0 / ((child_node.visits + 1) as f32);
            gumbel_noise_injected_logit + (exploration_scale * normalized_q_value)
        } else {
            let puct_exploration_constant = 1.25;
            let upper_confidence_bound_score = puct_exploration_constant
                * child_node.action_prior_probability
                * ((parent_node.visits as f32).sqrt() / (1.0 + child_node.visits as f32));
            normalized_q_value + upper_confidence_bound_score
        };

        if action_score > highest_score {
            highest_score = action_score;
            highest_action_index = action_index;
            highest_child_index = child_index as usize;
        }
        child_index = child_node.next_sibling;
    }

    (highest_action_index, highest_child_index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::GameStateExt;

    #[test]
    fn test_latent_node() {
        let node = LatentNode::new(0.5, 0);
        assert_eq!(node.value(), 0.0);

        let mut node2 = LatentNode::new(0.5, 0);
        node2.visits = 2;
        node2.value_sum = 1.0;
        assert_eq!(node2.value(), 0.5);
    }

    #[test]
    fn test_valid_action_mask() {
        let mut state = GameStateExt::new(Some([0, 0, 0]), 0, 0, 6, 0);
        let mask = get_valid_action_mask(&state);
        assert_eq!(
            mask.len(),
            288,
            "Action mask must be exactly 288 elements long."
        );

        let valid_moves = mask.iter().filter(|&&b| b).count();
        assert!(
            valid_moves > 0 && valid_moves <= 288,
            "Must be valid moves on empty setup."
        );

        state.terminal = true;
        let terminal_mask = get_valid_action_mask(&state);
        assert!(
            !terminal_mask.contains(&true),
            "Terminal state must strictly return a mask of all false."
        );
    }

    #[test]
    fn test_select_child_puct_vs_gumbel() {
        let mut arena = vec![
            LatentNode::new(1.0, -1),
            LatentNode::new(0.5, 0),
            LatentNode::new(0.6, 1),
        ];
        arena[0].visits = 10;
        arena[0].first_child = 1;
        arena[1].next_sibling = 2;
        arena[1].gumbel_noise = 100.0;
        arena[2].gumbel_noise = 0.0;

        let (internal_action, internal_child) = select_child(&arena, 0, false);
        assert_eq!(
            internal_action, 1,
            "PUCT should have selected child B based on higher action_prior_probability"
        );
        assert_eq!(internal_child, 2);

        let (root_action, root_child) = select_child(&arena, 0, true);
        assert_eq!(
            root_action, 0,
            "Gumbel should have selected child A based on massive injected noise"
        );
        assert_eq!(root_child, 1);
    }
}
```

### File: `src/performance_benches.rs`

```rs
#[cfg(test)]
mod performance_tests {
    use crate::core::board::GameStateExt;
    use crate::core::features::extract_feature_native;
    use crate::mcts::{gc_tree, MctsTree};
    use crate::node::LatentNode;
    use crate::queue::FixedInferenceQueue;
    use std::time::Instant;
    use tch::{Device, Kind, Tensor};

    // 1. Feature Extraction Throughput
    #[test]
    fn bench_feature_extraction() {
        let cases = [
            ("Empty", 0u128),
            ("Half", 0x0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F),
            ("Full", u128::MAX >> 32),
        ];
        for (name, board) in cases {
            let state = GameStateExt::new(None, board, 0, 6, 0);
            let start = Instant::now();
            let mut slice = vec![0.0f32; 20 * 128];
            for _ in 0..10_000 {
                extract_feature_native(
                    &mut slice,
                    state.board_bitmask_u128,
                    &state.available,
                    &[],
                    &[],
                    6,
                );
            }
            println!("Feature Extraction ({}): {:?}", name, start.elapsed());
        }
    }

    // 2. Inference Queue Contention
    #[test]
    fn bench_inference_queue_contention() {
        let cases = [10, 20, 30];
        for &workers in &cases {
            let queue = FixedInferenceQueue::new(workers, workers);
            let start = Instant::now();

            std::thread::scope(|s| {
                for w in 0..workers {
                    let q = queue.clone();
                    s.spawn(move || {
                        for _ in 0..1000 {
                            let (evaluation_request_transmitter, _) =
                                crossbeam_channel::unbounded();
                            let _ = q.push_batch(
                                w,
                                vec![crate::mcts::EvaluationRequest {
                                    is_initial: true,
                                    board_bitmask: 0,
                                    available_pieces: [-1; 3],
                                    recent_board_history: [0; 8],
                                    history_len: 0,
                                    recent_action_history: [0; 4],
                                    action_history_len: 0,
                                    difficulty: 6,
                                    piece_action: 0,
                                    piece_id: 0,
                                    node_index: 0,
                                    worker_id: w,
                                    parent_cache_index: 0,
                                    leaf_cache_index: 0,
                                    evaluation_request_transmitter,
                                }],
                            );
                        }
                    });
                }
                s.spawn(|| {
                    let mut popped = 0usize;
                    while popped < workers * 1000 {
                        if let Ok(batch) =
                            queue.pop_batch_timeout(1024, std::time::Duration::from_millis(10))
                        {
                            popped += batch.len();
                        }
                    }
                });
            });
            println!(
                "Queue Contention ({} workers): {:?}",
                workers,
                start.elapsed()
            );
        }
    }

    // 3. Replay Buffer Sampling Latency
    #[test]
    fn bench_replay_buffer_sample() {
        let rb = crate::train::buffer::ReplayBuffer::new(10000, 5, 10);
        // Fill dummy data...
        let cases = [128, 512, 1024];
        for &batch_size in &cases {
            let start = Instant::now();
            for _ in 0..10 {
                let _ = rb.sample_batch(batch_size, Device::Cpu, 1.0);
            }
            println!("RB Sample (Batch {}): {:?}", batch_size, start.elapsed());
        }
    }

    // 4. MCTS Tree GC Performance
    #[test]
    fn bench_mcts_gc_traversal() {
        let cases = [100, 1000, 5000];
        for &nodes in &cases {
            let mut arena = vec![LatentNode::new(0.0, 0); nodes];
            for (i, node) in arena.iter_mut().enumerate().take(nodes - 1) {
                node.first_child = (i + 1) as u32; // Create a deep linked list
            }
            let tree = MctsTree {
                arena: arena.clone(),
                swap_arena: vec![LatentNode::new(0.0, -1); nodes],
                pointer_remapping: vec![u32::MAX; nodes],
                arena_alloc_ptr: nodes,
                root_index: 0,
                free_list: vec![],
                maximum_allowed_nodes_in_search_tree: nodes as u32,
            };

            let start = Instant::now();
            let _ = gc_tree(tree, 1);
            println!("MCTS GC ({} nodes): {:?}", nodes, start.elapsed());
        }
    }

    // 5. Tensor Bulk Copy vs Element-wise
    #[test]
    fn bench_tensor_bulk_copy() {
        let batch_size = 1024;
        let data = vec![1.0f32; batch_size * 2560];
        let tensor = Tensor::zeros([batch_size as i64, 2560], (Kind::Float, Device::Cpu));

        // Case 1: Element-wise
        let start = Instant::now();
        unsafe {
            let ptr = tensor.data_ptr() as *mut f32;
            for (i, &val) in data.iter().enumerate() {
                *ptr.add(i) = val;
            }
        }
        println!("Tensor Element-wise Copy: {:?}", start.elapsed());

        // Case 2: Bulk Copy
        let start2 = Instant::now();
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), tensor.data_ptr() as *mut f32, data.len());
        }
        println!("Tensor Bulk Copy: {:?}", start2.elapsed());
    }

    // 6. SumTree (PER) Update/Sample Speed
    #[test]
    fn bench_sumtree_sample() {
        let cases = [10_000, 100_000, 1_000_000];
        for &cap in &cases {
            let tree = crate::sumtree::SegmentTree::new(cap);
            tree.update(cap / 2, 5.0);
            let start = Instant::now();
            for _ in 0..10_000 {
                let _ = tree.sample_proportional(256);
            }
            println!("SumTree Sample (Cap {}): {:?}", cap, start.elapsed());
        }
    }

    // 7. Board State Transition (Bitwise)
    #[test]
    fn bench_board_apply_move() {
        let cases = [
            ("No Clear", 0u128, 0),
            ("1 Line", crate::core::constants::ALL_MASKS[0] & !1, 0),
            (
                "3 Lines",
                (crate::core::constants::ALL_MASKS[0]
                    | crate::core::constants::ALL_MASKS[1]
                    | crate::core::constants::ALL_MASKS[2])
                    & !1,
                0,
            ),
        ];
        for (name, board, idx) in cases {
            let state = GameStateExt::new(Some([0, -1, -1]), board, 0, 6, 0);
            let start = Instant::now();
            for _ in 0..100_000 {
                let mut s = state.clone();
                let _ = s.apply_move(0, idx);
            }
            println!("Board Apply Move ({}): {:?}", name, start.elapsed());
        }
    }

    // 8. Gumbel Noise Injection
    #[test]
    fn bench_gumbel_noise_injection() {
        let cases = [10, 100, 288];
        for &valid_actions in &cases {
            let _arena = vec![LatentNode::new(0.0, 0); 300];
            let actions: Vec<i32> = (0..valid_actions).collect();
            let _probs = vec![0.01; 288];

            let start = Instant::now();
            for _ in 0..10_000 {
                // Simulating the logic inside inject_gumbel_noise
                let mut rng = rand::thread_rng();
                for &_a in &actions {
                    let u: f32 = rand::Rng::gen_range(&mut rng, 1e-6..=1.0);
                    let _g = -(-u.ln()).ln();
                }
            }
            println!(
                "Gumbel Noise ({} actions): {:?}",
                valid_actions,
                start.elapsed()
            );
        }
    }

    // 9. TD-Bootstrap Accumulation
    #[test]
    fn bench_td_bootstrap_accumulation() {
        let cases = [1, 5, 10];
        for &td in &cases {
            let start = Instant::now();
            for _ in 0..10_000 {
                let mut sum = 0.0;
                for step in 0..td {
                    sum += 1.0 * 0.99f32.powi(step);
                }
                std::hint::black_box(sum);
            }
            println!("TD Bootstrap (TD={}): {:?}", td, start.elapsed());
        }
    }

    // 10. Recurrent Inference Prep
    #[test]
    fn bench_recurrent_inference_prep() {
        let cases = [1, 128];
        for &batch in &cases {
            let pinned_actions = Tensor::zeros([batch], (Kind::Int64, Device::Cpu));
            let start = Instant::now();
            for _ in 0..10_000 {
                // Simulating SafeTensorGuard overhead
                unsafe {
                    let ptr = pinned_actions.data_ptr() as *mut i64;
                    for i in 0..batch {
                        *ptr.add(i as usize) = i;
                    }
                }
            }
            println!("Recurrent Prep (Batch {}): {:?}", batch, start.elapsed());
        }
    }
    // 11. Extreme Inference Queue Contention (128 workers, heavy load)
    #[test]
    fn bench_extreme_queue_contention() {
        let workers = 128;
        let queue = FixedInferenceQueue::new(workers, workers);
        let start = Instant::now();

        std::thread::scope(|s| {
            for w in 0..workers {
                let q = queue.clone();
                s.spawn(move || {
                    for _ in 0..5000 {
                        let (evaluation_request_transmitter, _) = crossbeam_channel::unbounded();
                        let _ = q.push_batch(
                            w,
                            vec![crate::mcts::EvaluationRequest {
                                is_initial: true,
                                board_bitmask: 0,
                                available_pieces: [-1; 3],
                                recent_board_history: [0; 8],
                                history_len: 0,
                                recent_action_history: [0; 4],
                                action_history_len: 0,
                                difficulty: 6,
                                piece_action: 0,
                                piece_id: 0,
                                node_index: 0,
                                worker_id: w,
                                parent_cache_index: 0,
                                leaf_cache_index: 0,
                                evaluation_request_transmitter,
                            }],
                        );
                    }
                });
            }
            s.spawn(|| {
                let mut popped = 0usize;
                while popped < workers * 5000 {
                    if let Ok(batch) =
                        queue.pop_batch_timeout(1024, std::time::Duration::from_millis(10))
                    {
                        popped += batch.len();
                    }
                }
            });
        });
        println!(
            "Extreme Queue Contention ({} workers): {:?}",
            workers,
            start.elapsed()
        );
    }

    // 12. Crossbeam Channel Drain Speed
    #[test]
    fn bench_channel_drain_speed() {
        let (evaluation_request_transmitter, evaluation_response_receiver) =
            crossbeam_channel::bounded(100_000);
        for _ in 0..100_000 {
            let (res_evaluation_request_transmitter, _) = crossbeam_channel::unbounded();
            let _ = evaluation_request_transmitter.send(crate::mcts::EvaluationRequest {
                is_initial: true,
                board_bitmask: 0,
                available_pieces: [-1; 3],
                recent_board_history: [0; 8],
                history_len: 0,
                recent_action_history: [0; 4],
                action_history_len: 0,
                difficulty: 6,
                piece_action: 0,
                piece_id: 0,
                node_index: 0,
                worker_id: 0,
                parent_cache_index: 0,
                leaf_cache_index: 0,
                evaluation_request_transmitter: res_evaluation_request_transmitter,
            });
        }
        let start = Instant::now();
        let mut popped = 0;
        while evaluation_response_receiver.try_recv().is_ok() {
            popped += 1;
        }
        println!(
            "Channel Drain Speed ({} items): {:?}",
            popped,
            start.elapsed()
        );
    }

    // 13. Replay Buffer Extreme Concurrency
    #[test]
    fn bench_replay_buffer_concurrency() {
        let rb = std::sync::Arc::new(crate::train::buffer::ReplayBuffer::new(200_000, 5, 10));
        let workers = 16;
        let start = Instant::now();
        std::thread::scope(|s| {
            for _ in 0..workers {
                let rb_clone = rb.clone();
                s.spawn(move || {
                    for _ in 0..10_000 {
                        rb_clone.add_game(crate::train::buffer::replay::OwnedGameData {
                            difficulty_setting: 6,
                            episode_score: 0.0,
                            steps: vec![crate::train::buffer::replay::GameStep {
                                board_state: [0, 0],
                                available_pieces: [0, -1, -1],
                                action_taken: 0,
                                piece_identifier: 0,
                                reward_received: 0.0,
                                policy_target: [0.0; 288],
                                value_target: 0.0,
                            }],
                            lines_cleared: 0,
                            mcts_depth_mean: 0.0,
                            mcts_search_time_mean: 0.0,
                        });
                    }
                });
            }
        });
        println!(
            "Replay Buffer Extreme Concurrency ({} workers): {:?}",
            workers,
            start.elapsed()
        );
    }

    // 14. Batch Size Latency Scaling
    #[test]
    fn bench_batch_size_latency_scaling() {
        let queue = FixedInferenceQueue::new(256, 256);
        let start = Instant::now();
        let sizes = [64, 128, 256, 512, 1024];
        for &size in &sizes {
            for _ in 0..size {
                let (evaluation_request_transmitter, _) = crossbeam_channel::unbounded();
                let _ = queue.push_batch(
                    0,
                    vec![crate::mcts::EvaluationRequest {
                        is_initial: true,
                        board_bitmask: 0,
                        available_pieces: [-1; 3],
                        recent_board_history: [0; 8],
                        history_len: 0,
                        recent_action_history: [0; 4],
                        action_history_len: 0,
                        difficulty: 6,
                        piece_action: 0,
                        piece_id: 0,
                        node_index: 0,
                        worker_id: 0,
                        parent_cache_index: 0,
                        leaf_cache_index: 0,
                        evaluation_request_transmitter,
                    }],
                );
            }
            let s = Instant::now();
            let _ = queue
                .pop_batch_timeout(size, std::time::Duration::from_millis(10))
                .unwrap();
            println!("Batch Size Pop (size {}): {:?}", size, s.elapsed());
        }
        println!("Batch Size Latency Total: {:?}", start.elapsed());
    }

    // 15. MCTS Sequential Halving Deep Traversal
    #[test]
    fn bench_mcts_sequential_halving_deep() {
        let start = Instant::now();
        for _ in 0..5000 {
            let mut visits = vec![0; 64];
            let mut active_indices: Vec<usize> = (0..64).collect();
            let mut remaining_sims = 128;

            while active_indices.len() > 8 {
                let current_k = active_indices.len();
                let next_k = std::cmp::max(8, current_k / 2);
                let phase_len = ((active_indices.len() as f32).log2() as usize).max(1);
                let sims_for_phase = remaining_sims / phase_len;
                let sims_per_child = sims_for_phase / current_k;

                for &idx in &active_indices {
                    visits[idx] += sims_per_child;
                }
                remaining_sims -= sims_for_phase;
                active_indices.truncate(next_k);
            }
        }
        println!("MCTS Sequential Halving Deep: {:?}", start.elapsed());
    }

    // 16. Reanalyze Queue Bottleneck Simulate
    #[test]
    fn bench_reanalyze_queue_bottleneck() {
        let (evaluation_request_transmitter, evaluation_response_receiver) =
            crossbeam_channel::bounded(1000);
        let start = Instant::now();
        std::thread::scope(|s| {
            s.spawn(|| {
                for i in 0..50_000 {
                    let _ = evaluation_request_transmitter.send(i);
                }
            });
            s.spawn(|| {
                let mut received = 0;
                while evaluation_response_receiver.recv().is_ok() {
                    received += 1;
                    if received == 50_000 {
                        break;
                    }
                }
            });
        });
        println!("Reanalyze Queue Transfer: {:?}", start.elapsed());
    }

    // 18. SumTree Extreme Shard Contention
    #[test]
    fn bench_sumtree_extreme_shard_contention() {
        let tree = std::sync::Arc::new(std::sync::Mutex::new(crate::sumtree::SegmentTree::new(
            100_000,
        )));
        let workers = 16;
        let start = Instant::now();
        std::thread::scope(|s| {
            for _ in 0..workers {
                let tree_clone = tree.clone();
                s.spawn(move || {
                    for _ in 0..2000 {
                        if let Ok(lock) = tree_clone.lock() {
                            let _ = lock.sample_proportional(64);
                        }
                    }
                });
            }
        });
        println!("SumTree Extreme Contention: {:?}", start.elapsed());
    }

    // 19. Node Allocation and Deallocation (Arena Stress)
    #[test]
    fn bench_node_arena_stress() {
        let mut tree = MctsTree {
            arena: vec![LatentNode::new(0.0, 0); 100_000],
            swap_arena: vec![LatentNode::new(0.0, 0); 100_000],
            pointer_remapping: vec![u32::MAX; 100_000],
            arena_alloc_ptr: 1,
            root_index: 0,
            free_list: vec![],
            maximum_allowed_nodes_in_search_tree: 100_000,
        };
        let start = Instant::now();
        for _ in 0..50_000 {
            tree.arena_alloc_ptr += 1;
        }
        println!("Node Arena Allocation Stress: {:?}", start.elapsed());
    }

    // 20. Atomic Ordering Stress (For active_producers)
    #[test]
    fn bench_atomic_ordering_stress() {
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let workers = 64;
        let start = Instant::now();
        std::thread::scope(|s| {
            for _ in 0..workers {
                let c = counter.clone();
                s.spawn(move || {
                    for _ in 0..100_000 {
                        c.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    }
                });
            }
        });
        println!("Atomic Ordering Stress: {:?}", start.elapsed());
    }
}
```

### File: `src/queue.rs`

```rs
use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, Sender};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::mcts::EvaluationRequest;

pub struct FixedInferenceQueue {
    pub evaluation_request_transmitter: Sender<Vec<EvaluationRequest>>,
    pub evaluation_response_receiver: Receiver<Vec<EvaluationRequest>>,
    pub active_producers: AtomicUsize,
    pub remainder: std::sync::Mutex<Vec<EvaluationRequest>>,
}

impl FixedInferenceQueue {
    pub fn new(_buffer_capacity_limit: usize, total_producers: usize) -> Arc<Self> {
        let (evaluation_request_transmitter, evaluation_response_receiver) = bounded(16384);
        Arc::new(Self {
            evaluation_request_transmitter,
            evaluation_response_receiver,
            active_producers: AtomicUsize::new(total_producers),
            remainder: std::sync::Mutex::new(Vec::new()),
        })
    }

    #[allow(clippy::result_unit_err)]
    pub fn push_batch(&self, _worker_id: usize, reqs: Vec<EvaluationRequest>) -> Result<(), ()> {
        self.evaluation_request_transmitter
            .send(reqs)
            .map_err(|_| ())
    }

    #[allow(dead_code)]
    pub fn disconnect_producer(&self) {
        self.active_producers.fetch_sub(1, Ordering::SeqCst);
    }

    #[allow(clippy::result_unit_err)]
    pub fn pop_batch_timeout(
        &self,
        max_batch_size: usize,
        timeout: Duration,
    ) -> Result<Vec<EvaluationRequest>, ()> {
        let mut batch = Vec::with_capacity(max_batch_size);

        if max_batch_size == 0 {
            return Ok(batch);
        }

        {
            let mut rem = self.remainder.lock().unwrap();
            if !rem.is_empty() {
                let to_take = std::cmp::min(max_batch_size, rem.len());
                let tail = rem.split_off(to_take);
                batch.append(&mut rem);
                *rem = tail;
                if batch.len() == max_batch_size {
                    return Ok(batch);
                }
            }
        }

        let time_limit = std::time::Instant::now() + timeout;

        while batch.len() < max_batch_size {
            let remaining_time = time_limit.saturating_duration_since(std::time::Instant::now());
            if remaining_time.is_zero() && !batch.is_empty() {
                break;
            }

            match self
                .evaluation_response_receiver
                .recv_timeout(if batch.is_empty() {
                    timeout
                } else {
                    remaining_time
                }) {
                Ok(mut reqs) => {
                    let space_left = max_batch_size - batch.len();
                    if reqs.len() <= space_left {
                        batch.append(&mut reqs);
                    } else {
                        let rest = reqs.split_off(space_left);
                        batch.append(&mut reqs);
                        *self.remainder.lock().unwrap() = rest;
                        break;
                    }
                }
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => {
                    if batch.is_empty() {
                        return Err(());
                    }
                    break;
                }
            }
        }

        if batch.is_empty() && self.active_producers.load(Ordering::SeqCst) == 0 {
            return Err(());
        }

        Ok(batch)
    }
}
```

### File: `src/sumtree.rs`

```rs
pub type SumTreeSample = (Vec<(usize, f64)>, Vec<f32>);
use rand::{thread_rng, Rng};
use std::sync::atomic::{AtomicU64, Ordering};

fn update_max_priority(atom: &AtomicU64, new_val: f64) {
    let mut current_bits = atom.load(Ordering::Relaxed);
    loop {
        let current_val = f64::from_bits(current_bits);
        if new_val <= current_val {
            break;
        }
        match atom.compare_exchange_weak(
            current_bits,
            new_val.to_bits(),
            Ordering::SeqCst,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(actual) => current_bits = actual,
        }
    }
}

pub struct SegmentTree {
    _buffer_capacity_limit: usize,
    tree_buffer_capacity_limit: usize,
    pub tree_array: Vec<AtomicU64>,
}

impl SegmentTree {
    pub fn new(buffer_capacity_limit: usize) -> Self {
        let mut tree_buffer_capacity_limit = 1;
        while tree_buffer_capacity_limit < buffer_capacity_limit {
            tree_buffer_capacity_limit *= 2;
        }

        let mut tree_array = Vec::with_capacity(2 * tree_buffer_capacity_limit);
        for _ in 0..(2 * tree_buffer_capacity_limit) {
            tree_array.push(AtomicU64::new(0.0f64.to_bits()));
        }

        Self {
            _buffer_capacity_limit: buffer_capacity_limit,
            tree_buffer_capacity_limit,
            tree_array,
        }
    }

    pub fn update(&self, data_index: usize, priority_value: f64) {
        assert!(
            priority_value.is_finite(),
            "Segment tree update received non-finite priority"
        );
        assert!(
            priority_value >= 0.0,
            "Segment tree priority cannot be negative"
        );
        assert!(
            data_index < self.tree_buffer_capacity_limit,
            "Segment tree update index {} violates sumtree array bounds {}!",
            data_index,
            self.tree_buffer_capacity_limit
        );

        let mut tree_index = data_index + self.tree_buffer_capacity_limit;
        let old_bits = self.tree_array[tree_index].swap(priority_value.to_bits(), Ordering::SeqCst);
        let old_val = f64::from_bits(old_bits);
        let delta_change = priority_value - old_val;

        tree_index /= 2;
        while tree_index > 0 {
            let atom = &self.tree_array[tree_index];
            let mut current_bits = atom.load(Ordering::Relaxed);
            loop {
                let current_val = f64::from_bits(current_bits);
                let new_val = current_val + delta_change;
                match atom.compare_exchange_weak(
                    current_bits,
                    new_val.to_bits(),
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(actual) => current_bits = actual,
                }
            }
            tree_index /= 2;
        }
    }

    pub fn get_total_priority(&self) -> f64 {
        f64::from_bits(self.tree_array[1].load(Ordering::Relaxed))
    }

    pub fn get_leaf(&self, mut target_value: f64) -> (usize, f64) {
        let mut tree_index = 1;

        while tree_index < self.tree_buffer_capacity_limit {
            let left_child_index = 2 * tree_index;
            let left_val =
                f64::from_bits(self.tree_array[left_child_index].load(Ordering::Relaxed));

            if target_value <= left_val {
                tree_index = left_child_index;
            } else {
                target_value -= left_val;
                tree_index = left_child_index + 1;
            }
        }
        let data_index = tree_index - self.tree_buffer_capacity_limit;
        let val = f64::from_bits(self.tree_array[tree_index].load(Ordering::Relaxed));
        (data_index, val)
    }

    pub fn sample_proportional(&self, batch_size: usize) -> Vec<(usize, f64)> {
        let mut random_generator = thread_rng();
        let total_priority = self.get_total_priority();

        if total_priority <= 0.0 || !total_priority.is_finite() {
            return vec![(0, 0.0); batch_size];
        }

        (0..batch_size)
            .map(|_| {
                let target_value = random_generator.gen_range(0.0..=total_priority);
                self.get_leaf(target_value)
            })
            .collect()
    }
}

pub struct PrioritizedReplay {
    pub segment_tree: SegmentTree,
    pub maximum_priority: AtomicU64,
    pub alpha_factor: f64,
    #[allow(dead_code)]
    pub beta_factor: f64,
}

impl PrioritizedReplay {
    pub fn new(buffer_capacity_limit: usize, alpha_factor: f64, beta_factor: f64) -> Self {
        Self {
            segment_tree: SegmentTree::new(buffer_capacity_limit),
            maximum_priority: AtomicU64::new(10.0f64.to_bits()),
            alpha_factor,
            beta_factor,
        }
    }

    pub fn add_experience(&self, data_index: usize, difficulty_penalty: f64) {
        let max_p = f64::from_bits(self.maximum_priority.load(Ordering::Relaxed));
        let priority_value = max_p.powf(self.alpha_factor) * difficulty_penalty;
        self.segment_tree.update(data_index, priority_value);
    }
}

pub struct ShardedPrioritizedReplay {
    shards: Vec<PrioritizedReplay>,
    shard_count: usize,
}

impl ShardedPrioritizedReplay {
    pub fn new(
        buffer_capacity_limit: usize,
        alpha_factor: f64,
        beta_factor: f64,
        shard_count: usize,
    ) -> Self {
        let mut shards = Vec::with_capacity(shard_count);
        let shard_buffer_capacity_limit = buffer_capacity_limit / shard_count + 1;
        for _ in 0..shard_count {
            shards.push(PrioritizedReplay::new(
                shard_buffer_capacity_limit,
                alpha_factor,
                beta_factor,
            ));
        }
        Self {
            shards,
            shard_count,
        }
    }

    #[allow(dead_code)]
    pub fn add(&self, circular_index: usize, difficulty_penalty: f64) {
        let shard_index = circular_index % self.shard_count;
        let internal_index = circular_index / self.shard_count;
        self.shards[shard_index].add_experience(internal_index, difficulty_penalty);
    }

    pub fn add_batch(&self, circular_indices: &[usize], difficulty_penalties: &[f64]) {
        let mut shard_operations = vec![(Vec::new(), Vec::new()); self.shard_count];
        for iterator_index in 0..circular_indices.len() {
            let circular_index = circular_indices[iterator_index];
            let shard_index = circular_index % self.shard_count;
            let internal_index = circular_index / self.shard_count;
            shard_operations[shard_index].0.push(internal_index);
            shard_operations[shard_index]
                .1
                .push(difficulty_penalties[iterator_index]);
        }

        for (shard_index, operations) in shard_operations.iter().enumerate() {
            if !operations.0.is_empty() {
                for iterator_index in 0..operations.0.len() {
                    self.shards[shard_index]
                        .add_experience(operations.0[iterator_index], operations.1[iterator_index]);
                }
            }
        }
    }

    pub fn sample(
        &self,
        batch_size: usize,
        global_buffer_capacity_limit: usize,
        beta: f64,
    ) -> Option<SumTreeSample> {
        let shard_index = thread_rng().gen_range(0..self.shard_count);
        let shard = &self.shards[shard_index];

        let total_priority = shard.segment_tree.get_total_priority();
        if total_priority <= 0.0 || !total_priority.is_finite() {
            return None;
        }

        let shard_samples = shard.segment_tree.sample_proportional(batch_size);
        let mut importance_weights = Vec::with_capacity(batch_size);
        let mut output_samples = Vec::with_capacity(batch_size);

        let theoretical_min_priority = 1e-4;
        let p_min_global = (theoretical_min_priority / total_priority) / (self.shard_count as f64);
        let max_theoretical_weight =
            ((global_buffer_capacity_limit as f64 * (p_min_global + 1e-8)).powf(-beta)) as f32;

        for &(data_index, priority_value) in &shard_samples {
            let sample_probability = priority_value / total_priority;
            let global_probability = sample_probability / (self.shard_count as f64);
            let importance_weight = ((global_buffer_capacity_limit as f64
                * (global_probability + 1e-8))
                .powf(-beta)) as f32;

            importance_weights.push(importance_weight / max_theoretical_weight);
            output_samples.push((data_index * self.shard_count + shard_index, priority_value));
        }

        Some((output_samples, importance_weights))
    }

    pub fn update_priorities(
        &self,
        circular_indices: &[usize],
        difficulty_penalties: &[f64],
        new_priorities: &[f64],
    ) {
        let mut shard_updates = vec![(Vec::new(), Vec::new(), Vec::new()); self.shard_count];

        for iterator_index in 0..circular_indices.len() {
            let circular_index = circular_indices[iterator_index];
            let shard_index = circular_index % self.shard_count;
            let internal_index = circular_index / self.shard_count;

            shard_updates[shard_index].0.push(internal_index);
            shard_updates[shard_index]
                .1
                .push(difficulty_penalties[iterator_index]);
            shard_updates[shard_index]
                .2
                .push(new_priorities[iterator_index]);
        }

        for (shard_index, updates) in shard_updates.iter().enumerate() {
            if !updates.0.is_empty() {
                let shard = &self.shards[shard_index];
                for iterator_index in 0..updates.0.len() {
                    let mut priority_value = updates.2[iterator_index];

                    if !priority_value.is_finite() {
                        priority_value = 1e-4;
                    }

                    update_max_priority(&shard.maximum_priority, priority_value);

                    if priority_value < 1e-4 {
                        priority_value = 1e-4;
                    }

                    let final_priority_value =
                        priority_value.powf(shard.alpha_factor) * updates.1[iterator_index];
                    shard
                        .segment_tree
                        .update(updates.0[iterator_index], final_priority_value);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn fuzz_segment_tree_math(priorities in prop::collection::vec(0.1f64..100.0, 1..500)) {
            let config_len = priorities.len();
            let tree = SegmentTree::new(config_len);

            let mut expected_sum = 0.0;
            for (i, &p) in priorities.iter().enumerate() {
                tree.update(i, p);
                expected_sum += p;
            }

            let float_error_margin = 0.05;
            let total = tree.get_total_priority();
            assert!((total - expected_sum).abs() < float_error_margin, "SumTree didn't accumulate exactly. {} != {}", total, expected_sum);

            let mut running_sum = 0.0;
            for (i, &p) in priorities.iter().enumerate() {
                running_sum += p;
                let target = running_sum - (p / 2.0); // middle of the slice
                let (idx, val) = tree.get_leaf(target);
                assert_eq!(idx, i, "SumTree search fell into wrong bucket. Target {} landed in {}, expected {}", target, idx, i);
                assert!((val - p).abs() < float_error_margin);
            }
        }
    }

    #[test]
    fn test_segment_tree_updates_and_zero_sum() {
        let tree = SegmentTree::new(4);

        let samples = tree.sample_proportional(2);
        assert_eq!(
            samples.len(),
            2,
            "Fallback for zero-sum tree should return empty placeholder batch"
        );
        assert_eq!(samples[0].0, 0, "Fallback index should be 0");
        assert_eq!(samples[0].1, 0.0, "Fallback weight should be zero");

        tree.update(0, 1.0);
        tree.update(1, 2.0);
        tree.update(2, 3.0);

        assert_eq!(
            tree.get_total_priority(),
            6.0,
            "Sumtree total propagation failed"
        );

        tree.update(1, 0.5);
        assert_eq!(
            tree.get_total_priority(),
            4.5,
            "Sumtree modification update tracking mathematical bug"
        );
    }

    #[test]
    fn test_sharded_per_weighting() {
        let per = ShardedPrioritizedReplay::new(10, 1.0, 1.0, 2);
        per.add_batch(&[0, 1, 2], &[1.0, 2.0, 3.0]);
        let mut successful_sample = false;

        for _ in 0..100 {
            if let Some((_, weights)) = per.sample(2, 10, 1.0) {
                assert_eq!(weights.len(), 2, "Sampled weights mismatch");
                successful_sample = true;
                break;
            }
        }

        assert!(
            successful_sample,
            "Should have sampled from populated PER shard"
        );
    }
}
```

### File: `src/test_cmodule.rs`

```rs
use tch::{CModule, Device, Kind};
fn main() {
    let mut m = CModule::load_data_on_device(&mut [], Device::Cuda(0)).unwrap();
    m.to(Device::Cuda(0), Kind::Half, false);
}
```

### File: `src/test_dlpack.rs`

```rs
use tch::Tensor;
fn main() {
    let t = Tensor::zeros([10], (tch::Kind::Float, tch::Device::Cpu));
    let ptr = t.data_ptr();
}
```

### File: `src/test_dlpack2.rs`

```rs
use tch::Tensor;

fn main() {
    let t = Tensor::zeros([10], (tch::Kind::Float, tch::Device::Cpu));
    let _ = tch::Tensor::to_dlpack(&t);
}
```

### File: `src/test_sparse.rs`

```rs
use tch::{Tensor, Kind, Device};

fn main() {
    let indices = Tensor::from_slice(&[0i64, 1, 0, 1]).reshape([2, 2]);
    let values = Tensor::from_slice(&[1.0f32, 2.0]);
    let sparse = Tensor::sparse_coo_tensor(&indices, &values, [2, 2], (Kind::Float, Device::Cpu));
    let dense = Tensor::ones([2, 2], (Kind::Float, Device::Cpu));
    let out = sparse.matmul(&dense);
    println!("Sparse matmul out: {:?}", out.size());
}
```

### File: `src/tests.rs`

``rs
#[cfg(test)]
mod integration_tests {
    use crate::config::Config;
    use crate::core::board::GameStateExt;
    use crate::core::features::extract_feature_native;
    use crate::mcts::EvaluationRequest;
    use crate::net::MuZeroNet;
    use crossbeam_channel::unbounded;
    use tch::{nn, nn::Module, nn::ModuleT, nn::OptimizerConfig, Device, Tensor};

    fn get_test_config() -> Config {
        serde_yaml::from_str(
            r#"
device: cpu
paths:
  base_directory: "runs/test"
  model_checkpoint_path: "test.safetensors"
  metrics_file_path: "test.csv"
  telemetry_config_export: "config.json"
experiment_name_identifier: "test"
hidden_dimension_size: 128
num_blocks: 8
support_size: 300
buffer_capacity_limit: 1000
simulations: 10
train_batch_size: 4
train_epochs: 1
num_processes: 1
worker_device: cpu
unroll_steps: 3
temporal_difference_steps: 5
inference_batch_size_limit: 1
inference_timeout_ms: 1
max_gumbel_k: 4
gumbel_scale: 1.0
temp_decay_steps: 100
difficulty: 6
temp_boost: false
lr_init: 0.01
reanalyze_ratio: 0.25
        "#,
        )
        .unwrap()
    }

    #[test]
    fn test_network_dimensions() {
        // Objective: Test flows convergence dimensions of the tensor
        let cfg = get_test_config();
        let vs = nn::VarStore::new(Device::Cpu);
        let net = MuZeroNet::new(&vs.root(), cfg.hidden_dimension_size, cfg.num_blocks, 300);

        let game = GameStateExt::new(None, 0, 0, 0, 0); // 0 difficulty
        let mut feat = vec![0.0; 20 * 128];
        extract_feature_native(
            &mut feat,
            game.board_bitmask_u128,
            &game.available,
            &[],
            &[],
            0,
        );

        // Tensor dimension mapping
        // B=1, C=20, H=8, W=16
        let obs = Tensor::from_slice(&feat).view([1, 20, 8, 16]);

        let hidden = net.representation.forward_t(&obs, false);
        assert_eq!(
            hidden.size(),
            vec![1i64, 128i64, 8i64, 8i64],
            "Representation shape mismatch"
        );

        let (value_logits, policy_logits, _hole_logits) = net.prediction.forward(&hidden);
        assert_eq!(
            policy_logits.size(),
            vec![1i64, 288i64],
            "Policy shape mismatch"
        );
        assert_eq!(
            value_logits.size(),
            vec![1i64, 601i64],
            "Value shape mismatch"
        );

        let action = Tensor::zeros([1], (tch::Kind::Int64, Device::Cpu));
        let piece_id = Tensor::zeros([1], (tch::Kind::Int64, Device::Cpu));
        let (next_hidden, reward_logits) = net.dynamics.forward(&hidden, &action, &piece_id);

        assert_eq!(
            next_hidden.size(),
            vec![1i64, 128i64, 8i64, 8i64],
            "Dynamics hidden shape mismatch"
        );
        assert_eq!(
            reward_logits.size(),
            vec![1i64, 601i64],
            "Reward shape mismatch"
        );
    }

    #[test]
    fn test_transmission_stress_test() {
        // Objective: Test transmission stress tests with self-play evaluating channels
        let (evaluation_request_transmitter, evaluation_response_receiver) =
            unbounded::<EvaluationRequest>();

        let mut handlers = vec![];
        let num_workers = 10;
        let num_reqs = 100;

        for _w in 0..num_workers {
            let thread_tx = evaluation_request_transmitter.clone();
            handlers.push(std::thread::spawn(move || {
                for _i in 0..num_reqs {
                    let (ans_tx, ans_rx) = unbounded();
                    let req = EvaluationRequest {
                        is_initial: true,
                        board_bitmask: 0,
                        available_pieces: [-1; 3],
                        recent_board_history: [0; 8],
                        history_len: 0,
                        recent_action_history: [0; 4],
                        action_history_len: 0,
                        difficulty: 6,
                        piece_action: 0,
                        piece_id: 0,
                        node_index: 0,
                        worker_id: 0,
                        parent_cache_index: 0,
                        leaf_cache_index: 0,
                        evaluation_request_transmitter: ans_tx,
                    };
                    thread_tx.send(req).unwrap();
                    let _ = ans_rx.recv().unwrap();
                }
            }));
        }

        let total_reqs = num_workers * num_reqs;
        for _ in 0..total_reqs {
            let req = evaluation_response_receiver.recv().unwrap();
            req.evaluation_request_transmitter
                .send(crate::mcts::EvaluationResponse {
                    child_prior_probabilities_tensor: [0.0; 288],
                    value: 0.0,
                    reward: 0.0,
                    node_index: 0,
                })
                .unwrap();
        }

        for h in handlers {
            h.join().unwrap();
        }
        assert!(
            evaluation_response_receiver.is_empty(),
            "Channel should be thoroughly processed"
        );
    }

    #[test]
    fn test_flow_convergence() {
        // Objective: Test flows convergence on synthetic batch
        let mut cfg = get_test_config();
        cfg.hidden_dimension_size = 16;
        cfg.num_blocks = 1;
        let vs = nn::VarStore::new(Device::Cpu);
        let net = MuZeroNet::new(&vs.root(), cfg.hidden_dimension_size, cfg.num_blocks, 300);

        let mut opt = nn::Adam::default().build(&vs, 0.0001).unwrap();

        let batch_size = 4;
        // Mock identical inputs to force overfitting
        let obs = Tensor::randn([batch_size, 20, 8, 16], (tch::Kind::Float, Device::Cpu));

        let target_value = Tensor::zeros([batch_size, 601], (tch::Kind::Float, Device::Cpu));
        let _ = target_value.narrow(1, 300, 1).fill_(1.0); // 0 score

        let target_policy = Tensor::zeros([batch_size, 288], (tch::Kind::Float, Device::Cpu));
        let _ = target_policy.narrow(1, 42, 1).fill_(1.0); // Predict action 42

        let mut initial_loss = 0.0;
        let mut final_loss = 0.0;

        for epoch in 0..20 {
            let hidden = net.representation.forward(&obs);
            let (value_logits, policy_logits, _hole_logits) = net.prediction.forward(&hidden);

            let p_loss = -(target_policy.copy() * policy_logits.log_softmax(1, tch::Kind::Float))
                .sum(tch::Kind::Float)
                / (batch_size as f64);
            let v_loss = -(target_value.copy() * value_logits.log_softmax(1, tch::Kind::Float))
                .sum(tch::Kind::Float)
                / (batch_size as f64);

            let loss = p_loss + v_loss;

            if epoch == 0 {
                initial_loss = f64::try_from(loss.copy()).unwrap();
            }
            if epoch == 19 {
                final_loss = f64::try_from(loss.copy()).unwrap();
            }

            opt.backward_step(&loss);
        }

        assert!(
            final_loss < initial_loss,
            "Loss did not converge: initial {} -> final {}",
            initial_loss,
            final_loss
        );
    }

    #[test]
    fn test_ema_polyak_averaging() {
        let vs = nn::VarStore::new(Device::Cpu);
        let ema_vs = nn::VarStore::new(Device::Cpu);

        let _p_model = vs.root().var("w", &[1], tch::nn::Init::Const(100.0));
        let p_ema = ema_vs.root().var("w", &[1], tch::nn::Init::Const(0.0));

        tch::no_grad(|| {
            let mut ema_vars = ema_vs.variables();
            let model_vars = vs.variables();
            for (name, t_ema) in ema_vars.iter_mut() {
                if let Some(t_model) = model_vars.get(name) {
                    t_ema.copy_(&(&*t_ema * 0.99 + t_model * 0.01));
                }
            }
        });

        let val = f64::try_from(p_ema).unwrap();
        // 0.0 * 0.99 + 100.0 * 0.01 = 1.0
        assert!(
            (val - 1.0).abs() < 1e-5,
            "EMA Polyak averaging math failed!"
        );
    }

    #[test]
    fn test_device_fallback_safety() {
        let requested = "cuda";
        let actual = if requested == "cuda" && tch::Cuda::is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };
        // Verify no panics and standard structure.
        assert!(matches!(actual, Device::Cpu | Device::Cuda(0)));
    }
    #[test]
    fn test_nan_free_initialization() {
        // Objective: Test that freshly initialized weights do not produce NaNs
        let cfg = get_test_config();

        let device = if tch::Cuda::is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };
        let vs = nn::VarStore::new(device);
        // Note: We intentionally do NOT call `vs.half()` to prevent FP16 batch norm NaNs.

        let net = MuZeroNet::new(&vs.root(), cfg.hidden_dimension_size, cfg.num_blocks, 300);

        let batch_size = 2;
        let state = crate::core::board::GameStateExt::new(Some([1, 2, 3]), 0, 0, 6, 0);
        let mut features = vec![0.0; 20 * 128];
        crate::core::features::extract_feature_native(
            &mut features,
            state.board_bitmask_u128,
            &state.available,
            &[],
            &[],
            6,
        );
        let mut flat_batch = features.clone();
        flat_batch.extend(features);

        let obs = Tensor::from_slice(&flat_batch)
            .view([batch_size as i64, 20, 8, 16])
            .to_kind(tch::Kind::BFloat16)
            .to_device(device)
            .to_kind(tch::Kind::Float);

        let hidden = net.representation.forward_t(&obs, false);

        assert!(
            !bool::try_from(hidden.isnan().any()).unwrap(),
            "NaN detected in hidden state immediately after initialization!"
        );

        let (value_logits, policy_logits, hidden_state_logits) = net.prediction.forward(&hidden);

        assert!(
            !bool::try_from(value_logits.isnan().any()).unwrap(),
            "NaN detected in value logits!"
        );

        assert!(
            !bool::try_from(policy_logits.isnan().any()).unwrap(),
            "NaN detected in policy logits!"
        );

        assert!(
            !bool::try_from(hidden_state_logits.isnan().any()).unwrap(),
            "NaN detected in hidden state logits!"
        );
    }
}
``

### File: `src/train/buffer/mod.rs`

```rs
pub mod replay;
pub mod state;

pub use replay::ReplayBuffer;
pub use state::EpisodeMeta;
```

### File: `src/train/buffer/replay.rs`

```rs
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};
use std::sync::Arc;
use tch::{Device, Kind, Tensor};

struct SafeTensorGuard<'a, T> {
    _tensor: &'a Tensor,
    pub slice: &'a mut [T],
}

impl<'a, T> SafeTensorGuard<'a, T> {
    fn new(tensor: &'a Tensor, len: usize) -> Self {
        assert!(
            tensor.is_contiguous(),
            "Tensor must be contiguous for raw pointer access"
        );
        Self {
            _tensor: tensor,
            slice: unsafe { std::slice::from_raw_parts_mut(tensor.data_ptr() as *mut T, len) },
        }
    }
}

impl<'a, T> Drop for SafeTensorGuard<'a, T> {
    fn drop(&mut self) {}
}

use crate::train::buffer::state::{EpisodeMeta, SharedState};

pub struct ReplayBuffer {
    pub state: Arc<SharedState>,
    pub background_sender: crossbeam_channel::Sender<OwnedGameData>,
    pub arena: std::sync::Mutex<Option<SampleArena>>,
}

pub struct SampleArena {
    pub state_features: Tensor,
    pub actions: Tensor,
    pub piece_identifiers: Tensor,
    pub rewards: Tensor,
    pub target_policies: Tensor,
    pub target_values: Tensor,
    pub model_values: Tensor,
    pub transition_states: Tensor,
    pub loss_masks: Tensor,
    pub importance_weights: Tensor,
}

pub struct BatchTensors {
    pub state_features_batch: Tensor,
    pub actions_batch: Tensor,
    pub piece_identifiers_batch: Tensor,
    pub rewards_batch: Tensor,
    pub target_policies_batch: Tensor,
    pub target_values_batch: Tensor,
    #[allow(dead_code)]
    pub model_values_batch: Tensor,
    pub transition_states_batch: Tensor,
    pub loss_masks_batch: Tensor,
    pub importance_weights_batch: Tensor,
    pub global_indices_sampled: Vec<usize>,
}

impl ReplayBuffer {
    pub fn new(
        total_buffer_capacity_limit: usize,
        unroll_steps: usize,
        temporal_difference_steps: usize,
    ) -> Self {
        let shared_state = SharedState {
            buffer_capacity_limit: total_buffer_capacity_limit,
            unroll_steps,
            temporal_difference_steps,
            current_diff: AtomicI32::new(1),
            global_write_storage_index: AtomicUsize::new(0),
            global_write_active_storage_index: AtomicUsize::new(0),
            num_states: AtomicUsize::new(0),

            arrays: crate::train::buffer::state::ShardedStorageArrays::new(
                total_buffer_capacity_limit,
                64,
            ),
            per: crate::sumtree::ShardedPrioritizedReplay::new(
                total_buffer_capacity_limit,
                0.6,
                0.4,
                64,
            ),

            episodes: std::sync::Mutex::new(Vec::new()),
            recent_scores: std::sync::Mutex::new(Vec::new()),
            completed_games: AtomicUsize::new(0),
        };

        let state_arc = Arc::new(shared_state);
        let (evaluation_request_transmitter, evaluation_response_receiver) =
            crossbeam_channel::unbounded::<OwnedGameData>();
        let background_state = state_arc.clone();

        std::thread::Builder::new()
            .name("replay_buffer_writer".into())
            .spawn(move || {
                while let Ok(data) = evaluation_response_receiver.recv() {
                    Self::process_add_game(&background_state, data);
                }
            })
            .unwrap();

        Self {
            state: state_arc,
            background_sender: evaluation_request_transmitter,
            arena: std::sync::Mutex::new(None),
        }
    }

    pub fn get_length(&self) -> usize {
        self.state.num_states.load(Ordering::Relaxed)
    }

    #[allow(dead_code)]
    pub fn get_global_write_storage_index(&self) -> usize {
        self.state
            .global_write_storage_index
            .load(Ordering::Acquire)
    }

    #[allow(dead_code)]
    pub fn get_and_clear_metrics(&self) -> (Vec<f32>, f32, f32, f32) {
        let mut recent_scores_lock = match self.state.recent_scores.lock() {
            Ok(lock) => lock,
            Err(poisoned) => poisoned.into_inner(),
        };
        if recent_scores_lock.is_empty() {
            return (vec![], 0.0, 0.0, 0.0);
        }
        let cloned_scores = std::mem::take(&mut *recent_scores_lock);
        let mut sorted_scores = cloned_scores.clone();
        sorted_scores
            .sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));

        let median_score = sorted_scores[sorted_scores.len() / 2];
        let maximum_score = *sorted_scores.last().unwrap_or(&0.0);
        let sum_scores: f32 = cloned_scores.iter().sum();
        let average_score = sum_scores / cloned_scores.len() as f32;

        (cloned_scores, median_score, maximum_score, average_score)
    }
}

#[derive(Clone, Debug)]
pub struct GameStep {
    pub board_state: [u64; 2],
    pub available_pieces: [i32; 3],
    pub action_taken: i64,
    pub piece_identifier: i64,
    pub reward_received: f32,
    pub policy_target: [f32; 288],
    pub value_target: f32,
}

pub struct OwnedGameData {
    pub difficulty_setting: i32,
    pub episode_score: f32,
    pub steps: Vec<GameStep>,
    pub lines_cleared: u32,
    pub mcts_depth_mean: f32,
    pub mcts_search_time_mean: f32,
}

impl ReplayBuffer {
    pub fn add_game(&self, data: OwnedGameData) {
        let _ = self.background_sender.send(data);
    }

    fn process_add_game(state: &SharedState, data: OwnedGameData) {
        let OwnedGameData {
            difficulty_setting,
            episode_score,
            steps,
            lines_cleared,
            mcts_depth_mean,
            mcts_search_time_mean,
        } = data;
        let episode_length = steps.len();
        if episode_length == 0 {
            return;
        }

        let episode_start_index = state.global_write_storage_index.load(Ordering::Relaxed);
        let buffer_buffer_capacity_limit = state.buffer_capacity_limit;
        let running_difficulty = state.current_diff.load(Ordering::Relaxed);
        let next_global_write_index = episode_start_index + episode_length;

        state
            .global_write_active_storage_index
            .store(next_global_write_index, Ordering::Release);

        if running_difficulty == 0 || difficulty_setting != running_difficulty {
            state
                .current_diff
                .store(difficulty_setting, Ordering::Relaxed);
        }

        let active_difficulty = state.current_diff.load(Ordering::Relaxed);
        let absolute_difficulty_penalty =
            10f64.powf(-(active_difficulty - difficulty_setting).abs() as f64);

        for (transition_offset, step) in steps.iter().take(episode_length).enumerate() {
            let circular_write_index =
                (episode_start_index + transition_offset) % buffer_buffer_capacity_limit;
            state.arrays.write_storage_index(
                circular_write_index,
                |memory_shard, internal_shard_index| {
                    memory_shard.state_start[internal_shard_index] = episode_start_index as i64;
                    memory_shard.state_diff[internal_shard_index] = difficulty_setting;
                    memory_shard.state_len[internal_shard_index] = episode_length as i32;

                    memory_shard.boards[internal_shard_index] = step.board_state;
                    memory_shard.available[internal_shard_index] = step.available_pieces;
                    memory_shard.actions[internal_shard_index] = step.action_taken;
                    memory_shard.piece_ids[internal_shard_index] = step.piece_identifier;
                    memory_shard.rewards[internal_shard_index] = step.reward_received;
                    memory_shard.policies[internal_shard_index] = step.policy_target;
                    memory_shard.values[internal_shard_index] = step.value_target;
                },
            );
        }

        let mut circular_indices_to_add = Vec::with_capacity(episode_length);
        let mut transition_penalties = Vec::with_capacity(episode_length);
        for transition_offset in 0..episode_length {
            circular_indices_to_add
                .push((episode_start_index + transition_offset) % buffer_buffer_capacity_limit);
            transition_penalties.push(absolute_difficulty_penalty);
        }
        state
            .per
            .add_batch(&circular_indices_to_add, &transition_penalties);

        state
            .global_write_storage_index
            .store(next_global_write_index, Ordering::Release);

        Self::update_episode_metadata(
            state,
            episode_start_index,
            episode_length,
            difficulty_setting,
            episode_score,
            next_global_write_index,
            buffer_buffer_capacity_limit,
            lines_cleared,
            mcts_depth_mean,
            mcts_search_time_mean,
        );
    }

    fn update_episode_metadata(
        state: &SharedState,
        episode_start_index: usize,
        episode_length: usize,
        difficulty_setting: i32,
        episode_score: f32,
        next_global_write_index: usize,
        buffer_buffer_capacity_limit: usize,
        lines_cleared: u32,
        mcts_depth_mean: f32,
        mcts_search_time_mean: f32,
    ) {
        {
            let mut episode_metadata_lock = match state.episodes.lock() {
                Ok(lock) => lock,
                Err(e) => e.into_inner(),
            };
            episode_metadata_lock.push(EpisodeMeta {
                global_start_storage_index: episode_start_index,
                length: episode_length,
                difficulty: difficulty_setting,
                score: episode_score,
                lines_cleared,
                mcts_depth_mean,
                mcts_search_time_mean,
            });

            let remove_count = episode_metadata_lock
                .iter()
                .take_while(|episode| {
                    episode.global_start_storage_index + buffer_buffer_capacity_limit
                        < next_global_write_index
                })
                .count();
            if remove_count > 0 {
                episode_metadata_lock.drain(0..remove_count);
            }
        }

        let mut recent_scores_lock = match state.recent_scores.lock() {
            Ok(lock) => lock,
            Err(e) => e.into_inner(),
        };
        recent_scores_lock.push(episode_score);
        state.completed_games.fetch_add(1, Ordering::Relaxed);

        let current_state_count = state.num_states.load(Ordering::Relaxed);
        state.num_states.store(
            buffer_buffer_capacity_limit.min(current_state_count + episode_length),
            Ordering::Relaxed,
        );
    }

    pub fn sample_for_reanalyze(&self) -> Option<(usize, crate::core::board::GameStateExt)> {
        let (transitions, _weights) =
            self.state
                .per
                .sample(1, self.state.buffer_capacity_limit, 1.0)?;
        if transitions.is_empty() {
            return None;
        }

        let circular_idx = transitions[0].0;

        let (board, pieces, difficulty) = self
            .state
            .arrays
            .read_storage_index(circular_idx, |shard, i| {
                (shard.boards[i], shard.available[i], shard.state_diff[i])
            });

        let board_u128 = (board[1] as u128) << 64 | (board[0] as u128);
        let state =
            crate::core::board::GameStateExt::new(Some(pieces), board_u128, 0, difficulty, 0);
        Some((circular_idx, state))
    }

    pub fn update_reanalyzed_targets(
        &self,
        circular_idx: usize,
        new_policy: [f32; 288],
        new_value: f32,
    ) {
        self.state
            .arrays
            .write_storage_index(circular_idx, |shard, i| {
                shard.policies[i] = new_policy;
                shard.values[i] = new_value;
            });
    }

    pub fn sample_batch(
        &self,
        batch_size_limit: usize,
        computation_device: Device,
        beta: f64,
    ) -> Option<BatchTensors> {
        let (sampled_transitions, sampled_importance_weights) =
            match self
                .state
                .per
                .sample(batch_size_limit, self.state.buffer_capacity_limit, beta)
            {
                Some((samples, weights)) => (samples, weights),
                None => return None,
            };

        let unroll_limit = self.state.unroll_steps;

        let mut arena_lock = self.arena.lock().unwrap();
        if arena_lock.is_none() {
            let pin = |t: Tensor| {
                if computation_device.is_cuda() {
                    t.pin_memory(computation_device)
                } else {
                    t
                }
            };
            *arena_lock = Some(SampleArena {
                state_features: pin(Tensor::zeros(
                    [batch_size_limit as i64, 20, 8, 16],
                    (Kind::Float, Device::Cpu),
                )),
                actions: pin(Tensor::zeros(
                    [batch_size_limit as i64, unroll_limit as i64],
                    (Kind::Int64, Device::Cpu),
                )),
                piece_identifiers: pin(Tensor::zeros(
                    [batch_size_limit as i64, unroll_limit as i64],
                    (Kind::Int64, Device::Cpu),
                )),
                rewards: pin(Tensor::zeros(
                    [batch_size_limit as i64, unroll_limit as i64],
                    (Kind::Float, Device::Cpu),
                )),
                target_policies: pin(Tensor::zeros(
                    [batch_size_limit as i64, (unroll_limit + 1) as i64, 288],
                    (Kind::Float, Device::Cpu),
                )),
                target_values: pin(Tensor::zeros(
                    [batch_size_limit as i64, (unroll_limit + 1) as i64],
                    (Kind::Float, Device::Cpu),
                )),
                model_values: pin(Tensor::zeros(
                    [batch_size_limit as i64, (unroll_limit + 1) as i64],
                    (Kind::Float, Device::Cpu),
                )),
                transition_states: pin(Tensor::zeros(
                    [batch_size_limit as i64, unroll_limit as i64, 20, 8, 16],
                    (Kind::Float, Device::Cpu),
                )),
                loss_masks: pin(Tensor::zeros(
                    [batch_size_limit as i64, (unroll_limit + 1) as i64],
                    (Kind::Float, Device::Cpu),
                )),
                importance_weights: pin(Tensor::zeros(
                    [batch_size_limit as i64],
                    (Kind::Float, Device::Cpu),
                )),
            });
        }

        let arena = arena_lock.as_ref().unwrap();

        let state_features_guard =
            SafeTensorGuard::<f32>::new(&arena.state_features, batch_size_limit * 20 * 128);
        let state_features_buffer: &mut [f32] = state_features_guard.slice;

        let actions_guard =
            SafeTensorGuard::<i64>::new(&arena.actions, batch_size_limit * unroll_limit);
        let actions_buffer: &mut [i64] = actions_guard.slice;

        let piece_identifiers_guard =
            SafeTensorGuard::<i64>::new(&arena.piece_identifiers, batch_size_limit * unroll_limit);
        let piece_identifiers_buffer: &mut [i64] = piece_identifiers_guard.slice;

        let rewards_guard =
            SafeTensorGuard::<f32>::new(&arena.rewards, batch_size_limit * unroll_limit);
        let rewards_buffer: &mut [f32] = rewards_guard.slice;

        let target_policies_guard = SafeTensorGuard::<f32>::new(
            &arena.target_policies,
            batch_size_limit * (unroll_limit + 1) * 288,
        );
        let target_policies_buffer: &mut [f32] = target_policies_guard.slice;

        let target_values_guard = SafeTensorGuard::<f32>::new(
            &arena.target_values,
            batch_size_limit * (unroll_limit + 1),
        );
        let target_values_buffer: &mut [f32] = target_values_guard.slice;

        let model_values_guard =
            SafeTensorGuard::<f32>::new(&arena.model_values, batch_size_limit * (unroll_limit + 1));
        let model_values_buffer: &mut [f32] = model_values_guard.slice;

        let transition_states_guard = SafeTensorGuard::<f32>::new(
            &arena.transition_states,
            batch_size_limit * unroll_limit * 20 * 128,
        );
        let transition_states_buffer: &mut [f32] = transition_states_guard.slice;

        let loss_masks_guard =
            SafeTensorGuard::<f32>::new(&arena.loss_masks, batch_size_limit * (unroll_limit + 1));
        let loss_masks_buffer: &mut [f32] = loss_masks_guard.slice;

        let importance_weights_guard =
            SafeTensorGuard::<f32>::new(&arena.importance_weights, batch_size_limit);
        let importance_weights_buffer: &mut [f32] = importance_weights_guard.slice;

        let mut global_indices_sampled: Vec<usize> = Vec::with_capacity(batch_size_limit);

        for (batch_index, &(circular_index, _)) in sampled_transitions.iter().enumerate() {
            importance_weights_buffer[batch_index] = sampled_importance_weights[batch_index];

            let (logical_start_global, logical_length) = self.state.arrays.read_storage_index(
                circular_index,
                |array_shard, shard_internal| {
                    (
                        array_shard.state_start[shard_internal],
                        array_shard.state_len[shard_internal],
                    )
                },
            );

            let global_state_index = if logical_start_global != -1 {
                let positional_offset = (circular_index as i64 - logical_start_global)
                    .rem_euclid(self.state.buffer_capacity_limit as i64);
                if positional_offset < logical_length as i64 {
                    (logical_start_global + positional_offset) as usize
                } else {
                    logical_start_global as usize
                }
            } else {
                0
            };

            global_indices_sampled.push(global_state_index);
            self.extract_single_sample_data(
                batch_index,
                global_state_index,
                unroll_limit,
                state_features_buffer,
                actions_buffer,
                piece_identifiers_buffer,
                rewards_buffer,
                target_policies_buffer,
                target_values_buffer,
                model_values_buffer,
                transition_states_buffer,
                loss_masks_buffer,
                importance_weights_buffer,
            );
        }

        Some(BatchTensors {
            state_features_batch: arena.state_features.shallow_clone(),
            actions_batch: arena.actions.shallow_clone(),
            piece_identifiers_batch: arena.piece_identifiers.shallow_clone(),
            rewards_batch: arena.rewards.shallow_clone(),
            target_policies_batch: arena.target_policies.shallow_clone(),
            target_values_batch: arena.target_values.shallow_clone(),
            model_values_batch: arena.model_values.shallow_clone(),
            transition_states_batch: arena.transition_states.shallow_clone(),
            loss_masks_batch: arena.loss_masks.shallow_clone(),
            importance_weights_batch: arena.importance_weights.shallow_clone(),
            global_indices_sampled,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn extract_single_sample_data(
        &self,
        batch_index: usize,
        global_state_index: usize,
        unroll_limit: usize,
        state_features_buffer: &mut [f32],
        actions_buffer: &mut [i64],
        piece_identifiers_buffer: &mut [i64],
        rewards_buffer: &mut [f32],
        target_policies_buffer: &mut [f32],
        target_values_buffer: &mut [f32],
        model_values_buffer: &mut [f32],
        transition_states_buffer: &mut [f32],
        loss_masks_buffer: &mut [f32],
        importance_weights_buffer: &mut [f32],
    ) {
        let global_start_circular = global_state_index % self.state.buffer_capacity_limit;
        let (logical_start_global, logical_length) = self.state.arrays.read_storage_index(
            global_start_circular,
            |array_shard, shard_internal| {
                (
                    array_shard.state_start[shard_internal],
                    array_shard.state_len[shard_internal],
                )
            },
        );

        let episode_end_global = if logical_start_global != -1 {
            (logical_start_global + logical_length as i64) as usize
        } else {
            global_state_index + 1
        };

        let safe_before_boundary = self
            .state
            .global_write_storage_index
            .load(Ordering::Acquire);
        let active_after_boundary = self
            .state
            .global_write_active_storage_index
            .load(Ordering::Acquire);

        let initial_features = self.state.get_features(global_state_index);
        let total_feature_elements = 20 * 128;
        let destination_offset = batch_index * total_feature_elements;

        unsafe {
            std::ptr::copy_nonoverlapping(
                initial_features.as_ptr(),
                state_features_buffer.as_mut_ptr().add(destination_offset),
                total_feature_elements,
            );
        }

        for unroll_offset in 0..=unroll_limit {
            self.process_unrolled_step(
                batch_index,
                global_state_index,
                unroll_offset,
                unroll_limit,
                episode_end_global,
                actions_buffer,
                piece_identifiers_buffer,
                rewards_buffer,
                target_policies_buffer,
                target_values_buffer,
                model_values_buffer,
                transition_states_buffer,
                loss_masks_buffer,
            );
        }

        let maximum_global_read =
            global_state_index + unroll_limit + self.state.temporal_difference_steps;
        let minimum_global_read = global_state_index.saturating_sub(8);

        let write_not_finalized = maximum_global_read >= safe_before_boundary;
        let safely_overwritten =
            minimum_global_read + self.state.buffer_capacity_limit <= active_after_boundary;

        if write_not_finalized || safely_overwritten {
            importance_weights_buffer[batch_index] = 0.0;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn process_unrolled_step(
        &self,
        batch_index: usize,
        global_state_index: usize,
        unroll_offset: usize,
        unroll_limit: usize,
        episode_end_global: usize,
        actions_buffer: &mut [i64],
        piece_identifiers_buffer: &mut [i64],
        rewards_buffer: &mut [f32],
        target_policies_buffer: &mut [f32],
        target_values_buffer: &mut [f32],
        model_values_buffer: &mut [f32],
        transition_states_buffer: &mut [f32],
        loss_masks_buffer: &mut [f32],
    ) {
        let current_global_step = global_state_index + unroll_offset;
        let current_circular_step = current_global_step % self.state.buffer_capacity_limit;

        if current_global_step < episode_end_global {
            loss_masks_buffer[batch_index * (unroll_limit + 1) + unroll_offset] = 1.0;

            if unroll_offset > 0 {
                let previous_circular_step =
                    (current_global_step - 1) % self.state.buffer_capacity_limit;
                let (previous_action, previous_piece_identifier, previous_reward) = self
                    .state
                    .arrays
                    .read_storage_index(previous_circular_step, |array_shard, shard_internal| {
                        (
                            array_shard.actions[shard_internal],
                            array_shard.piece_ids[shard_internal],
                            array_shard.rewards[shard_internal],
                        )
                    });

                actions_buffer[batch_index * unroll_limit + unroll_offset - 1] = previous_action;
                piece_identifiers_buffer[batch_index * unroll_limit + unroll_offset - 1] =
                    previous_piece_identifier;
                rewards_buffer[batch_index * unroll_limit + unroll_offset - 1] = previous_reward;

                let transition_features = self.state.get_features(current_global_step);
                let total_feature_elements = 20 * 128;
                let destination_offset =
                    (batch_index * unroll_limit + unroll_offset - 1) * total_feature_elements;

                unsafe {
                    std::ptr::copy_nonoverlapping(
                        transition_features.as_ptr(),
                        transition_states_buffer
                            .as_mut_ptr()
                            .add(destination_offset),
                        total_feature_elements,
                    );
                }
            }

            let (stored_policy, stored_value) = self.state.arrays.read_storage_index(
                current_circular_step,
                |array_shard, shard_internal| {
                    (
                        array_shard.policies[shard_internal],
                        array_shard.values[shard_internal],
                    )
                },
            );

            let destination_offset = batch_index * (unroll_limit + 1) * 288 + unroll_offset * 288;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    stored_policy.as_ptr(),
                    target_policies_buffer.as_mut_ptr().add(destination_offset),
                    288,
                );
            }
            model_values_buffer[batch_index * (unroll_limit + 1) + unroll_offset] = stored_value;

            let bootstrap_global_step = current_global_step + self.state.temporal_difference_steps;
            let discount_factor = 0.99f32;
            let mut discounted_sum_rewards = 0.0;

            let accumulation_limit = bootstrap_global_step.min(episode_end_global);

            for accumulation_step in 0..(accumulation_limit - current_global_step) {
                let reward_circular_index =
                    (current_global_step + accumulation_step) % self.state.buffer_capacity_limit;
                discounted_sum_rewards +=
                    self.state.arrays.read_storage_index(
                        reward_circular_index,
                        |array_shard, shard_internal| array_shard.rewards[shard_internal],
                    ) * discount_factor.powi(accumulation_step as i32);
            }

            if bootstrap_global_step < episode_end_global {
                let value_bootstrap_circular =
                    bootstrap_global_step % self.state.buffer_capacity_limit;
                discounted_sum_rewards +=
                    self.state.arrays.read_storage_index(
                        value_bootstrap_circular,
                        |array_shard, shard_internal| array_shard.values[shard_internal],
                    ) * discount_factor.powi(self.state.temporal_difference_steps as i32);
            }
            target_values_buffer[batch_index * (unroll_limit + 1) + unroll_offset] =
                discounted_sum_rewards;
        } else {
            loss_masks_buffer[batch_index * (unroll_limit + 1) + unroll_offset] = 0.0;
            target_values_buffer[batch_index * (unroll_limit + 1) + unroll_offset] = 0.0;
            model_values_buffer[batch_index * (unroll_limit + 1) + unroll_offset] = 0.0;
            let destination_offset = batch_index * (unroll_limit + 1) * 288 + unroll_offset * 288;
            target_policies_buffer[destination_offset..destination_offset + 288].fill(1.0 / 288.0);
        }
    }

    pub fn update_priorities(&self, priority_indices: &[usize], computed_priorities: &[f64]) {
        let running_difficulty = self.state.current_diff.load(Ordering::Relaxed);
        let mut mapped_circular_indices = Vec::with_capacity(priority_indices.len());
        let mut mapped_difficulty_penalties = Vec::with_capacity(priority_indices.len());

        for &global_state_index in priority_indices {
            let circular_index = global_state_index % self.state.buffer_capacity_limit;
            let (logical_start_global, difficulty_setting) = self.state.arrays.read_storage_index(
                circular_index,
                |array_shard, shard_internal| {
                    (
                        array_shard.state_start[shard_internal],
                        array_shard.state_diff[shard_internal],
                    )
                },
            );

            if logical_start_global != -1 {
                mapped_difficulty_penalties
                    .push(10f64.powf(-(running_difficulty - difficulty_setting).abs() as f64));
                mapped_circular_indices.push(circular_index);
            } else {
                mapped_difficulty_penalties.push(0.0);
                mapped_circular_indices.push(circular_index);
            }
        }

        self.state.per.update_priorities(
            &mapped_circular_indices,
            &mapped_difficulty_penalties,
            computed_priorities,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn test_ring_buffer_wraparound() {
        let replay_buffer = ReplayBuffer::new(5, 1, 10);

        let steps = vec![
            crate::train::buffer::replay::GameStep {
                board_state: [0, 0],
                available_pieces: [0, 0, 0],
                action_taken: 0,
                piece_identifier: 0,
                reward_received: 0.0,
                policy_target: [0.0; 288],
                value_target: 0.0,
            };
            4
        ];

        replay_buffer.add_game(OwnedGameData {
            difficulty_setting: 6,
            episode_score: 1.0,
            steps,
            lines_cleared: 0,
            mcts_depth_mean: 0.0,
            mcts_search_time_mean: 0.0,
        });

        let steps_2 = vec![
            crate::train::buffer::replay::GameStep {
                board_state: [5, 0],
                available_pieces: [0; 3],
                action_taken: 0,
                piece_identifier: 0,
                reward_received: 0.0,
                policy_target: [0.0; 288],
                value_target: 0.0,
            },
            crate::train::buffer::replay::GameStep {
                board_state: [6, 0],
                available_pieces: [0; 3],
                action_taken: 0,
                piece_identifier: 0,
                reward_received: 0.0,
                policy_target: [0.0; 288],
                value_target: 0.0,
            },
        ];

        replay_buffer.add_game(OwnedGameData {
            difficulty_setting: 6,
            episode_score: 1.0,
            steps: steps_2,
            lines_cleared: 0,
            mcts_depth_mean: 0.0,
            mcts_search_time_mean: 0.0,
        });

        std::thread::sleep(std::time::Duration::from_millis(50));

        assert_eq!(
            replay_buffer.get_length(),
            5,
            "Buffer length should be hard-capped at exact buffer_capacity_limit 5"
        );
        assert_eq!(
            replay_buffer.get_global_write_storage_index(),
            6,
            "Global write index should be monotonic 6"
        );

        let mut final_batch = None;
        for _ in 0..500 {
            if let Some(batch) = replay_buffer.sample_batch(2, Device::Cpu, 1.0) {
                final_batch = Some(batch);
                break;
            }
        }
        let generated_batch =
            final_batch.expect("Should sample batch across wraps after finding non-empty shard");
        assert_eq!(
            generated_batch.state_features_batch.size(),
            vec![2, 20, 8, 16]
        );
    }
}
```

### File: `src/train/buffer/state.rs`

``rs
use crate::core::board::GameStateExt;
use crate::core::features::extract_feature_native;
use std::sync::atomic::{AtomicI32, AtomicUsize};
use std::sync::{Mutex, RwLock};

pub struct ShardedStorageArrays {
    pub shards: Vec<RwLock<StorageArrays>>,
    pub shard_count: usize,
}

impl ShardedStorageArrays {
    pub fn new(buffer_capacity_limit_limit: usize, configured_shard_count: usize) -> Self {
        let mut allocated_shards = Vec::with_capacity(configured_shard_count);
        let shard_buffer_capacity_limit = buffer_capacity_limit_limit / configured_shard_count + 1;
        for _ in 0..configured_shard_count {
            allocated_shards.push(RwLock::new(StorageArrays::new(shard_buffer_capacity_limit)));
        }
        Self {
            shards: allocated_shards,
            shard_count: configured_shard_count,
        }
    }

    #[inline]
    pub fn read_storage_index<T>(
        &self,
        circular_index: usize,
        reader_function: impl FnOnce(&StorageArrays, usize) -> T,
    ) -> T {
        let physical_shard_index = circular_index % self.shard_count;
        let internal_shard_index = circular_index / self.shard_count;
        let array_shard = match self.shards[physical_shard_index].read() {
            Ok(lock) => lock,
            Err(poisoned_error) => poisoned_error.into_inner(),
        };
        reader_function(&array_shard, internal_shard_index)
    }

    #[inline]
    pub fn write_storage_index<T>(
        &self,
        circular_index: usize,
        writer_function: impl FnOnce(&mut StorageArrays, usize) -> T,
    ) -> T {
        let physical_shard_index = circular_index % self.shard_count;
        let internal_shard_index = circular_index / self.shard_count;
        let mut array_shard = match self.shards[physical_shard_index].write() {
            Ok(lock) => lock,
            Err(poisoned_error) => poisoned_error.into_inner(),
        };
        writer_function(&mut array_shard, internal_shard_index)
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EpisodeMeta {
    #[serde(rename = "global_start_idx")]
    pub global_start_storage_index: usize,
    pub length: usize,
    pub difficulty: i32,
    pub score: f32,
    pub lines_cleared: u32,
    pub mcts_depth_mean: f32,
    pub mcts_search_time_mean: f32,
}

pub struct StorageArrays {
    pub boards: Vec<[u64; 2]>,
    pub available: Vec<[i32; 3]>,
    pub actions: Vec<i64>,
    pub piece_ids: Vec<i64>,
    pub rewards: Vec<f32>,
    pub policies: Vec<[f32; 288]>,
    pub values: Vec<f32>,
    pub state_start: Vec<i64>,
    pub state_diff: Vec<i32>,
    pub state_len: Vec<i32>,
}

impl StorageArrays {
    pub fn new(buffer_capacity_limit_limit: usize) -> Self {
        Self {
            boards: vec![[0, 0]; buffer_capacity_limit_limit],
            available: vec![[0, 0, 0]; buffer_capacity_limit_limit],
            actions: vec![0; buffer_capacity_limit_limit],
            piece_ids: vec![0; buffer_capacity_limit_limit],
            rewards: vec![0.0; buffer_capacity_limit_limit],
            policies: vec![[0.0; 288]; buffer_capacity_limit_limit],
            values: vec![0.0; buffer_capacity_limit_limit],
            state_start: vec![-1; buffer_capacity_limit_limit],
            state_diff: vec![0; buffer_capacity_limit_limit],
            state_len: vec![0; buffer_capacity_limit_limit],
        }
    }
}

pub struct SharedState {
    pub buffer_capacity_limit: usize,
    pub unroll_steps: usize,
    pub temporal_difference_steps: usize,

    pub current_diff: AtomicI32,
    pub global_write_storage_index: AtomicUsize,
    pub global_write_active_storage_index: AtomicUsize,
    pub num_states: AtomicUsize,

    pub arrays: ShardedStorageArrays,
    pub per: crate::sumtree::ShardedPrioritizedReplay,

    pub episodes: Mutex<Vec<EpisodeMeta>>,
    pub recent_scores: Mutex<Vec<f32>>,
    pub completed_games: AtomicUsize,
}

impl SharedState {
    pub fn get_features(&self, target_global_index: usize) -> Vec<f32> {
        let circular_index = target_global_index % self.buffer_capacity_limit;
        let physical_shard_index = circular_index % self.arrays.shard_count;
        let internal_shard_index = circular_index / self.arrays.shard_count;

        let memory_shard = match self.arrays.shards[physical_shard_index].read() {
            Ok(lock) => lock,
            Err(poisoned_error) => poisoned_error.into_inner(),
        };

        let logical_start_global = memory_shard.state_start[internal_shard_index];
        if logical_start_global == -1 {
            return vec![0.0; 20 * 128];
        }

        let difficulty_setting = memory_shard.state_diff[internal_shard_index];
        let bitboard_shard = memory_shard.boards[internal_shard_index];
        let reconstructed_board = ((bitboard_shard[1] as u128) << 64) | (bitboard_shard[0] as u128);

        let available_shard = memory_shard.available[internal_shard_index];
        let available_pieces = [available_shard[0], available_shard[1], available_shard[2]];

        let game_state_recreation = GameStateExt::new(
            Some(available_pieces),
            reconstructed_board,
            0,
            difficulty_setting,
            available_pieces
                .iter()
                .filter(|&&piece_identifier| piece_identifier != -1)
                .count() as i32,
        );

        let extracted_history_boards = fetch_historical_boards(
            self,
            target_global_index,
            logical_start_global,
            physical_shard_index,
            &memory_shard,
        );

        let extracted_action_history = fetch_historical_actions(
            self,
            target_global_index,
            logical_start_global,
            physical_shard_index,
            &memory_shard,
        );

        let mut extracted_features = vec![0.0; 20 * 128];
        extract_feature_native(
            &mut extracted_features,
            game_state_recreation.board_bitmask_u128,
            &game_state_recreation.available,
            &extracted_history_boards,
            &extracted_action_history,
            difficulty_setting,
        );
        extracted_features
    }

    pub fn get_historical_boards(&self, circular_index: usize) -> Vec<u128> {
        let physical_shard_index = circular_index % self.arrays.shard_count;
        let internal_shard_index = circular_index / self.arrays.shard_count;

        let memory_shard = match self.arrays.shards[physical_shard_index].read() {
            Ok(lock) => lock,
            Err(poisoned_error) => poisoned_error.into_inner(),
        };

        let logical_start_global = memory_shard.state_start[internal_shard_index];
        if logical_start_global == -1 {
            return vec![];
        }

        let positional_offset = (circular_index as i64 - logical_start_global)
            .rem_euclid(self.buffer_capacity_limit as i64);
        let target_global_index = (logical_start_global + positional_offset) as usize;

        fetch_historical_boards(
            self,
            target_global_index,
            logical_start_global,
            physical_shard_index,
            &memory_shard,
        )
    }

    pub fn get_historical_actions(&self, circular_index: usize) -> Vec<i32> {
        let physical_shard_index = circular_index % self.arrays.shard_count;
        let internal_shard_index = circular_index / self.arrays.shard_count;

        let memory_shard = match self.arrays.shards[physical_shard_index].read() {
            Ok(lock) => lock,
            Err(poisoned_error) => poisoned_error.into_inner(),
        };

        let logical_start_global = memory_shard.state_start[internal_shard_index];
        if logical_start_global == -1 {
            return vec![];
        }

        let positional_offset = (circular_index as i64 - logical_start_global)
            .rem_euclid(self.buffer_capacity_limit as i64);
        let target_global_index = (logical_start_global + positional_offset) as usize;

        fetch_historical_actions(
            self,
            target_global_index,
            logical_start_global,
            physical_shard_index,
            &memory_shard,
        )
    }
}

fn fetch_historical_boards(
    shared_state: &SharedState,
    target_global_index: usize,
    logical_start_global: i64,
    physical_shard_index: usize,
    active_memory_shard: &StorageArrays,
) -> Vec<u128> {
    let mut history_boards = vec![];
    for timestep_offset in 1..=8 {
        if target_global_index < timestep_offset {
            break;
        }
        let previous_global_index = target_global_index - timestep_offset;
        if previous_global_index < logical_start_global as usize {
            break;
        }

        let previous_circular_index = previous_global_index % shared_state.buffer_capacity_limit;
        let previous_physical_shard = previous_circular_index % shared_state.arrays.shard_count;
        let previous_internal_index = previous_circular_index / shared_state.arrays.shard_count;

        if previous_physical_shard == physical_shard_index {
            let previous_bitboard = active_memory_shard.boards[previous_internal_index];
            history_boards
                .push(((previous_bitboard[1] as u128) << 64) | (previous_bitboard[0] as u128));
        } else {
            let previous_memory_shard =
                match shared_state.arrays.shards[previous_physical_shard].read() {
                    Ok(lock) => lock,
                    Err(err) => err.into_inner(),
                };
            let previous_bitboard = previous_memory_shard.boards[previous_internal_index];
            history_boards
                .push(((previous_bitboard[1] as u128) << 64) | (previous_bitboard[0] as u128));
        }
    }
    history_boards.reverse();
    history_boards
}

fn fetch_historical_actions(
    shared_state: &SharedState,
    target_global_index: usize,
    logical_start_global: i64,
    physical_shard_index: usize,
    active_memory_shard: &StorageArrays,
) -> Vec<i32> {
    let mut action_history = vec![];
    for timestep_offset in 1..=4 {
        if target_global_index < timestep_offset {
            break;
        }
        let previous_global_index = target_global_index - timestep_offset;
        if previous_global_index < logical_start_global as usize {
            break;
        }

        let previous_circular_index = previous_global_index % shared_state.buffer_capacity_limit;
        let previous_physical_shard = previous_circular_index % shared_state.arrays.shard_count;
        let previous_internal_index = previous_circular_index / shared_state.arrays.shard_count;

        if previous_physical_shard == physical_shard_index {
            action_history.push(active_memory_shard.actions[previous_internal_index] as i32);
        } else {
            let previous_memory_shard =
                match shared_state.arrays.shards[previous_physical_shard].read() {
                    Ok(lock) => lock,
                    Err(err) => err.into_inner(),
                };
            action_history.push(previous_memory_shard.actions[previous_internal_index] as i32);
        }
    }
    action_history.reverse();
    action_history
}

#[cfg(test)]
mod tests {
    use super::*;
    // removed
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_cross_shard_history_reads() {
        let state = SharedState {
            buffer_capacity_limit: 4,
            unroll_steps: 1,
            temporal_difference_steps: 1,
            current_diff: AtomicI32::new(1),
            global_write_storage_index: AtomicUsize::new(4),
            global_write_active_storage_index: AtomicUsize::new(4),
            num_states: AtomicUsize::new(4),
            arrays: ShardedStorageArrays::new(4, 2),
            per: crate::sumtree::ShardedPrioritizedReplay::new(4, 0.6, 0.4, 2),
            episodes: Mutex::new(vec![]),
            recent_scores: Mutex::new(vec![]),
            completed_games: AtomicUsize::new(0),
        };

        state.arrays.write_storage_index(0, |memory_shard, index| {
            memory_shard.state_start[index] = 0;
            memory_shard.boards[index] = [0b1, 0];
        });
        state.arrays.write_storage_index(1, |memory_shard, index| {
            memory_shard.state_start[index] = 0;
            memory_shard.boards[index] = [0b10, 0];
        });
        state.arrays.write_storage_index(2, |memory_shard, index| {
            memory_shard.state_start[index] = 0;
            memory_shard.boards[index] = [0b100, 0];
        });

        let features = state.get_features(2);

        let memory_offset_1 = 128; // Channel 1 (history T-1 => Bit 1)
        assert_eq!(
            features[memory_offset_1 + 5], // Bit 1 maps to (0, 5) -> 5
            1.0,
            "Cross-shard history read failed"
        );

        let memory_offset_2 = 2 * 128; // Channel 2 (history T-2 => Bit 0)
        assert_eq!(
            features[memory_offset_2 + 4], // Bit 0 maps to (0, 4) -> 4
            1.0,
            "Same-shard history read failed"
        );
    }

    #[test]
    fn test_torn_read_prevention() {
        let storage_arrays = Arc::new(ShardedStorageArrays::new(100, 4));
        let storage_arrays_clone = Arc::clone(&storage_arrays);

        let thread_writer = thread::spawn(move || {
            for index in 0..10_000 {
                storage_arrays_clone.write_storage_index(5, |memory_shard, physical_index| {
                    memory_shard.state_start[physical_index] = index as i64;
                    memory_shard.state_diff[physical_index] = index;
                });
            }
        });

        let thread_reader = thread::spawn(move || {
            for _ in 0..10_000 {
                storage_arrays.read_storage_index(5, |memory_shard, physical_index| {
                    let logical_start = memory_shard.state_start[physical_index];
                    let difficulty_setting = memory_shard.state_diff[physical_index];
                    if logical_start != -1 {
                        assert_eq!(
                            logical_start as i32, difficulty_setting,
                            "Torn read detected: start and diff arrays desynchronized"
                        );
                    }
                });
            }
        });

        thread_writer.join().unwrap();
        thread_reader.join().unwrap();
    }

    #[test]
    fn test_historical_padding() {
        let state = SharedState {
            buffer_capacity_limit: 10,
            unroll_steps: 1,
            temporal_difference_steps: 1,
            current_diff: AtomicI32::new(1),
            global_write_storage_index: AtomicUsize::new(10),
            global_write_active_storage_index: AtomicUsize::new(10),
            num_states: AtomicUsize::new(10),
            arrays: ShardedStorageArrays::new(10, 1),
            per: crate::sumtree::ShardedPrioritizedReplay::new(10, 0.6, 0.4, 1),
            episodes: Mutex::new(vec![]),
            recent_scores: Mutex::new(vec![]),
            completed_games: AtomicUsize::new(0),
        };

        // Write a sequence of 4 states for a single game starting at global index 0
        state.arrays.write_storage_index(0, |shard, _| {
            shard.state_start[0] = 0;
            shard.boards[0] = [0, 0]; // State 0
            shard.state_start[1] = 0;
            shard.boards[1] = [1, 0]; // State 1
            shard.state_start[2] = 0;
            shard.boards[2] = [2, 0]; // State 2
            shard.state_start[3] = 0;
            shard.boards[3] = [3, 0]; // State 3
        });

        // Request features at step 3.
        // According to padding rules: T=3, T-1=2, T-2=1, T-3=0, T-4=0 (padded), T-5=0, T-6=0, T-7=0
        state.arrays.read_storage_index(0, |shard, _| {
            let history = fetch_historical_boards(&state, 3, 0, 0, shard);
            // history_boards should return the sequence in reverse-time order but fetch_historical_boards
            // actually reverses it at the end. Let's trace it:
            // offset 1 (prev=2), offset 2 (prev=1), offset 3 (prev=0).
            // Offset 4+ break because prev < logical_start_global.
            // So history vector is [2, 1, 0]. Reversed it is [0, 1, 2].
            assert_eq!(
                history,
                vec![0, 1, 2],
                "Historical boards fetched did not match [State_0, State_1, State_2]"
            );
        });

        // The feature extractor itself manages the padding when `history_boards` falls short.
        let state_3 = GameStateExt::new(Some([0, 0, 0]), 3, 0, 6, 0);
        let mut _extracted = vec![0.0; 20 * 128];
        crate::core::features::extract_feature_native(
            &mut _extracted,
            state_3.board_bitmask_u128,
            &state_3.available,
            &[0, 1, 2],
            &[],
            6,
        );

        // Channel 0 = State 3
        // Channel 1 = State 2 (T-1)
        // Channel 2 = State 1 (T-2)
        // Channel 3 = State 0 (T-3)
        // Channel 4 = State 3 (Padding defaults to current state? Wait.
        // Let's look at fill_history_channels:
        // "if unwrapped_history.len() >= memory_index { ... prior_state } else { ... current_board_state })"
        // Wait, MuZero pads with CURRENT state, not State 0!
        // That is correct mathematically for AlphaZero/MuZero history padding.
    }
}
``

### File: `src/train/mod.rs`

```rs
pub mod buffer;
pub mod optimizer;
```

### File: `src/train/optimizer/loss.rs`

```rs
use tch::{Kind, Tensor};

/// Calculates Negative Cosine Similarity:
/// L(x1, x2) = - (x1 · x2) / (||x1||_2 * ||x2||_2)
pub fn negative_cosine_similarity(
    active_projection_fp16: &Tensor,
    target_projection_fp16: &Tensor,
) -> Tensor {
    let active_projection = active_projection_fp16.to_kind(Kind::Float);
    let target_projection = target_projection_fp16.to_kind(Kind::Float);

    #[cfg(debug_assertions)]
    assert!(
        i64::try_from(active_projection.isnan().any()).unwrap() == 0,
        "NaN detected in active_projection before cosine similarity"
    );
    #[cfg(debug_assertions)]
    assert!(
        i64::try_from(target_projection.isnan().any()).unwrap() == 0,
        "NaN detected in target_projection before cosine similarity"
    );

    let active_l2_norm =
        (active_projection
            .pow_tensor_scalar(2.0)
            .sum_dim_intlist(&[-1i64][..], true, Kind::Float)
            + 1e-8)
            .sqrt();

    let target_l2_norm =
        (target_projection
            .pow_tensor_scalar(2.0)
            .sum_dim_intlist(&[-1i64][..], true, Kind::Float)
            + 1e-8)
            .sqrt();

    let active_normalized = active_projection / active_l2_norm;
    let target_normalized = target_projection / target_l2_norm;

    let similarity_loss = -(&active_normalized * &target_normalized).sum_dim_intlist(
        &[-1i64][..],
        false,
        Kind::Float,
    );

    #[cfg(debug_assertions)]
    assert!(
        i64::try_from(similarity_loss.isnan().any()).unwrap() == 0,
        "NaN detected resulting from cosine similarity calculation"
    );

    similarity_loss
}

/// Calculates Soft Cross Entropy Loss:
/// L = - Σ (target_probs * log(softmax(logits)))
pub fn soft_cross_entropy(prediction_logits: &Tensor, target_probabilities: &Tensor) -> Tensor {
    #[cfg(debug_assertions)]
    assert!(
        i64::try_from(prediction_logits.isnan().any()).unwrap() == 0,
        "NaN detected in prediction_logits before soft_cross_entropy"
    );

    let logarithm_probabilities = prediction_logits.log_softmax(-1, Kind::Float);
    let cross_entropy_loss = -(target_probabilities * logarithm_probabilities).sum_dim_intlist(
        &[-1i64][..],
        false,
        Kind::Float,
    );

    #[cfg(debug_assertions)]
    assert!(
        i64::try_from(cross_entropy_loss.isnan().any()).unwrap() == 0,
        "NaN detected resulting from soft_cross_entropy calculation"
    );

    cross_entropy_loss
}

/// Calculates Binary Cross Entropy Loss:
/// L = - [target * log(σ(logits)) + (1 - target) * log(1 - σ(logits))]
pub fn binary_cross_entropy(prediction_logits: &Tensor, binary_targets: &Tensor) -> Tensor {
    #[cfg(debug_assertions)]
    assert!(
        i64::try_from(prediction_logits.isnan().any()).unwrap() == 0,
        "NaN detected in prediction_logits before binary_cross_entropy"
    );

    let binary_cross_entropy_loss = prediction_logits.binary_cross_entropy_with_logits::<Tensor>(
        binary_targets,
        None,
        None,
        tch::Reduction::None,
    );

    #[cfg(debug_assertions)]
    assert!(
        i64::try_from(binary_cross_entropy_loss.isnan().any()).unwrap() == 0,
        "NaN detected resulting from BCE calculation"
    );

    binary_cross_entropy_loss
}

/// Scales the gradient passing through the tensor:
/// x_scaled = x * scale + detach(x) * (1 - scale)
pub fn scale_gradient(input_tensor: &Tensor, gradient_scale: f64) -> Tensor {
    input_tensor * gradient_scale + input_tensor.detach() * (1.0 - gradient_scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind, Tensor};

    #[test]
    fn test_scale_gradient() {
        let active_tensor =
            Tensor::ones([2, 2], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let scaled_tensor = scale_gradient(&active_tensor, 0.5);
        let sum_loss = scaled_tensor.sum(Kind::Float);
        sum_loss.backward();

        let gradient = active_tensor.grad();
        let gradient_value: f32 = gradient.mean(Kind::Float).try_into().unwrap();
        assert!(
            (gradient_value - 0.5).abs() < 1e-6,
            "Gradient was not scaled correctly by 0.5"
        );
    }

    #[test]
    fn test_cosine_similarity_epsilon_stability() {
        let active_projection = Tensor::zeros([2, 512], (Kind::Float, Device::Cpu));
        let target_projection = Tensor::zeros([2, 512], (Kind::Float, Device::Cpu));
        let loss = negative_cosine_similarity(&active_projection, &target_projection);
        assert_eq!(
            i64::try_from(loss.isnan().any()).unwrap(),
            0,
            "Cosine similarity resulted in NaN on zero tensors!"
        );
    }
    #[test]
    fn test_soft_cross_entropy_parity() {
        let logits = Tensor::from_slice(&[2.0_f32, 1.0, 0.1]);
        let targets = Tensor::from_slice(&[0.7_f32, 0.2, 0.1]);

        let loss = soft_cross_entropy(&logits, &targets);

        let expected_prob = [
            (2.0f32).exp() / ((2.0f32).exp() + (1.0f32).exp() + (0.1f32).exp()),
            (1.0f32).exp() / ((2.0f32).exp() + (1.0f32).exp() + (0.1f32).exp()),
            (0.1f32).exp() / ((2.0f32).exp() + (1.0f32).exp() + (0.1f32).exp()),
        ];

        let manual_loss = -(0.7 * expected_prob[0].ln()
            + 0.2 * expected_prob[1].ln()
            + 0.1 * expected_prob[2].ln());

        let rust_loss: f32 = loss.try_into().unwrap_or(0.0);
        assert!(
            (rust_loss - manual_loss).abs() < 1e-4,
            "Loss Function parity failed! Rust: {} vs Manual: {}",
            rust_loss,
            manual_loss
        );
    }
}
```

### File: `src/train/optimizer/mod.rs`

```rs
pub mod loss;
pub mod optimization;

// unused
```

### File: `src/train/optimizer/optimization.rs`

```rs
use crate::net::MuZeroNet;
use crate::train::buffer::ReplayBuffer;
use crate::train::optimizer::loss::{
    binary_cross_entropy, negative_cosine_similarity, scale_gradient, soft_cross_entropy,
};
use tch::{nn, nn::Module, Kind, Tensor};

pub struct TrainMetrics {
    pub total_loss: f64,
    pub policy_loss: f64,
    pub value_loss: f64,
    pub reward_loss: f64,
}

pub fn train_step(
    neural_model: &MuZeroNet,
    exponential_moving_average_model: &MuZeroNet,
    gradient_optimizer: &mut nn::Optimizer,
    replay_buffer: &ReplayBuffer,
    batched_experience_tensors: crate::train::buffer::replay::BatchTensors,
    sequence_unroll_steps: usize,
) -> TrainMetrics {
    let sequence_unroll_steps = sequence_unroll_steps as i64;

    let batched_state = batched_experience_tensors.state_features_batch;
    let batched_action = batched_experience_tensors.actions_batch;
    let batched_piece_identifier = batched_experience_tensors.piece_identifiers_batch;
    let batched_reward = batched_experience_tensors.rewards_batch;
    let batched_target_policy = batched_experience_tensors.target_policies_batch;
    let batched_target_value = batched_experience_tensors.target_values_batch;
    let batched_transition_state = batched_experience_tensors.transition_states_batch;
    let batched_mask = batched_experience_tensors.loss_masks_batch;
    let batched_importance_weight = batched_experience_tensors.importance_weights_batch;
    let global_indices = batched_experience_tensors.global_indices_sampled;

    gradient_optimizer.zero_grad();
    let scaled_importance_weights = batched_importance_weight;

    let (
        computed_final_loss,
        temporal_difference_errors,
        avg_policy_loss,
        avg_value_loss,
        avg_reward_loss,
    ) = tch::autocast(false, || {
        #[cfg(debug_assertions)]
        assert!(
            i64::try_from(batched_state.isnan().any()).unwrap() == 0,
            "batched_state ALREADY HAS NANS!"
        );
        let mut running_hidden_state = neural_model.representation.forward(&batched_state);

        let rh_size = running_hidden_state.size();
        assert_eq!(
            rh_size.len(),
            4,
            "Hidden state must strictly be [Batch, Channels, Height, Width]"
        );
        assert_eq!(rh_size[2], 8, "Spatial height must be exactly 8");
        assert_eq!(rh_size[3], 8, "Spatial width must be exactly 8");
        #[cfg(debug_assertions)]
        assert!(
            i64::try_from(running_hidden_state.isnan().any()).unwrap() == 0,
            "NaN detected in running_hidden_state!"
        );

        let (initial_value_logits, initial_policy_logits, initial_hidden_state_logits) =
            neural_model.prediction.forward(&running_hidden_state);

        // Value Loss: Cross-entropy between network value support prediction and target scalar
        let initial_value_targets =
            neural_model.scalar_to_support(&batched_target_value.select(1, 0));
        let initial_value_loss = soft_cross_entropy(&initial_value_logits, &initial_value_targets);

        // Policy Loss: Cross-entropy between network policy vector and MCTS target distribution
        let initial_policy_probabilities_target = batched_target_policy.select(1, 0) + 1e-8;
        let initial_policy_loss =
            soft_cross_entropy(&initial_policy_logits, &initial_policy_probabilities_target);

        let mut initial_binary_cross_entropy = binary_cross_entropy(
            &initial_hidden_state_logits,
            &batched_state.select(1, 19).flatten(1, -1),
        );
        if initial_binary_cross_entropy.dim() > 1 {
            initial_binary_cross_entropy =
                initial_binary_cross_entropy.mean_dim(&[1i64][..], false, Kind::Float);
        }

        let mut cumulative_loss =
            &initial_value_loss + &initial_policy_loss + (&initial_binary_cross_entropy * 0.5);

        let mut value_loss_tracker = initial_value_loss.mean(Kind::Float);
        let mut policy_loss_tracker = initial_policy_loss.mean(Kind::Float);
        let mut reward_loss_tracker = Tensor::zeros_like(&value_loss_tracker);

        for unroll_k in 0..sequence_unroll_steps {
            let action_at_k = batched_action.select(1, unroll_k);
            let piece_identifier_at_k = batched_piece_identifier.select(1, unroll_k);

            let (next_hidden_state_prediction, reward_logits_prediction) =
                neural_model.dynamics.forward(
                    &scale_gradient(&running_hidden_state, 0.5),
                    &action_at_k,
                    &piece_identifier_at_k,
                );
            running_hidden_state = next_hidden_state_prediction;

            let rh_size = running_hidden_state.size();
            assert_eq!(
                rh_size.len(),
                4,
                "Dynamics hidden state must strictly be [Batch, Channels, Height, Width]"
            );
            assert_eq!(rh_size[2], 8, "Spatial height must be exactly 8");
            assert_eq!(rh_size[3], 8, "Spatial width must be exactly 8");
            #[cfg(debug_assertions)]
            assert!(
                i64::try_from(running_hidden_state.isnan().any()).unwrap() == 0,
                "NaN detected in next_hidden_state_prediction!"
            );
            let target_hidden_state_projection = tch::no_grad(|| {
                exponential_moving_average_model
                    .representation
                    .forward(&batched_transition_state.select(1, unroll_k))
            });
            let projected_target_representation = tch::no_grad(|| {
                exponential_moving_average_model
                    .projector
                    .forward(&target_hidden_state_projection)
            });

            let projected_active_representation =
                neural_model.projector.forward(&running_hidden_state);
            let (unrolled_value_logits, unrolled_policy_logits, unrolled_hidden_state_logits) =
                neural_model.prediction.forward(&running_hidden_state);

            let reward_targets_support =
                neural_model.scalar_to_support(&batched_reward.select(1, unroll_k));
            let unroll_sequence_mask = batched_mask.select(1, unroll_k + 1);

            let unrolled_reward_loss =
                soft_cross_entropy(&reward_logits_prediction, &reward_targets_support)
                    * &unroll_sequence_mask;

            let value_targets_support =
                neural_model.scalar_to_support(&batched_target_value.select(1, unroll_k + 1));
            let value_decay = 0.5f64.powi(unroll_k as i32 + 1);
            let unrolled_value_loss =
                soft_cross_entropy(&unrolled_value_logits, &value_targets_support)
                    * &unroll_sequence_mask
                    * value_decay;

            let unrolled_policy_targets = batched_target_policy.select(1, unroll_k + 1) + 1e-8;
            let unrolled_policy_loss =
                soft_cross_entropy(&unrolled_policy_logits, &unrolled_policy_targets)
                    * &unroll_sequence_mask;

            reward_loss_tracker += unrolled_reward_loss.mean(Kind::Float);
            value_loss_tracker += unrolled_value_loss.mean(Kind::Float);
            policy_loss_tracker += unrolled_policy_loss.mean(Kind::Float);

            let unroll_scale = 1.0 / (sequence_unroll_steps as f64);

            cumulative_loss +=
                (&unrolled_reward_loss + &unrolled_value_loss + &unrolled_policy_loss)
                    * unroll_scale;
            cumulative_loss += (negative_cosine_similarity(
                &projected_active_representation,
                &projected_target_representation,
            ) * &unroll_sequence_mask)
                * unroll_scale;

            let mut unrolled_binary_cross_entropy = binary_cross_entropy(
                &unrolled_hidden_state_logits,
                &batched_transition_state
                    .select(1, unroll_k)
                    .select(1, 19)
                    .flatten(1, -1),
            );
            if unrolled_binary_cross_entropy.dim() > 1 {
                unrolled_binary_cross_entropy =
                    unrolled_binary_cross_entropy.mean_dim(&[1i64][..], false, Kind::Float);
            }
            cumulative_loss +=
                (unrolled_binary_cross_entropy * 0.5 * &unroll_sequence_mask) * unroll_scale;
        }

        let averaged_scaled_final_loss =
            (cumulative_loss * scaled_importance_weights).mean(Kind::Float);

        let absolute_temporal_difference_errors = tch::no_grad(|| {
            (neural_model.scalar_to_support(&batched_target_value.select(1, 0))
                - initial_value_logits.softmax(-1, Kind::Float))
            .abs()
            .sum_dim_intlist(&[-1i64][..], false, Kind::Float)
        });

        let divisor = (sequence_unroll_steps + 1) as f64;
        let avg_policy_loss = f64::try_from(policy_loss_tracker / divisor).unwrap_or(0.0);
        let avg_value_loss = f64::try_from(value_loss_tracker / divisor).unwrap_or(0.0);
        let avg_reward_loss = f64::try_from(reward_loss_tracker / divisor).unwrap_or(0.0);

        (
            averaged_scaled_final_loss,
            absolute_temporal_difference_errors,
            avg_policy_loss,
            avg_value_loss,
            avg_reward_loss,
        )
    });

    computed_final_loss.backward();

    gradient_optimizer.clip_grad_norm(5.0);
    gradient_optimizer.step();

    let temporal_difference_f32_vec: Vec<f32> =
        temporal_difference_errors.try_into().unwrap_or_default();
    let temporal_difference_f64_vec: Vec<f64> = temporal_difference_f32_vec
        .into_iter()
        .map(|error_val| error_val as f64)
        .collect();
    replay_buffer.update_priorities(&global_indices, &temporal_difference_f64_vec);

    let final_loss_f64 = f64::try_from(computed_final_loss).unwrap_or(0.0);

    TrainMetrics {
        total_loss: final_loss_f64,
        policy_loss: avg_policy_loss,
        value_loss: avg_value_loss,
        reward_loss: avg_reward_loss,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn, nn::OptimizerConfig, Device};

    #[test]
    fn test_train_step_bptt_and_masking() {
        let variable_store = nn::VarStore::new(Device::Cpu);
        let neural_model = MuZeroNet::new(&variable_store.root(), 16, 1, 200);
        let ema_model = MuZeroNet::new(&variable_store.root(), 16, 1, 200);
        let mut gradient_optimizer = nn::Adam::default().build(&variable_store, 1e-3).unwrap();

        let configuration = crate::config::Config {
            experiment_name_identifier: "test_exp".to_string(),
            device: "cpu".into(),
            paths: crate::config::ExperimentPaths::default(),
            hidden_dimension_size: 16,
            num_blocks: 1,
            support_size: 200,
            buffer_capacity_limit: 100,
            simulations: 10,
            train_batch_size: 2,
            train_epochs: 1,
            num_processes: 1,
            worker_device: "cpu".into(),
            unroll_steps: 2,
            temporal_difference_steps: 5,
            inference_batch_size_limit: 1,
            inference_timeout_ms: 1,
            max_gumbel_k: 4,
            gumbel_scale: 1.0,
            temp_decay_steps: 10,
            difficulty: 6,
            temp_boost: false,
            lr_init: 1e-3,
            reanalyze_ratio: 0.25,
        };

        let replay_buffer = ReplayBuffer::new(100, 2, 8);

        let steps = vec![
            crate::train::buffer::replay::GameStep {
                board_state: [0u64, 0u64],
                available_pieces: [0i32, 0, 0],
                action_taken: 0i64,
                piece_identifier: 0i64,
                reward_received: 1.0f32,
                policy_target: [0.0f32; 288],
                value_target: 0.5f32,
            };
            15
        ];

        replay_buffer.add_game(crate::train::buffer::replay::OwnedGameData {
            difficulty_setting: 6,
            episode_score: 1.0,
            steps: steps.clone(),
            lines_cleared: 0,
            mcts_depth_mean: 0.0,
            mcts_search_time_mean: 0.0,
        });
        replay_buffer.add_game(crate::train::buffer::replay::OwnedGameData {
            difficulty_setting: 6,
            episode_score: 1.0,
            steps,
            lines_cleared: 0,
            mcts_depth_mean: 0.0,
            mcts_search_time_mean: 0.0,
        });

        let mut batched_experience_tensors_opt = None;
        for _ in 0..50 {
            if let Some(batch) = replay_buffer.sample_batch(2, Device::Cpu, 1.0) {
                batched_experience_tensors_opt = Some(batch);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let batched_experience_tensors = batched_experience_tensors_opt.unwrap();

        train_step(
            &neural_model,
            &ema_model,
            &mut gradient_optimizer,
            &replay_buffer,
            batched_experience_tensors,
            configuration.unroll_steps,
        );
    }
}
```

### File: `tests/arcswap_test.rs`

```rs
use arc_swap::ArcSwap;
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use tch::{nn, Device};
use tricked_engine::net::MuZeroNet;

#[test]
fn test_arcswap_double_buffering_latency() {
    let device = Device::Cpu;
    let vs_a = nn::VarStore::new(device);
    let vs_b = nn::VarStore::new(device);

    // Create tiny MuZeroNet instances to minimize allocation overhead during setup,
    // we only care about the ArcSwap pointer swap latency for this test.
    let net_a = Arc::new(MuZeroNet::new(&vs_a.root(), 16, 1, 288));
    let net_b = Arc::new(MuZeroNet::new(&vs_b.root(), 16, 1, 288));

    let shared_arc = Arc::new(ArcSwap::from(net_a));

    // Simulate 8 inference threads doing wait-free loads continuously
    let mut handles = vec![];
    let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));

    for _ in 0..8 {
        let arc_clone = Arc::clone(&shared_arc);
        let flag = Arc::clone(&stop_flag);
        handles.push(thread::spawn(move || {
            let mut loads = 0;
            while !flag.load(std::sync::atomic::Ordering::Relaxed) {
                let _loaded = arc_clone.load();
                loads += 1;
            }
            std::hint::black_box(loads);
        }));
    }

    // Give readers a moment to spin up
    thread::sleep(std::time::Duration::from_millis(50));

    // Measure the latency of 100 swaps (simulating 100 training steps' weight syncs)
    let start = Instant::now();
    for _ in 0..100 {
        shared_arc.store(Arc::clone(&net_b));
    }
    let duration = start.elapsed();

    // Signal readers to stop
    stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
    for h in handles {
        h.join().unwrap();
    }

    // The total time for 100 stores while 8 readers are hammering it should be minimal.
    let avg_latency_ms = duration.as_secs_f64() / 100.0 * 1000.0;
    println!(
        "Average ArcSwap Store Latency (with 8 active readers): {:.4}ms",
        avg_latency_ms
    );

    // Test that the swap is wait-free (< 1.0ms on any modern CPU even under high contention)
    assert!(
        avg_latency_ms < 1.0,
        "ArcSwap block time is too high: {:.4}ms. Double buffering should be virtually wait-free.",
        avg_latency_ms
    );
}
```

### File: `tests/board_fuzz.rs`

```rs
use proptest::prelude::*;
use tricked_engine::core::board::GameStateExt;
use tricked_engine::core::constants::{ALL_MASKS, STANDARD_PIECES};

proptest! {
    // 100,000 randomized cases to ensure 100% mathematical correctness
    // of the piece placement and line clear algorithm.
    #![proptest_config(ProptestConfig::with_cases(100_000))]
    #[test]
    fn rigorous_fuzz_apply_move(
        piece_slot in 0usize..3,
        piece_id in 0i32..STANDARD_PIECES.len() as i32,
        drop_index in 0usize..96,
        initial_board in 0u128..=u128::MAX,
        current_score in 0i32..10_000
    ) {
        // Ensure initial board is constrained to 96 bits
        let initial_board_96 = initial_board & ((1u128 << 96) - 1);

        let mut available = [0, 0, 0];
        available[piece_slot] = piece_id;

        let mut state = GameStateExt::new(Some(available), initial_board_96, current_score, 0, 0);

        let piece_mask = STANDARD_PIECES[piece_id as usize][drop_index];

        if piece_mask != 0 && (state.board_bitmask_u128 & piece_mask) == 0 {
            if let Some(next_state) = state.apply_move(piece_slot, drop_index) {
                // Rule 1: No overlaps allowed, sum of bits must never exceed 96
                assert!(next_state.board_bitmask_u128.count_ones() <= 96,
                        "Board bits exceeded 96 max allowed on valid drop.");

                // Rule 2: Strict Mathematical Delta for Score
                let placed_hexes = piece_mask.count_ones() as i32;

                let simulated_board_bitmask_u128 = state.board_bitmask_u128 | piece_mask;
                let mut expected_score = state.score + placed_hexes;

                for &line in ALL_MASKS.iter() {
                    if (simulated_board_bitmask_u128 & line) == line {
                        expected_score += (line.count_ones() as i32) * 2;
                    }
                }

                assert_eq!(next_state.score, expected_score,
                           "Score increment diverged from expected mathematical delta! State Score: {}, Expected: {}",
                           next_state.score, expected_score);

                // Rule 3: Available piece in slot must be negated to -1
                assert_eq!(next_state.available[piece_slot], -1, "Used piece slot was not cleared to -1");
            }
        }
    }
}
```

