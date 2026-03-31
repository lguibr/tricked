```
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

