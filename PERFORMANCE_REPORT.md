# 🏎️ Tricked AI Performance Report
This report outlines the hardware footprint and operational latency of the Tricked Engine across 4 benchmark quadrants.

## Small Model, Shallow Search
**Total Execution Time:** `37.62s`
- **RAM Usage:** `11899.88 MB`
- **GPU VRAM:** `1175.00 MB`
- **Nodes / Sec (NPS):** `0.94`

### Top Execution Bottlenecks
| Function                                  | Calls   | Avg       | P95       | Total     | % Total  |
| search::process_evaluation_responses      | 7176    | 237.20 ms | 517.73 ms | 1702.16 s | 4525.02% |
| search::expand_and_evaluate_candidates    | 7113    | 234.71 ms | 509.08 ms | 1669.52 s | 4438.23% |
| search::mcts_search                       | 2294    | 724.54 ms | 1.26 s    | 1662.08 s | 4418.47% |
| search::execute_sequential_halving        | 2261    | 734.92 ms | 1.26 s    | 1661.65 s | 4417.33% |
| tricked_engine::main                      | 1       | 37.63 s   | 37.65 s   | 37.62 s   | 100.00%  |
| tricked_engine::run_training              | 1       | 37.63 s   | 37.65 s   | 37.62 s   | 100.00%  |
| worker::process_recurrent_inference       | 861     | 35.78 ms  | 85.79 ms  | 30.80 s   | 81.89%   |
| optimization::train_step                  | 5       | 5.99 s    | 7.10 s    | 29.97 s   | 79.67%   |
| worker::process_initial_inference         | 358     | 11.60 ms  | 51.31 ms  | 4.15 s    | 11.04%   |
| worker::compute_max_depth                 | 3906279 | 482 ns    | 26 ns     | 1.88 s    | 5.01%    |
| search::compute_final_action_distribution | 2261    | 88.48 µs  | 193.41 µs | 200.06 ms | 0.53%    |
| search::traverse_tree_to_leaf             | 33887   | 2.46 µs   | 8.45 µs   | 83.43 ms  | 0.22%    |
| search::inject_gumbel_noise               | 2324    | 16.29 µs  | 46.17 µs  | 37.87 ms  | 0.10%    |
| search::prune_candidates                  | 7113    | 4.19 µs   | 8.20 µs   | 29.81 ms  | 0.08%    |
| search::normalize_policy_distributions    | 2358    | 8.80 µs   | 13.61 µs  | 20.74 ms  | 0.06%    |

### Largest Memory Allocations
| Function                                  | Calls | Avg      | P95     | Total    | % Total   |
| search::mcts_search                       | 2294  | 1.6 MB   | 1.6 MB  | 3.6 GB   | 14930.35% |
| search::process_evaluation_responses      | 7176  | 421.1 KB | 1.6 MB  | 2.9 GB   | 11872.78% |
| search::expand_and_evaluate_candidates    | 7113  | 417.2 KB | 1.6 MB  | 2.8 GB   | 11660.08% |
| search::execute_sequential_halving        | 2261  | 1.3 MB   | 1.6 MB  | 2.8 GB   | 11489.77% |
| tricked_engine::main                      | 1     | 24.9 MB  | 24.9 MB | 24.9 MB  | 100.00%   |
| tricked_engine::run_training              | 1     | 24.8 MB  | 24.8 MB | 24.8 MB  | 99.86%    |
| search::normalize_policy_distributions    | 2358  | 1.6 KB   | 2.1 KB  | 3.7 MB   | 14.86%    |
| search::inject_gumbel_noise               | 2324  | 1.4 KB   | 1.5 KB  | 3.2 MB   | 12.85%    |
| search::compute_final_action_distribution | 2261  | 610 B    | 1.1 KB  | 1.3 MB   | 5.30%     |
| search::prune_candidates                  | 7113  | 137 B    | 472 B   | 954.1 KB | 3.75%     |
| worker::calculate_policy_targets          | 498   | 1.1 KB   | 1.1 KB  | 560.2 KB | 2.20%     |
| worker::process_recurrent_inference       | 861   | 564 B    | 564 B   | 474.2 KB | 1.86%     |
| worker::process_initial_inference         | 358   | 813 B    | 530 B   | 284.4 KB | 1.12%     |
| optimization::train_step                  | 5     | 36.2 KB  | 36.2 KB | 180.9 KB | 0.71%     |
| board::refill_tray                        | 172   | 207 B    | 34.8 KB | 34.8 KB  | 0.14%     |

---
## Small Model, Deep Search
**Total Execution Time:** `60.46s`
- **RAM Usage:** `12654.82 MB`
- **GPU VRAM:** `1208.00 MB`
- **Nodes / Sec (NPS):** `0.59`

### Top Execution Bottlenecks
| Function                                  | Calls    | Avg       | P95       | Total     | % Total  |
| search::process_evaluation_responses      | 2079     | 986.46 ms | 3.69 s    | 2050.84 s | 3392.25% |
| search::expand_and_evaluate_candidates    | 2030     | 960.93 ms | 3.69 s    | 1950.70 s | 3226.61% |
| search::mcts_search                       | 579      | 3.35 s    | 10.28 s   | 1940.77 s | 3210.18% |
| search::execute_sequential_halving        | 574      | 3.38 s    | 10.28 s   | 1940.21 s | 3209.25% |
| tricked_engine::main                      | 1        | 60.45 s   | 60.47 s   | 60.46 s   | 100.00%  |
| tricked_engine::run_training              | 1        | 60.45 s   | 60.47 s   | 60.46 s   | 100.00%  |
| worker::inference_loop                    | 1        | 60.38 s   | 60.40 s   | 60.39 s   | 99.89%   |
| worker::game_loop                         | 1        | 60.38 s   | 60.40 s   | 60.37 s   | 99.85%   |
| worker::process_recurrent_inference       | 1221     | 47.25 ms  | 96.01 ms  | 57.69 s   | 95.43%   |
| optimization::train_step                  | 5        | 7.23 s    | 7.71 s    | 36.13 s   | 59.76%   |
| worker::compute_max_depth                 | 15172074 | 697 ns    | 30 ns     | 10.58 s   | 17.49%   |
| worker::process_initial_inference         | 329      | 5.25 ms   | 10.83 ms  | 1.73 s    | 2.85%    |
| search::traverse_tree_to_leaf             | 75253    | 5.33 µs   | 12.87 µs  | 401.37 ms | 0.66%    |
| search::compute_final_action_distribution | 560      | 168.42 µs | 239.87 µs | 94.32 ms  | 0.16%    |
| search::inject_gumbel_noise               | 626      | 26.18 µs  | 68.61 µs  | 16.39 ms  | 0.03%    |

### Largest Memory Allocations
| Function                                  | Calls | Avg      | P95      | Total    | % Total   |
| search::mcts_search                       | 579   | 5.9 MB   | 6.5 MB   | 3.3 GB   | 13685.63% |
| search::process_evaluation_responses      | 2079  | 563.4 KB | 3.2 MB   | 1.1 GB   | 4602.91%  |
| search::expand_and_evaluate_candidates    | 2030  | 538.3 KB | 3.2 MB   | 1.0 GB   | 4293.47%  |
| search::execute_sequential_halving        | 574   | 1.9 MB   | 4.8 MB   | 1.0 GB   | 4275.49%  |
| worker::game_loop                         | 1     | 622.7 MB | 623.0 MB | 622.7 MB | 2505.67%  |
| worker::inference_loop                    | 1     | 56.6 MB  | 56.7 MB  | 56.6 MB  | 227.87%   |
| tricked_engine::main                      | 1     | 24.9 MB  | 24.9 MB  | 24.9 MB  | 100.00%   |
| tricked_engine::run_training              | 1     | 24.8 MB  | 24.8 MB  | 24.8 MB  | 99.86%    |
| search::normalize_policy_distributions    | 631   | 1.7 KB   | 2.1 KB   | 1.0 MB   | 4.15%     |
| search::inject_gumbel_noise               | 626   | 1.3 KB   | 1.5 KB   | 795.9 KB | 3.13%     |
| worker::process_recurrent_inference       | 1221  | 564 B    | 564 B    | 672.5 KB | 2.64%     |
| search::compute_final_action_distribution | 560   | 1.2 KB   | 1.8 KB   | 660.3 KB | 2.59%     |
| worker::calculate_policy_targets          | 372   | 1.1 KB   | 1.1 KB   | 418.5 KB | 1.64%     |
| search::prune_candidates                  | 2016  | 176 B    | 476 B    | 347.1 KB | 1.36%     |
| worker::process_initial_inference         | 329   | 838 B    | 530 B    | 269.4 KB | 1.06%     |

---
## Big Model, Shallow Search
**Total Execution Time:** `781.15s`
- **RAM Usage:** `35362.45 MB`
- **GPU VRAM:** `1189.00 MB`
- **Nodes / Sec (NPS):** `0.04`

### Top Execution Bottlenecks
| Function                                  | Calls   | Avg       | P95       | Total      | % Total  |
| search::process_evaluation_responses      | 11089   | 3.69 s    | 6.98 s    | 40876.74 s | 5232.88% |
| search::expand_and_evaluate_candidates    | 11080   | 3.69 s    | 6.98 s    | 40852.14 s | 5229.73% |
| search::mcts_search                       | 3668    | 11.14 s   | 15.72 s   | 40844.79 s | 5228.79% |
| search::execute_sequential_halving        | 3601    | 11.34 s   | 15.74 s   | 40843.95 s | 5228.68% |
| tricked_engine::main                      | 1       | 781.42 s  | 781.68 s  | 781.15 s   | 100.00%  |
| tricked_engine::run_training              | 1       | 781.42 s  | 781.68 s  | 781.15 s   | 100.00%  |
| worker::game_loop                         | 1       | 779.80 s  | 780.07 s  | 779.87 s   | 99.84%   |
| worker::process_recurrent_inference       | 1124    | 631.78 ms | 1.16 s    | 710.11 s   | 90.91%   |
| optimization::train_step                  | 5       | 141.97 s  | 146.57 s  | 709.84 s   | 90.87%   |
| worker::process_initial_inference         | 395     | 169.64 ms | 716.18 ms | 67.01 s    | 8.58%    |
| worker::compute_max_depth                 | 4287559 | 772 ns    | 36 ns     | 3.31 s     | 0.42%    |
| search::compute_final_action_distribution | 3551    | 120.16 µs | 277.25 µs | 426.67 ms  | 0.05%    |
| search::traverse_tree_to_leaf             | 50477   | 3.61 µs   | 11.27 µs  | 182.19 ms  | 0.02%    |
| search::inject_gumbel_noise               | 3610    | 20.05 µs  | 60.29 µs  | 72.39 ms   | 0.01%    |
| search::prune_candidates                  | 11030   | 5.87 µs   | 11.13 µs  | 64.79 ms   | 0.01%    |

### Largest Memory Allocations
| Function                                  | Calls | Avg      | P95      | Total    | % Total  |
| search::mcts_search                       | 3668  | 1.6 MB   | 1.6 MB   | 5.7 GB   | 4926.12% |
| search::process_evaluation_responses      | 11089 | 451.8 KB | 1.6 MB   | 4.8 GB   | 4110.13% |
| search::execute_sequential_halving        | 3601  | 1.4 MB   | 1.6 MB   | 4.8 GB   | 4103.21% |
| search::expand_and_evaluate_candidates    | 11080 | 451.3 KB | 1.6 MB   | 4.8 GB   | 4102.05% |
| worker::game_loop                         | 1     | 221.6 MB | 221.6 MB | 221.6 MB | 186.12%  |
| tricked_engine::main                      | 1     | 119.0 MB | 119.1 MB | 119.0 MB | 100.00%  |
| tricked_engine::run_training              | 1     | 119.0 MB | 119.1 MB | 119.0 MB | 99.97%   |
| search::normalize_policy_distributions    | 3677  | 1.6 KB   | 2.1 KB   | 5.8 MB   | 4.86%    |
| search::inject_gumbel_noise               | 3610  | 1.4 KB   | 1.5 KB   | 5.0 MB   | 4.24%    |
| search::compute_final_action_distribution | 3551  | 629 B    | 1.1 KB   | 2.1 MB   | 1.79%    |
| search::prune_candidates                  | 11030 | 130 B    | 208 B    | 1.4 MB   | 1.16%    |
| worker::process_recurrent_inference       | 1124  | 756 B    | 756 B    | 829.8 KB | 0.68%    |
| worker::calculate_policy_targets          | 543   | 1.1 KB   | 1.1 KB   | 610.9 KB | 0.50%    |
| worker::process_initial_inference         | 395   | 979 B    | 722 B    | 377.7 KB | 0.31%    |
| optimization::train_step                  | 5     | 44.7 KB  | 44.7 KB  | 223.7 KB | 0.18%    |

---
## Big Model, Deep Search
**Total Execution Time:** `0.0s`
- **RAM Usage:** `37099.71 MB`
- **GPU VRAM:** `1138.00 MB`
- **Nodes / Sec (NPS):** `0.00`

### Top Execution Bottlenecks

### Largest Memory Allocations

---