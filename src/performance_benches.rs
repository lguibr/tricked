#[cfg(test)]
mod performance_tests {
    use crate::core::board::GameStateExt;
    use crate::core::features::extract_feature_native;
    use crate::mcts::{advance_root, MctsTree};
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
                                    generation: 0,
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
                            popped += batch.0.len() + batch.1.len();
                            // BatchGuard drops naturally here
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
        let rb = crate::train::buffer::ReplayBuffer::new(10000, 5, 10, 32);
        // Fill dummy data...
        let cases = [128, 512, 1024];
        for &batch_size in &cases {
            let start = Instant::now();
            for _ in 0..10 {
                let _ = rb.sample_batch(2, 1.0);
            }
            println!("RB Sample (Batch {}): {:?}", batch_size, start.elapsed());
        }
    }

    // 4. MCTS Tree GC Performance
    #[test]
    fn bench_mcts_gc_traversal() {
        let cases = [100, 1000, 5000];
        for &nodes in &cases {
            let mut arena = Vec::with_capacity(nodes);
            for _ in 0..nodes {
                arena.push(LatentNode::new(0.0, 0, 0));
            }
            for (i, node) in arena.iter().enumerate().take(nodes - 1) {
                node.first_child
                    .store((i + 1) as u32, std::sync::atomic::Ordering::SeqCst);
                // Create a deep linked list
            }
            let tree = MctsTree {
                arena: crate::mcts::SharedArena(std::sync::Arc::new(arena)),
                node_free_list: std::sync::Arc::new(crossbeam_queue::ArrayQueue::new(nodes)),
                gpu_cache_free_list: std::sync::Arc::new(crossbeam_queue::ArrayQueue::new(nodes)),
                current_generation: 0,
                root_index: 0,
                max_tree_nodes: nodes as u32,
                max_cache_slots: nodes as u32,
            };

            let (tx, _) = crossbeam_channel::unbounded();
            let start = Instant::now();
            let _ = advance_root(tree, 1, &tx);
            println!("MCTS advance_root ({} nodes): {:?}", nodes, start.elapsed());
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
            let mut _arena = Vec::with_capacity(300);
            for _ in 0..300 {
                _arena.push(LatentNode::new(0.0, 0, 0));
            }
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
                                generation: 0,
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
                        popped += batch.0.len() + batch.1.len();
                        // BatchGuard drops naturally here
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
                generation: 0,
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
        let rb = std::sync::Arc::new(crate::train::buffer::ReplayBuffer::new(200_000, 5, 10, 256));
        let workers = 16;
        let start = Instant::now();
        std::thread::scope(|s| {
            for _ in 0..workers {
                let rb_clone = rb.clone();
                s.spawn(move || {
                    for _ in 0..10_000 {
                        rb_clone.add_game(crate::train::buffer::OwnedGameData {
                            difficulty_setting: 6,
                            episode_score: 0.0,
                            steps: vec![crate::train::buffer::GameStep {
                                board_state: [0, 0],
                                available_pieces: [0, -1, -1],
                                action_taken: 0,
                                piece_identifier: 0,
                                value_prefix_received: 0.0,
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
                        generation: 0,
                        worker_id: 0,
                        parent_cache_index: 0,
                        leaf_cache_index: 0,
                        evaluation_request_transmitter,
                    }],
                );
            }
            let s = Instant::now();
            let _batch = queue
                .pop_batch_timeout(size, std::time::Duration::from_millis(10))
                .unwrap();
            // Batch drops naturally
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
        let elapsed = start.elapsed();
        println!("MCTS Sequential Halving Deep: {:?}", elapsed);
        assert!(
            elapsed.as_secs_f32() < 5.0,
            "MCTS Deep Search performance severely degraded! Took {:?}",
            elapsed
        );
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
        let free_q = std::sync::Arc::new(crossbeam_queue::ArrayQueue::new(100_000));
        for i in 1..100_000 {
            let _ = free_q.push(i);
        }
        let mut arena_vec = Vec::with_capacity(100_000);
        for _ in 0..100_000 {
            arena_vec.push(LatentNode::new(0.0, 0, 0));
        }
        let tree = MctsTree {
            arena: crate::mcts::SharedArena(std::sync::Arc::new(arena_vec)),
            node_free_list: free_q,
            gpu_cache_free_list: std::sync::Arc::new(crossbeam_queue::ArrayQueue::new(50000)),
            current_generation: 0,
            root_index: 0,
            max_tree_nodes: 50000,
            max_cache_slots: 50000,
        };
        let start = Instant::now();
        for _ in 0..50_000 {
            tree.node_free_list.pop();
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
