use super::*;
use crossbeam_channel::unbounded;
use std::thread;
use std::time::Instant;

#[test]
fn test_inference_queue_parallel_saturation() {
    let num_workers = 8;
    let queue = FixedInferenceQueue::new(1024, num_workers);

    let start = Instant::now();
    let mut handles = vec![];

    for i in 0..num_workers {
        let q = queue.clone();
        handles.push(thread::spawn(move || {

            for _ in 0..50 {
                // Simulate MCTS fast evaluation logic
                thread::sleep(Duration::from_micros(100)); // Fast worker doing some MCTS

                let req = crate::mcts::EvaluationRequest {
                    is_initial: true,
                    board_bitmask: 0,
                    available_pieces: [0; 3],
                    recent_board_history: [0; 8],
                    history_len: 0,
                    recent_action_history: [0; 4],
                    action_history_len: 0,
                    difficulty: 0,
                    piece_action: 0,
                    piece_id: 0,
                    node_index: 0,
                    generation: 0,
                    worker_id: i,
                    parent_cache_index: 0,
                    leaf_cache_index: 0,
                    mailbox: std::sync::Arc::new(crate::mcts::mailbox::AtomicMailbox::new()),
                };

                q.push_batch(i, vec![req]).unwrap();
            }
            q.disconnect_producer();
        }));
    }

    let mut total_processed = 0;
    let mut loop_iterations = 0;

    while queue.active_producers.load(Ordering::SeqCst) > 0 || total_processed < num_workers * 50 {
        loop_iterations += 1;
        let (initial, recurrent) = queue
            .pop_batch_timeout(64, Duration::from_millis(50))
            .unwrap_or((vec![], vec![]));
        total_processed += initial.len() + recurrent.len();
    }

    for h in handles {
        h.join().unwrap();
    }

    let elapsed = start.elapsed();

    // 50 iterations of 100-microsecond waits ≈ 5ms total actual working time per thread.
    // It's parallel so it shouldn't take more than 200ms total if polling is efficient.
    assert!(
        elapsed < Duration::from_millis(1500),
        "Workers starved! Execution took {:?}",
        elapsed
    );
    // The inference thread should not loop uselessly waiting.
    assert!(
        loop_iterations < 2000,
        "Too many inefficient poll loops: {}",
        loop_iterations
    );
    assert_eq!(total_processed, num_workers * 50);
}

#[test]
fn test_zero_max_batch_size_returns_immediately() {
    let queue = FixedInferenceQueue::new(1024, 1);
    let start = Instant::now();
    let (initial, recurrent) = queue.pop_batch_timeout(0, Duration::from_secs(10)).unwrap();
    assert!(start.elapsed() < Duration::from_millis(5));
    assert!(initial.is_empty() && recurrent.is_empty());
}

#[test]
fn test_all_producers_blocked_causes_early_pop() {
    let queue = FixedInferenceQueue::new(1024, 2);

    let req = crate::mcts::EvaluationRequest {
        is_initial: true,
        board_bitmask: 0,
        available_pieces: [0; 3],
        recent_board_history: [0; 8],
        history_len: 0,
        recent_action_history: [0; 4],
        action_history_len: 0,
        difficulty: 0,
        piece_action: 0,
        piece_id: 0,
        node_index: 0,
        generation: 0,
        worker_id: 0,
        parent_cache_index: 0,
        leaf_cache_index: 0,
        mailbox: std::sync::Arc::new(crate::mcts::mailbox::AtomicMailbox::new()),
    };
    queue.push_batch(0, vec![req]).unwrap();

    queue.blocked_producers.store(2, Ordering::SeqCst);
    let start = Instant::now();
    let (initial, _) = queue
        .pop_batch_timeout(64, Duration::from_secs(10))
        .unwrap();

    // Should return immediately instead of waiting for 10 seconds because all producers are blocked
    assert!(start.elapsed() < Duration::from_millis(100));
    assert_eq!(initial.len(), 1);
}

#[test]
fn test_producer_disconnect_unblocks_inference() {
    let queue = FixedInferenceQueue::new(1024, 1);

    let q_clone = queue.clone();
    thread::spawn(move || {
        thread::sleep(Duration::from_millis(50));
        q_clone.disconnect_producer();
    });

    let start = Instant::now();
    let res = queue.pop_batch_timeout(64, Duration::from_secs(10));
    assert!(start.elapsed() < Duration::from_millis(500));
    assert!(res.is_err());
}

#[test]
fn test_microbatching_window_respects_time() {
    let queue = FixedInferenceQueue::new(1024, 1);

    let req = crate::mcts::EvaluationRequest {
        is_initial: true,
        board_bitmask: 0,
        available_pieces: [0; 3],
        recent_board_history: [0; 8],
        history_len: 0,
        recent_action_history: [0; 4],
        action_history_len: 0,
        difficulty: 0,
        piece_action: 0,
        piece_id: 0,
        node_index: 0,
        generation: 0,
        worker_id: 0,
        parent_cache_index: 0,
        leaf_cache_index: 0,
        mailbox: std::sync::Arc::new(crate::mcts::mailbox::AtomicMailbox::new()),
    };
    queue.push_batch(0, vec![req]).unwrap(); // Single item

    let start = Instant::now();
    let (initial, _) = queue
        .pop_batch_timeout(64, Duration::from_millis(100))
        .unwrap();

    // Because producer counts > blocked counts, it should wait for the micro-batching window (250us)!
    // Not the full 100ms timeout!
    assert!(start.elapsed() < Duration::from_millis(50));
    assert_eq!(initial.len(), 1);
}

#[test]
fn test_recurrent_and_initial_interleaved() {
    let queue = FixedInferenceQueue::new(1024, 1);


    let req_init = crate::mcts::EvaluationRequest {
        is_initial: true,
        board_bitmask: 0,
        available_pieces: [0; 3],
        recent_board_history: [0; 8],
        history_len: 0,
        recent_action_history: [0; 4],
        action_history_len: 0,
        difficulty: 0,
        piece_action: 0,
        piece_id: 0,
        node_index: 0,
        generation: 0,
        worker_id: 0,
        parent_cache_index: 0,
        leaf_cache_index: 0,
        mailbox: std::sync::Arc::new(crate::mcts::mailbox::AtomicMailbox::new()),
    };

    let req_recur = crate::mcts::EvaluationRequest {
        is_initial: false,
        board_bitmask: 0,
        available_pieces: [0; 3],
        recent_board_history: [0; 8],
        history_len: 0,
        recent_action_history: [0; 4],
        action_history_len: 0,
        difficulty: 0,
        piece_action: 0,
        piece_id: 0,
        node_index: 0,
        generation: 0,
        worker_id: 0,
        parent_cache_index: 0,
        leaf_cache_index: 0,
        mailbox: std::sync::Arc::new(crate::mcts::mailbox::AtomicMailbox::new()),
    };

    queue.push_batch(0, vec![req_init, req_recur]).unwrap();
    queue.blocked_producers.store(1, Ordering::SeqCst);

    let (initial, recurrent) = queue
        .pop_batch_timeout(64, Duration::from_millis(50))
        .unwrap();
    assert_eq!(initial.len(), 1);
    assert_eq!(recurrent.len(), 1);
}

#[test]
fn test_drop_queue_slot_guard_returns_slots() {
    let queue = FixedInferenceQueue::new(16384, 1);


    let req = crate::mcts::EvaluationRequest {
        is_initial: true,
        board_bitmask: 0,
        available_pieces: [0; 3],
        recent_board_history: [0; 8],
        history_len: 0,
        recent_action_history: [0; 4],
        action_history_len: 0,
        difficulty: 0,
        piece_action: 0,
        piece_id: 0,
        node_index: 0,
        generation: 0,
        worker_id: 0,
        parent_cache_index: 0,
        leaf_cache_index: 0,
        mailbox: std::sync::Arc::new(crate::mcts::mailbox::AtomicMailbox::new()),
    };

    queue.push_batch(0, vec![req]).unwrap();
    queue.blocked_producers.store(1, Ordering::SeqCst);

    let (initial, _) = queue
        .pop_batch_timeout(64, Duration::from_millis(50))
        .unwrap();
    assert_eq!(initial.len(), 1);

    assert_eq!(queue.free_slots.len(), 16383);
    drop(initial);
    // Automatically returns to free_slots!
    assert_eq!(queue.free_slots.len(), 16384);
}

#[test]
fn test_partial_starvation_does_not_deadlock() {
    let queue = FixedInferenceQueue::new(1024, 2);

    let req = crate::mcts::EvaluationRequest {
        is_initial: true,
        board_bitmask: 0,
        available_pieces: [0; 3],
        recent_board_history: [0; 8],
        history_len: 0,
        recent_action_history: [0; 4],
        action_history_len: 0,
        difficulty: 0,
        piece_action: 0,
        piece_id: 0,
        node_index: 0,
        generation: 0,
        worker_id: 0,
        parent_cache_index: 0,
        leaf_cache_index: 0,
        mailbox: std::sync::Arc::new(crate::mcts::mailbox::AtomicMailbox::new()),
    };

    // Producer 1 pushes a batch
    queue.push_batch(0, vec![req]).unwrap();

    // Setup state: Both producers are active. Producer 1 goes blocked. Producer 2 is running (starving queue).
    queue.blocked_producers.store(1, Ordering::SeqCst);

    let start = Instant::now();
    // Since Producer 2 is active, the queue will not wait the full 50ms artificially; it will jump out after 250us.
    let (initial, _) = queue
        .pop_batch_timeout(64, Duration::from_millis(100))
        .unwrap();

    assert!(
        start.elapsed() < Duration::from_millis(50),
        "Queue waited too long when partial starvation happened"
    );
    assert_eq!(initial.len(), 1);
}

#[test]
fn test_single_producer_bursts() {
    let queue = FixedInferenceQueue::new(1024, 1);


    let mut reqs = Vec::new();
    for _ in 0..100 {
        reqs.push(crate::mcts::EvaluationRequest {
            is_initial: true,
            board_bitmask: 0,
            available_pieces: [0; 3],
            recent_board_history: [0; 8],
            history_len: 0,
            recent_action_history: [0; 4],
            action_history_len: 0,
            difficulty: 0,
            piece_action: 0,
            piece_id: 0,
            node_index: 0,
            generation: 0,
            worker_id: 0,
            parent_cache_index: 0,
            leaf_cache_index: 0,
            mailbox: std::sync::Arc::new(crate::mcts::mailbox::AtomicMailbox::new()),
        });
    }

    queue.push_batch(0, reqs).unwrap();
    queue.blocked_producers.store(1, Ordering::SeqCst);

    let mut total = 0;
    while total < 100 {
        let (initial, _) = queue
            .pop_batch_timeout(64, Duration::from_millis(50))
            .unwrap();
        total += initial.len();
    }
    assert_eq!(total, 100);
}

#[test]
fn test_massive_concurrency_fuzzing() {
    let num_workers = 20;
    let queue = FixedInferenceQueue::new(4096, num_workers);
    let mut handles = vec![];

    for i in 0..num_workers {
        let q = queue.clone();
        handles.push(thread::spawn(move || {

            for _ in 0..100 {
                let req = crate::mcts::EvaluationRequest {
                    is_initial: true,
                    board_bitmask: 0,
                    available_pieces: [0; 3],
                    recent_board_history: [0; 8],
                    history_len: 0,
                    recent_action_history: [0; 4],
                    action_history_len: 0,
                    difficulty: 0,
                    piece_action: 0,
                    piece_id: 0,
                    node_index: 0,
                    generation: 0,
                    worker_id: i,
                    parent_cache_index: 0,
                    leaf_cache_index: 0,
                    mailbox: std::sync::Arc::new(crate::mcts::mailbox::AtomicMailbox::new()),
                };
                q.push_batch(i, vec![req]).unwrap();
            }
            q.disconnect_producer();
        }));
    }

    let mut popped = 0;
    while queue.active_producers.load(Ordering::SeqCst) > 0 || popped < num_workers * 100 {
        let (initial, _) = queue
            .pop_batch_timeout(128, Duration::from_millis(10))
            .unwrap_or((vec![], vec![]));
        popped += initial.len();
    }

    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(popped, num_workers * 100);
}

#[test]
fn test_timeout_triggers_with_no_data() {
    let queue = FixedInferenceQueue::new(1024, 1);
    let start = Instant::now();

    let (initial, recurrent) = queue
        .pop_batch_timeout(64, Duration::from_millis(50))
        .unwrap();
    let elapsed = start.elapsed();

    assert!(elapsed >= Duration::from_millis(45));
    assert!(initial.is_empty());
    assert!(recurrent.is_empty());
}

#[test]
fn test_inference_queue_starvation_recovery() {
    let queue = FixedInferenceQueue::new(1024, 4); // 4 producers


    // Push requests
    for _ in 0..5 {
        let req = crate::mcts::EvaluationRequest {
            is_initial: true,
            board_bitmask: 0,
            available_pieces: [0; 3],
            recent_board_history: [0; 8],
            history_len: 0,
            recent_action_history: [0; 4],
            action_history_len: 0,
            difficulty: 0,
            piece_action: 0,
            piece_id: 0,
            node_index: 0,
            generation: 0,
            worker_id: 0,
            parent_cache_index: 0,
            leaf_cache_index: 0,
            mailbox: std::sync::Arc::new(crate::mcts::mailbox::AtomicMailbox::new()),
        };
        queue.push_batch(0, vec![req]).unwrap();
    }

    // Disconnect half the producers (simulate dropping/panicking)
    queue.disconnect_producer();
    queue.disconnect_producer();
    // and let's assume we are waiting for a batch size of 10 but we only have 5.
    // Producer 3 and 4 are still alive but doing nothing.

    let start = std::time::Instant::now();
    // Since producers are still active, it should just timeout and return what it has.
    // It shouldn't spin infinitely due to backoff bugs.
    let (initial, _) = queue
        .pop_batch_timeout(10, Duration::from_millis(150))
        .unwrap();
    let elapsed = start.elapsed();

    assert_eq!(initial.len(), 5);
    assert!(
        elapsed < Duration::from_millis(50),
        "Micro-batching window was ignored, elapsed: {:?}",
        elapsed
    );
}
