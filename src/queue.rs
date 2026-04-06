use crossbeam_channel::{bounded, Receiver, Sender};
#[cfg(loom)]
use loom::sync::atomic::{AtomicUsize, Ordering};
#[cfg(not(loom))]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tch::{Device, Kind, Tensor};

use crate::mcts::EvaluationRequest;

pub struct QueueSlotGuard {
    pub slot: usize,
    free_tx: Option<Sender<usize>>,
}

impl QueueSlotGuard {
    pub fn new(slot: usize, free_tx: Sender<usize>) -> Self {
        Self {
            slot,
            free_tx: Some(free_tx),
        }
    }

    pub fn disarm(mut self) -> usize {
        self.free_tx = None;
        self.slot
    }
}

impl Drop for QueueSlotGuard {
    fn drop(&mut self) {
        if let Some(tx) = &self.free_tx {
            let _ = tx.send(self.slot);
        }
    }
}

pub struct FixedInferenceQueue {
    pub initial_ready_tx: Sender<usize>,
    pub initial_ready_rx: Receiver<usize>,
    pub recurrent_ready_tx: Sender<usize>,
    pub recurrent_ready_rx: Receiver<usize>,
    pub free_tx: Sender<usize>,
    pub free_rx: Receiver<usize>,

    pub initial_boards_pinned: Tensor,
    pub initial_avail_pinned: Tensor,
    pub initial_hist_pinned: Tensor,
    pub initial_acts_pinned: Tensor,
    pub initial_diff_pinned: Tensor,
    pub recurrent_actions_pinned: Tensor,
    pub recurrent_ids_pinned: Tensor,

    // UnsafeCell is safe here because we use slot ownership via channels to provide zero-contention
    // and mutual exclusivity, entirely eliminating the Mutex cache line bouncing lock overhead.
    pub metadata: Vec<std::cell::UnsafeCell<Option<EvaluationRequest>>>,

    pub active_producers: AtomicUsize,
    pub blocked_producers: AtomicUsize,
}

unsafe impl Send for FixedInferenceQueue {}
unsafe impl Sync for FixedInferenceQueue {}

impl FixedInferenceQueue {
    pub fn new(buffer_capacity_limit: usize, total_producers: usize) -> Arc<Self> {
        let capacity = buffer_capacity_limit.max(16384);
        let (initial_ready_tx, initial_ready_rx) = bounded(capacity);
        let (recurrent_ready_tx, recurrent_ready_rx) = bounded(capacity);
        let (free_tx, free_rx) = bounded(capacity);

        let pin = |size: &[i64], kind: Kind| {
            let t = Tensor::zeros(size, (kind, Device::Cpu));
            if tch::Cuda::is_available() {
                t.pin_memory(Device::Cuda(0))
            } else {
                t
            }
        };

        let initial_boards_pinned = pin(&[capacity as i64, 2], Kind::Int64);
        let initial_avail_pinned = pin(&[capacity as i64, 3], Kind::Int);
        let initial_hist_pinned = pin(&[capacity as i64, 7, 2], Kind::Int64);
        let initial_acts_pinned = pin(&[capacity as i64, 3], Kind::Int);
        let initial_diff_pinned = pin(&[capacity as i64], Kind::Int);

        let recurrent_actions = pin(&[capacity as i64], Kind::Int64);
        let recurrent_ids = pin(&[capacity as i64], Kind::Int64);

        let mut metadata = Vec::with_capacity(capacity);
        for i in 0..capacity {
            free_tx.send(i).unwrap();
            metadata.push(std::cell::UnsafeCell::new(None));
        }

        Arc::new(Self {
            initial_ready_tx,
            initial_ready_rx,
            recurrent_ready_tx,
            recurrent_ready_rx,
            free_tx,
            free_rx,
            initial_boards_pinned,
            initial_avail_pinned,
            initial_hist_pinned,
            initial_acts_pinned,
            initial_diff_pinned,
            recurrent_actions_pinned: recurrent_actions,
            recurrent_ids_pinned: recurrent_ids,
            metadata,
            active_producers: AtomicUsize::new(total_producers),
            blocked_producers: AtomicUsize::new(0),
        })
    }

    #[allow(clippy::result_unit_err)]
    pub fn push_batch(
        &self,
        _worker_id: usize,
        reqs: impl IntoIterator<Item = EvaluationRequest>,
    ) -> Result<(), ()> {
        // Workers execute this. They pull free slots, write to pinned memory, and push ready indices.
        for req in reqs {
            let slot = match self.free_rx.recv() {
                Ok(s) => s,
                Err(_) => return Err(()),
            };

            let guard = QueueSlotGuard::new(slot, self.free_tx.clone());

            let is_initial = req.is_initial;
            if is_initial {
                unsafe {
                    let ptr_boards = self.initial_boards_pinned.data_ptr() as *mut i64;
                    *ptr_boards.add(slot * 2) = (req.board_bitmask & 0xFFFFFFFFFFFFFFFF) as i64;
                    *ptr_boards.add(slot * 2 + 1) = (req.board_bitmask >> 64) as i64;

                    let ptr_avail = self.initial_avail_pinned.data_ptr() as *mut i32;
                    for i in 0..3 {
                        *ptr_avail.add(slot * 3 + i) = req.available_pieces[i];
                    }

                    let ptr_hist = self.initial_hist_pinned.data_ptr() as *mut i64;
                    for i in 0..7 {
                        let hist_board = if req.history_len > i {
                            req.recent_board_history[req.history_len - 1 - i]
                        } else {
                            req.board_bitmask
                        };
                        *ptr_hist.add(slot * 14 + i * 2) = (hist_board & 0xFFFFFFFFFFFFFFFF) as i64;
                        *ptr_hist.add(slot * 14 + i * 2 + 1) = (hist_board >> 64) as i64;
                    }

                    let ptr_acts = self.initial_acts_pinned.data_ptr() as *mut i32;
                    for i in 0..3 {
                        let action = if req.action_history_len > i {
                            req.recent_action_history[req.action_history_len - 1 - i]
                        } else {
                            -1
                        };
                        *ptr_acts.add(slot * 3 + i) = action;
                    }

                    let ptr_diff = self.initial_diff_pinned.data_ptr() as *mut i32;
                    *ptr_diff.add(slot) = req.difficulty;
                }
            } else {
                unsafe {
                    let ptr_actions = self.recurrent_actions_pinned.data_ptr() as *mut i64;
                    *ptr_actions.add(slot) = req.piece_action;

                    let ptr_ids = self.recurrent_ids_pinned.data_ptr() as *mut i64;
                    *ptr_ids.add(slot) = req.piece_id;
                }
            }

            unsafe {
                *self.metadata[slot].get() = Some(req);
            }

            let final_slot = guard.disarm();

            if is_initial {
                let _ = self.initial_ready_tx.send(final_slot);
            } else {
                let _ = self.recurrent_ready_tx.send(final_slot);
            }
        }
        Ok(())
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
    ) -> Result<(Vec<QueueSlotGuard>, Vec<QueueSlotGuard>), ()> {
        let mut initial_batch = Vec::new();
        let mut recurrent_batch = Vec::new();

        if max_batch_size == 0 {
            return Ok((initial_batch, recurrent_batch));
        }

        let start = std::time::Instant::now();
        let loop_interval = std::time::Duration::from_millis(10);

        loop {
            if let Ok(slot) = self.initial_ready_rx.try_recv() {
                initial_batch.push(QueueSlotGuard::new(slot, self.free_tx.clone()));
                break;
            }
            if let Ok(slot) = self.recurrent_ready_rx.try_recv() {
                recurrent_batch.push(QueueSlotGuard::new(slot, self.free_tx.clone()));
                break;
            }

            if self.active_producers.load(Ordering::SeqCst) == 0 {
                return Err(());
            }

            let elapsed = start.elapsed();
            if elapsed >= timeout {
                return Ok((initial_batch, recurrent_batch));
            }

            let wait_time = loop_interval.min(timeout - elapsed);

            crossbeam_channel::select! {
                recv(self.initial_ready_rx) -> msg => {
                    if let Ok(slot) = msg {
                        initial_batch.push(QueueSlotGuard::new(slot, self.free_tx.clone()));
                        break;
                    }
                }
                recv(self.recurrent_ready_rx) -> msg => {
                    if let Ok(slot) = msg {
                        recurrent_batch.push(QueueSlotGuard::new(slot, self.free_tx.clone()));
                        break;
                    }
                }
                default(wait_time) => {}
            }
        }

        // We got at least 1 item. Now quickly grab any others that are IMMEDIATELY available,
        // or wait a tiny micro-batching window (250 microseconds) to gather more.
        let gather_window = Duration::from_micros(250);
        let start_gather = std::time::Instant::now();

        while (initial_batch.len() + recurrent_batch.len()) < max_batch_size {
            if let Ok(slot) = self.initial_ready_rx.try_recv() {
                initial_batch.push(QueueSlotGuard::new(slot, self.free_tx.clone()));
                continue;
            }
            if let Ok(slot) = self.recurrent_ready_rx.try_recv() {
                recurrent_batch.push(QueueSlotGuard::new(slot, self.free_tx.clone()));
                continue;
            }

            // If all active producers are currently blocked waiting for us, fire the batch immediately!
            if self.blocked_producers.load(Ordering::SeqCst)
                >= self.active_producers.load(Ordering::SeqCst)
            {
                break;
            }

            if start_gather.elapsed() > gather_window {
                break;
            }
            std::hint::spin_loop();
        }

        Ok((initial_batch, recurrent_batch))
    }
}

#[cfg(test)]
mod tests {
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
                let (tx, _rx) = unbounded();
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
                        evaluation_request_transmitter: tx.clone(),
                    };

                    q.push_batch(i, vec![req]).unwrap();
                }
                q.disconnect_producer();
            }));
        }

        let mut total_processed = 0;
        let mut loop_iterations = 0;

        while queue.active_producers.load(Ordering::SeqCst) > 0
            || total_processed < num_workers * 50
        {
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
            evaluation_request_transmitter: unbounded().0,
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
            evaluation_request_transmitter: unbounded().0,
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
        let (tx, _) = unbounded();

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
            evaluation_request_transmitter: tx.clone(),
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
            evaluation_request_transmitter: tx.clone(),
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
        let (tx, _) = unbounded();

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
            evaluation_request_transmitter: tx.clone(),
        };

        queue.push_batch(0, vec![req]).unwrap();
        queue.blocked_producers.store(1, Ordering::SeqCst);

        let (initial, _) = queue
            .pop_batch_timeout(64, Duration::from_millis(50))
            .unwrap();
        assert_eq!(initial.len(), 1);

        assert_eq!(queue.free_rx.len(), 16383);
        drop(initial);
        // Automatically returns to free_tx!
        assert_eq!(queue.free_rx.len(), 16384);
    }

    #[test]
    fn test_partial_starvation_does_not_deadlock() {
        let queue = FixedInferenceQueue::new(1024, 2);
        let (tx, _) = unbounded();
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
            evaluation_request_transmitter: tx,
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
        let (tx, _) = unbounded();

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
                evaluation_request_transmitter: tx.clone(),
            });
        }

        queue.push_batch(0, reqs).unwrap();
        queue.blocked_producers.store(1, Ordering::SeqCst);

        let (initial, _) = queue
            .pop_batch_timeout(64, Duration::from_millis(50))
            .unwrap();
        assert_eq!(initial.len(), 64);

        let (initial2, _) = queue
            .pop_batch_timeout(64, Duration::from_millis(50))
            .unwrap();
        assert_eq!(initial2.len(), 36);
    }

    #[test]
    fn test_massive_concurrency_fuzzing() {
        let num_workers = 20;
        let queue = FixedInferenceQueue::new(4096, num_workers);
        let mut handles = vec![];

        for i in 0..num_workers {
            let q = queue.clone();
            handles.push(thread::spawn(move || {
                let (tx, _) = unbounded();
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
                        evaluation_request_transmitter: tx.clone(),
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
}
