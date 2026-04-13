use crossbeam::queue::ArrayQueue;
#[cfg(loom)]
use loom::sync::atomic::{AtomicUsize, Ordering};
use std::cell::UnsafeCell;
#[cfg(not(loom))]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::mcts::EvaluationRequest;

pub struct QueueSlotGuard {
    pub slot: usize,
    free_q: Option<Arc<ArrayQueue<usize>>>,
}

impl QueueSlotGuard {
    pub fn new(slot: usize, free_q: Arc<ArrayQueue<usize>>) -> Self {
        Self {
            slot,
            free_q: Some(free_q),
        }
    }

    pub fn disarm(mut self) -> usize {
        self.free_q = None;
        self.slot
    }
}

impl Drop for QueueSlotGuard {
    fn drop(&mut self) {
        if let Some(q) = &self.free_q {
            let _ = q.push(self.slot);
        }
    }
}

pub struct FixedInferenceQueue {
    pub initial_ready: Arc<ArrayQueue<usize>>,
    pub recurrent_ready: Arc<ArrayQueue<usize>>,
    pub free_slots: Arc<ArrayQueue<usize>>,

    pub initial_boards_pinned: UnsafeCell<Vec<i64>>,
    pub initial_avail_pinned: UnsafeCell<Vec<i32>>,
    pub initial_hist_pinned: UnsafeCell<Vec<i64>>,
    pub initial_acts_pinned: UnsafeCell<Vec<i32>>,
    pub initial_diff_pinned: UnsafeCell<Vec<i32>>,
    pub recurrent_actions_pinned: UnsafeCell<Vec<i64>>,
    pub recurrent_ids_pinned: UnsafeCell<Vec<i64>>,

    // UnsafeCell is safe here because we use slot ownership via channels to provide zero-contention
    // and mutual exclusivity, entirely eliminating the Mutex cache line bouncing lock overhead.
    pub metadata: Vec<std::cell::UnsafeCell<Option<(EvaluationRequest, std::time::Instant)>>>,

    pub active_producers: AtomicUsize,
    pub blocked_producers: AtomicUsize,

    pub latency_sum_nanos: loom_or_std::AtomicU64,
    pub latency_count: loom_or_std::AtomicU64,
}

#[cfg(loom)]
mod loom_or_std {
    pub use loom::sync::atomic::AtomicU64;
}
#[cfg(not(loom))]
mod loom_or_std {
    pub use std::sync::atomic::AtomicU64;
}

unsafe impl Send for FixedInferenceQueue {}
unsafe impl Sync for FixedInferenceQueue {}

impl FixedInferenceQueue {
    pub fn new(buffer_capacity_limit: usize, total_producers: usize) -> Arc<Self> {
        let capacity = buffer_capacity_limit.max(16384);
        let initial_ready = Arc::new(ArrayQueue::new(capacity));
        let recurrent_ready = Arc::new(ArrayQueue::new(capacity));
        let free_slots = Arc::new(ArrayQueue::new(capacity));

        let initial_boards_pinned = UnsafeCell::new(vec![0; capacity * 2]);
        let initial_avail_pinned = UnsafeCell::new(vec![0; capacity * 3]);
        let initial_hist_pinned = UnsafeCell::new(vec![0; capacity * 14]);
        let initial_acts_pinned = UnsafeCell::new(vec![0; capacity * 3]);
        let initial_diff_pinned = UnsafeCell::new(vec![0; capacity]);

        let recurrent_actions = UnsafeCell::new(vec![0; capacity]);
        let recurrent_ids = UnsafeCell::new(vec![0; capacity]);

        let mut metadata = Vec::with_capacity(capacity);
        for i in 0..capacity {
            let _ = free_slots.push(i);
            metadata.push(std::cell::UnsafeCell::new(None));
        }

        Arc::new(Self {
            initial_ready,
            recurrent_ready,
            free_slots,
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
            latency_sum_nanos: loom_or_std::AtomicU64::new(0),
            latency_count: loom_or_std::AtomicU64::new(0),
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
            let mut spins = 0;
            let slot = loop {
                if let Some(s) = self.free_slots.pop() {
                    break s;
                }
                if self.active_producers.load(Ordering::Relaxed) == 0 {
                    return Err(());
                }
                if spins < 100 {
                    std::hint::spin_loop();
                    spins += 1;
                } else {
                    std::thread::sleep(Duration::from_micros(1));
                }
            };

            let guard = QueueSlotGuard::new(slot, self.free_slots.clone());

            let is_initial = req.is_initial;
            if is_initial {
                unsafe {
                    let ptr_boards = (*self.initial_boards_pinned.get()).as_mut_ptr();
                    let len_boards = (*self.initial_boards_pinned.get()).len();
                    debug_assert!(
                        slot * 2 + 1 < len_boards,
                        "Queue OOB access into initial boards"
                    );
                    *ptr_boards.add(slot * 2) = (req.board_bitmask & 0xFFFFFFFFFFFFFFFF) as i64;
                    *ptr_boards.add(slot * 2 + 1) = (req.board_bitmask >> 64) as i64;

                    let ptr_avail = (*self.initial_avail_pinned.get()).as_mut_ptr();
                    for i in 0..3 {
                        *ptr_avail.add(slot * 3 + i) = req.available_pieces[i];
                    }

                    let ptr_hist = (*self.initial_hist_pinned.get()).as_mut_ptr();
                    for i in 0..7 {
                        let hist_board = if req.history_len > i {
                            req.recent_board_history[req.history_len - 1 - i]
                        } else {
                            req.board_bitmask
                        };
                        *ptr_hist.add(slot * 14 + i * 2) = (hist_board & 0xFFFFFFFFFFFFFFFF) as i64;
                        *ptr_hist.add(slot * 14 + i * 2 + 1) = (hist_board >> 64) as i64;
                    }

                    let ptr_acts = (*self.initial_acts_pinned.get()).as_mut_ptr();
                    for i in 0..3 {
                        let action = if req.action_history_len > i {
                            req.recent_action_history[req.action_history_len - 1 - i]
                        } else {
                            -1
                        };
                        *ptr_acts.add(slot * 3 + i) = action;
                    }

                    let ptr_diff = (*self.initial_diff_pinned.get()).as_mut_ptr();
                    *ptr_diff.add(slot) = req.difficulty;
                }
            } else {
                unsafe {
                    let ptr_actions = (*self.recurrent_actions_pinned.get()).as_mut_ptr();
                    let len_actions = (*self.recurrent_actions_pinned.get()).len();
                    debug_assert!(
                        slot < len_actions,
                        "Queue OOB access into recurrent actions"
                    );
                    *ptr_actions.add(slot) = req.piece_action;

                    let ptr_ids = (*self.recurrent_ids_pinned.get()).as_mut_ptr();
                    let len_ids = (*self.recurrent_ids_pinned.get()).len();
                    debug_assert!(slot < len_ids, "Queue OOB access into recurrent ids");
                    *ptr_ids.add(slot) = req.piece_id;
                }
            }

            unsafe {
                *self.metadata[slot].get() = Some((req, std::time::Instant::now()));
            }

            let final_slot = guard.disarm();

            let target_q = if is_initial {
                &self.initial_ready
            } else {
                &self.recurrent_ready
            };

            let backoff = crossbeam::utils::Backoff::new();
            while target_q.push(final_slot).is_err() {
                backoff.spin();
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
        let backoff = crossbeam::utils::Backoff::new();

        loop {
            if let Some(slot) = self.initial_ready.pop() {
                initial_batch.push(QueueSlotGuard::new(slot, self.free_slots.clone()));
                break;
            }
            if let Some(slot) = self.recurrent_ready.pop() {
                recurrent_batch.push(QueueSlotGuard::new(slot, self.free_slots.clone()));
                break;
            }

            if self.active_producers.load(Ordering::SeqCst) == 0 {
                return Err(());
            }

            let elapsed = start.elapsed();
            if elapsed >= timeout {
                return Ok((initial_batch, recurrent_batch));
            }

            if elapsed < Duration::from_micros(100) {
                backoff.snooze();
            } else {
                let remaining = timeout.saturating_sub(elapsed);
                if remaining.is_zero() {
                    return Ok((initial_batch, recurrent_batch));
                }
                backoff.snooze();
            }
        }

        // We got at least 1 item. Now quickly grab any others that are IMMEDIATELY available,
        // or wait a tiny micro-batching window (250 microseconds) to gather more.
        let gather_window = Duration::from_micros(250);
        let start_gather = std::time::Instant::now();
        let small_backoff = crossbeam::utils::Backoff::new();

        while (initial_batch.len() + recurrent_batch.len()) < max_batch_size {
            let remaining_gather = gather_window.saturating_sub(start_gather.elapsed());
            if remaining_gather.is_zero() {
                break;
            }

            if let Some(slot) = self.initial_ready.pop() {
                initial_batch.push(QueueSlotGuard::new(slot, self.free_slots.clone()));
                continue;
            }
            if let Some(slot) = self.recurrent_ready.pop() {
                recurrent_batch.push(QueueSlotGuard::new(slot, self.free_slots.clone()));
                continue;
            }

            small_backoff.snooze();
        }

        let mut sum_nanos = 0;
        let mut cnt = 0;
        for guard in initial_batch.iter().chain(recurrent_batch.iter()) {
            unsafe {
                if let Some((_, queued_at)) = &*self.metadata[guard.slot].get() {
                    sum_nanos += queued_at.elapsed().as_nanos() as u64;
                    cnt += 1;
                }
            }
        }
        if cnt > 0 {
            self.latency_sum_nanos
                .fetch_add(sum_nanos, Ordering::Relaxed);
            self.latency_count.fetch_add(cnt, Ordering::Relaxed);
        }

        Ok((initial_batch, recurrent_batch))
    }
}

#[cfg(test)]
mod queue_tests;
