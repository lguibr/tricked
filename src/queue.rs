use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::mcts::EvalReq;

pub struct QueueState {
    pub pending_count: usize,
    pub active_producers: usize,
}

pub struct FixedInferenceQueue {
    pub slots: Vec<Mutex<Vec<EvalReq>>>,
    pub notifier: Condvar,
    pub state: Mutex<QueueState>,
}

impl FixedInferenceQueue {
    pub fn new(capacity: usize, total_producers: usize) -> Arc<Self> {
        let mut slots = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            slots.push(Mutex::new(Vec::new()));
        }
        Arc::new(Self {
            slots,
            notifier: Condvar::new(),
            state: Mutex::new(QueueState {
                pending_count: 0,
                active_producers: total_producers,
            }),
        })
    }

    pub fn push(&self, worker_id: usize, req: EvalReq) -> Result<(), ()> {
        {
            let mut slot = self.slots[worker_id].lock().unwrap();
            slot.push(req);
        }

        let mut state = self.state.lock().unwrap();
        if state.active_producers == 0 {
            return Err(());
        }
        state.pending_count += 1;
        self.notifier.notify_one();
        Ok(())
    }

    pub fn disconnect_producer(&self) {
        let mut state = self.state.lock().unwrap();
        state.active_producers = state.active_producers.saturating_sub(1);
        if state.active_producers == 0 {
            self.notifier.notify_all();
        }
    }

    pub fn pop_batch_timeout(
        &self,
        max_batch_size: usize,
        timeout: Duration,
    ) -> Result<Vec<EvalReq>, ()> {
        let mut batch = Vec::with_capacity(max_batch_size);
        let mut state = self.state.lock().unwrap();

        if state.pending_count == 0 {
            if state.active_producers == 0 {
                return Err(());
            }
            let (new_state, timeout_result) = self.notifier.wait_timeout(state, timeout).unwrap();
            state = new_state;
            if timeout_result.timed_out() || state.pending_count == 0 {
                if state.active_producers == 0 && state.pending_count == 0 {
                    return Err(());
                }
                return Ok(batch);
            }
        }

        for slot_mutex in &self.slots {
            if batch.len() >= max_batch_size {
                break;
            }
            if let Ok(mut slot) = slot_mutex.try_lock() {
                while let Some(req) = slot.pop() {
                    batch.push(req);
                    state.pending_count -= 1;
                    if batch.len() >= max_batch_size {
                        break;
                    }
                }
            }
        }

        Ok(batch)
    }
}
