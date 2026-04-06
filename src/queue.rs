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

        while (initial_batch.len() + recurrent_batch.len()) < max_batch_size {
            let elapsed = start.elapsed();
            if elapsed >= timeout {
                break;
            }
            let remaining = timeout - elapsed;

            let wait = remaining.min(Duration::from_micros(50));
            // We use select! or opportunistic checking. Opportunistic is fine.
            let mut got_something = false;

            while let Ok(slot) = self.initial_ready_rx.try_recv() {
                initial_batch.push(QueueSlotGuard::new(slot, self.free_tx.clone()));
                got_something = true;
                if (initial_batch.len() + recurrent_batch.len()) == max_batch_size {
                    break;
                }
            }

            if (initial_batch.len() + recurrent_batch.len()) == max_batch_size {
                break;
            }

            while let Ok(slot) = self.recurrent_ready_rx.try_recv() {
                recurrent_batch.push(QueueSlotGuard::new(slot, self.free_tx.clone()));
                got_something = true;
                if (initial_batch.len() + recurrent_batch.len()) == max_batch_size {
                    break;
                }
            }

            if !got_something {
                let is_everyone_blocked = self.blocked_producers.load(Ordering::SeqCst)
                    >= self.active_producers.load(Ordering::SeqCst);
                if is_everyone_blocked && (initial_batch.len() + recurrent_batch.len()) > 0 {
                    break;
                }

                crossbeam_channel::select! {
                    recv(self.initial_ready_rx) -> msg => {
                        if let Ok(slot) = msg {
                            initial_batch.push(QueueSlotGuard::new(slot, self.free_tx.clone()));
                        }
                    }
                    recv(self.recurrent_ready_rx) -> msg => {
                        if let Ok(slot) = msg {
                            recurrent_batch.push(QueueSlotGuard::new(slot, self.free_tx.clone()));
                        }
                    }
                    default(remaining) => {
                        break;
                    }
                }
            }
        }

        if initial_batch.is_empty()
            && recurrent_batch.is_empty()
            && self.active_producers.load(Ordering::SeqCst) == 0
        {
            return Err(());
        }

        Ok((initial_batch, recurrent_batch))
    }
}
