use crossbeam_channel::{bounded, Receiver, Sender};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tch::{Device, Kind, Tensor};

use crate::mcts::EvaluationRequest;

pub struct FixedInferenceQueue {
    pub initial_ready_tx: Sender<usize>,
    pub initial_ready_rx: Receiver<usize>,
    pub recurrent_ready_tx: Sender<usize>,
    pub recurrent_ready_rx: Receiver<usize>,
    pub free_tx: Sender<usize>,
    pub free_rx: Receiver<usize>,

    pub initial_states_pinned: Tensor,
    pub recurrent_actions_pinned: Tensor,
    pub recurrent_ids_pinned: Tensor,

    // UnsafeCell isn't fully necessary if we use Mutex, but since it's zero-contention
    // Mutex is completely fine and safe.
    pub metadata: Vec<std::sync::Mutex<Option<EvaluationRequest>>>,

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

        let mut initial_states =
            Tensor::zeros([capacity as i64, 20, 8, 16], (Kind::Float, Device::Cpu));
        let mut recurrent_actions = Tensor::zeros([capacity as i64], (Kind::Int64, Device::Cpu));
        let mut recurrent_ids = Tensor::zeros([capacity as i64], (Kind::Int64, Device::Cpu));

        if tch::Cuda::is_available() {
            initial_states = initial_states.pin_memory(Device::Cuda(0));
            recurrent_actions = recurrent_actions.pin_memory(Device::Cuda(0));
            recurrent_ids = recurrent_ids.pin_memory(Device::Cuda(0));
        }

        let mut metadata = Vec::with_capacity(capacity);
        for i in 0..capacity {
            free_tx.send(i).unwrap();
            metadata.push(std::sync::Mutex::new(None));
        }

        Arc::new(Self {
            initial_ready_tx,
            initial_ready_rx,
            recurrent_ready_tx,
            recurrent_ready_rx,
            free_tx,
            free_rx,
            initial_states_pinned: initial_states,
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

            let is_initial = req.is_initial;
            if is_initial {
                unsafe {
                    let ptr = self.initial_states_pinned.data_ptr() as *mut f32;
                    let target_slice =
                        std::slice::from_raw_parts_mut(ptr.add(slot * 20 * 8 * 16), 20 * 8 * 16);
                    crate::core::features::extract_feature_native(
                        target_slice,
                        req.board_bitmask,
                        &req.available_pieces,
                        &req.recent_board_history[..req.history_len],
                        &req.recent_action_history[..req.action_history_len],
                        req.difficulty,
                    );
                }
            } else {
                unsafe {
                    let ptr_actions = self.recurrent_actions_pinned.data_ptr() as *mut i64;
                    *ptr_actions.add(slot) = req.piece_action;

                    let ptr_ids = self.recurrent_ids_pinned.data_ptr() as *mut i64;
                    *ptr_ids.add(slot) = req.piece_id;
                }
            }

            *self.metadata[slot].lock().unwrap() = Some(req);

            if is_initial {
                let _ = self.initial_ready_tx.send(slot);
            } else {
                let _ = self.recurrent_ready_tx.send(slot);
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
    ) -> Result<(Vec<usize>, Vec<usize>), ()> {
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
                initial_batch.push(slot);
                got_something = true;
                if (initial_batch.len() + recurrent_batch.len()) == max_batch_size {
                    break;
                }
            }

            if (initial_batch.len() + recurrent_batch.len()) == max_batch_size {
                break;
            }

            while let Ok(slot) = self.recurrent_ready_rx.try_recv() {
                recurrent_batch.push(slot);
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
                std::thread::sleep(wait);
            }
        }

        if initial_batch.is_empty() && recurrent_batch.is_empty() {
            if self.active_producers.load(Ordering::SeqCst) == 0 {
                return Err(());
            }
        }

        Ok((initial_batch, recurrent_batch))
    }
}
