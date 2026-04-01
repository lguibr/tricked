use crossbeam_channel::{bounded, Receiver, Sender};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::mcts::EvaluationRequest;

pub struct FixedInferenceQueue {
    pub sender: Sender<EvaluationRequest>,
    pub receiver: Receiver<EvaluationRequest>,
    pub active_producers: AtomicUsize,
    pub blocked_producers: AtomicUsize,
}

impl FixedInferenceQueue {
    pub fn new(_buffer_capacity_limit: usize, total_producers: usize) -> Arc<Self> {
        let (sender, receiver) = bounded(16384);
        Arc::new(Self {
            sender,
            receiver,
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
        for req in reqs {
            let _ = self.sender.send(req);
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
    ) -> Result<Vec<EvaluationRequest>, ()> {
        let mut batch = Vec::with_capacity(max_batch_size);

        if max_batch_size == 0 {
            return Ok(batch);
        }

        let start = std::time::Instant::now();

        while batch.len() < max_batch_size {
            let elapsed = start.elapsed();
            if elapsed >= timeout {
                break;
            }
            let remaining = timeout - elapsed;

            if batch.is_empty() {
                match self.receiver.recv_timeout(remaining) {
                    Ok(req) => batch.push(req),
                    Err(_) => break,
                }
            } else {
                let is_everyone_blocked = self.blocked_producers.load(Ordering::SeqCst)
                    >= self.active_producers.load(Ordering::SeqCst);

                if is_everyone_blocked {
                    while let Ok(req) = self.receiver.try_recv() {
                        batch.push(req);
                        if batch.len() == max_batch_size {
                            break;
                        }
                    }
                    break;
                }

                let wait = remaining.min(Duration::from_micros(50));
                match self.receiver.recv_timeout(wait) {
                    Ok(req) => batch.push(req),
                    Err(_) => continue,
                }
            }
        }

        if batch.is_empty() && self.active_producers.load(Ordering::SeqCst) == 0 {
            return Err(());
        }

        Ok(batch)
    }
}
