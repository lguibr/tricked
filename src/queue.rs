use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, Sender};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::mcts::EvalReq;

pub struct FixedInferenceQueue {
    pub tx: Sender<EvalReq>,
    pub rx: Receiver<EvalReq>,
    pub active_producers: AtomicUsize,
}

impl FixedInferenceQueue {
    pub fn new(_capacity: usize, total_producers: usize) -> Arc<Self> {
        let (tx, rx) = bounded(16384);
        Arc::new(Self {
            tx,
            rx,
            active_producers: AtomicUsize::new(total_producers),
        })
    }

    pub fn push(&self, _worker_id: usize, req: EvalReq) -> Result<(), ()> {
        self.tx.send(req).map_err(|_| ())
    }

    pub fn disconnect_producer(&self) {
        self.active_producers.fetch_sub(1, Ordering::SeqCst);
    }

    pub fn pop_batch_timeout(
        &self,
        max_batch_size: usize,
        timeout: Duration,
    ) -> Result<Vec<EvalReq>, ()> {
        let mut batch = Vec::with_capacity(max_batch_size);

        if max_batch_size == 0 {
            return Ok(batch);
        }

        if self.rx.is_empty() && self.active_producers.load(Ordering::SeqCst) == 0 {
            return Err(());
        }

        match self.rx.recv_timeout(timeout) {
            Ok(req) => {
                batch.push(req);
                std::thread::sleep(Duration::from_micros(1500));
            }
            Err(RecvTimeoutError::Timeout) => {
                if self.active_producers.load(Ordering::SeqCst) == 0 {
                    return Err(());
                }
                return Ok(batch);
            }
            Err(RecvTimeoutError::Disconnected) => return Err(()),
        }

        while batch.len() < max_batch_size {
            match self.rx.try_recv() {
                Ok(req) => batch.push(req),
                Err(_) => break,
            }
        }

        Ok(batch)
    }
}
