use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, Sender};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::mcts::EvalReq;

pub struct FixedInferenceQueue {
    pub evaluation_request_transmitter: Sender<EvalReq>,
    pub evaluation_response_receiver: Receiver<EvalReq>,
    pub active_producers: AtomicUsize,
}

impl FixedInferenceQueue {
    pub fn new(_buffer_capacity_limit: usize, total_producers: usize) -> Arc<Self> {
        let (evaluation_request_transmitter, evaluation_response_receiver) = bounded(16384);
        Arc::new(Self {
            evaluation_request_transmitter,
            evaluation_response_receiver,
            active_producers: AtomicUsize::new(total_producers),
        })
    }

    #[allow(clippy::result_unit_err)]
    pub fn push(&self, _worker_id: usize, req: EvalReq) -> Result<(), ()> {
        self.evaluation_request_transmitter
            .send(req)
            .map_err(|_| ())
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
    ) -> Result<Vec<EvalReq>, ()> {
        let mut batch = Vec::with_capacity(max_batch_size);

        if max_batch_size == 0 {
            return Ok(batch);
        }

        if self.evaluation_response_receiver.is_empty()
            && self.active_producers.load(Ordering::SeqCst) == 0
        {
            return Err(());
        }

        match self.evaluation_response_receiver.recv_timeout(timeout) {
            Ok(req) => {
                batch.push(req);
                while batch.len() < max_batch_size {
                    if let Ok(r) = self.evaluation_response_receiver.try_recv() {
                        batch.push(r);
                    } else {
                        break;
                    }
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                if self.active_producers.load(Ordering::SeqCst) == 0 {
                    return Err(());
                }
                return Ok(batch);
            }
            Err(RecvTimeoutError::Disconnected) => return Err(()),
        }

        Ok(batch)
    }
}
