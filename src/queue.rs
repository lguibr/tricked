use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, Sender};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::mcts::EvaluationRequest;

pub struct FixedInferenceQueue {
    pub evaluation_request_transmitter: Sender<EvaluationRequest>,
    pub evaluation_response_receiver: Receiver<EvaluationRequest>,
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
    pub fn push_batch(
        &self,
        _worker_id: usize,
        reqs: impl IntoIterator<Item = EvaluationRequest>,
    ) -> Result<(), ()> {
        for req in reqs {
            if self.evaluation_request_transmitter.send(req).is_err() {
                return Err(());
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
    ) -> Result<Vec<EvaluationRequest>, ()> {
        let mut batch = Vec::with_capacity(max_batch_size);

        if max_batch_size == 0 {
            return Ok(batch);
        }

        let time_limit = std::time::Instant::now() + timeout;

        while batch.len() < max_batch_size {
            let remaining_time = time_limit.saturating_duration_since(std::time::Instant::now());
            if remaining_time.is_zero() && !batch.is_empty() {
                break;
            }

            match self
                .evaluation_response_receiver
                .recv_timeout(if batch.is_empty() {
                    timeout
                } else {
                    remaining_time
                }) {
                Ok(req) => {
                    batch.push(req);
                    while batch.len() < max_batch_size {
                        match self.evaluation_response_receiver.try_recv() {
                            Ok(r) => batch.push(r),
                            Err(_) => break,
                        }
                    }
                }
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => {
                    if batch.is_empty() {
                        return Err(());
                    }
                    break;
                }
            }
        }

        if batch.is_empty() && self.active_producers.load(Ordering::SeqCst) == 0 {
            return Err(());
        }

        Ok(batch)
    }
}
