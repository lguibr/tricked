pub struct EvaluationRequest {
    pub is_initial: bool,
    pub board_bitmask: u128,
    pub available_pieces: [i32; 3],
    pub recent_board_history: [u128; 8],
    pub history_len: usize,
    pub recent_action_history: [i32; 4],
    pub action_history_len: usize,
    pub difficulty: i32,
    pub piece_action: i64,
    pub piece_id: i64,
    pub node_index: usize,
    pub worker_id: usize,
    pub parent_cache_index: u32,
    pub leaf_cache_index: u32,
    pub evaluation_request_transmitter: crossbeam_channel::Sender<EvaluationResponse>,
}

pub struct EvaluationResponse {
    pub reward: f32,
    pub value: f32,
    pub child_prior_probabilities_tensor: [f32; 288],
    pub node_index: usize,
}

pub trait NetworkEvaluator: Send + Sync {
    fn send_batch(&self, reqs: arrayvec::ArrayVec<EvaluationRequest, 256>) -> Result<(), String>;
    fn mark_blocked(&self) {}
    fn mark_unblocked(&self) {}
}

impl NetworkEvaluator for std::sync::Arc<crate::queue::FixedInferenceQueue> {
    fn send_batch(&self, reqs: arrayvec::ArrayVec<EvaluationRequest, 256>) -> Result<(), String> {
        if reqs.is_empty() {
            return Ok(());
        }
        self.push_batch(reqs[0].worker_id, reqs)
            .map_err(|_| "Queue Disconnected".to_string())
    }

    fn mark_blocked(&self) {
        self.blocked_producers
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }

    fn mark_unblocked(&self) {
        self.blocked_producers
            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

#[cfg(test)]
pub struct MockEvaluator;

#[cfg(test)]
impl NetworkEvaluator for MockEvaluator {
    fn send_batch(&self, reqs: arrayvec::ArrayVec<EvaluationRequest, 256>) -> Result<(), String> {
        for request in reqs {
            let response = EvaluationResponse {
                reward: 0.0,
                value: 0.0,
                child_prior_probabilities_tensor: [1.0 / 288.0; 288],
                node_index: request.node_index,
            };
            let _ = request.evaluation_request_transmitter.send(response);
        }
        Ok(())
    }
}

#[cfg(test)]
pub struct CustomEvaluator {
    pub reward: f32,
    pub value: f32,
}

#[cfg(test)]
impl NetworkEvaluator for CustomEvaluator {
    fn send_batch(&self, reqs: arrayvec::ArrayVec<EvaluationRequest, 256>) -> Result<(), String> {
        for request in reqs {
            let response = EvaluationResponse {
                reward: self.reward,
                value: self.value,
                child_prior_probabilities_tensor: [1.0 / 288.0; 288],
                node_index: request.node_index,
            };
            let _ = request.evaluation_request_transmitter.send(response);
        }
        Ok(())
    }
}
