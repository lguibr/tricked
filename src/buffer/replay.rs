use tch::{Device, Tensor};
use std::sync::Arc;
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};

use crate::buffer::state::{EpisodeMeta, SharedState, StorageArrays};
use std::sync::RwLock;

pub struct ReplayBuffer {
    pub state: Arc<SharedState>,
}

pub struct BatchTensors {
    pub b_states: Tensor,
    pub b_acts: Tensor,
    pub b_pids: Tensor,
    pub b_rews: Tensor,
    pub b_t_pols: Tensor,
    pub b_t_vals: Tensor,
    pub b_m_vals: Tensor,
    pub b_t_states: Tensor,
    pub b_masks: Tensor,
    pub b_weights: Tensor,
    pub indices: Vec<usize>,
}

impl ReplayBuffer {
    pub fn new(capacity: usize, unroll_steps: usize, td_steps: usize) -> Self {
        let state = SharedState {
            capacity,
            unroll_steps,
            td_steps,
            current_diff: AtomicI32::new(1),
            global_write_idx: AtomicUsize::new(0),
            global_write_active_idx: AtomicUsize::new(0),
            num_states: AtomicUsize::new(0),

            arrays: RwLock::new(StorageArrays::new(capacity)),
            per: crate::sumtree::ShardedPrioritizedReplay::new(capacity, 0.6, 0.4, 8),

            episodes: std::sync::Mutex::new(Vec::new()),
            recent_scores: std::sync::Mutex::new(Vec::new()),
            completed_games: AtomicUsize::new(0),
        };

        Self {
            state: Arc::new(state),
        }
    }

    pub fn get_length(&self) -> usize {
        self.state.num_states.load(Ordering::Relaxed)
    }

    pub fn get_global_write_idx(&self) -> usize {
        self.state.global_write_idx.load(Ordering::Acquire)
    }

    pub fn get_and_clear_metrics(&self) -> (Vec<f32>, f32, f32, f32) {
        let mut recent = match self.state.recent_scores.lock() {
            Ok(lock) => lock,
            Err(e) => e.into_inner(),
        };
        if recent.is_empty() {
            return (vec![], 0.0, 0.0, 0.0);
        }
        let scores = std::mem::take(&mut *recent);
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = sorted[sorted.len() / 2];
        let max = *sorted.last().unwrap_or(&0.0);
        let sum: f32 = scores.iter().sum();
        let avg = sum / scores.len() as f32;
        (scores, med, max, avg)
    }

    pub fn add_game(
        &self,
        diff: i32,
        score: f32,
        b_np: &[[u64; 2]],
        av_np: &[[i32; 3]],
        a_np: &[i64],
        pid_np: &[i64],
        r_np: &[f32],
        pol_np: &[[f32; 288]],
        v_np: &[f32],
    ) {
        let length = b_np.len();
        if length == 0 {
            return;
        }

        let eps_start = self.state.global_write_idx.load(Ordering::Relaxed);
        let cap = self.state.capacity;
        let current_diff = self.state.current_diff.load(Ordering::Relaxed);
        let next_write_idx = eps_start + length;
        self.state.global_write_active_idx.store(next_write_idx, Ordering::Release);

        if current_diff == 0 || diff != current_diff {
            self.state.current_diff.store(diff, Ordering::Relaxed);
        }
        let current_diff = self.state.current_diff.load(Ordering::Relaxed);
        let diff_penalty = 10f64.powf(-(current_diff - diff).abs() as f64);

        {
            let mut arr = match self.state.arrays.write() {
                Ok(lock) => lock,
                Err(e) => e.into_inner(),
            };

            for i in 0..length {
                let idx = (eps_start + i) % cap;
                arr.state_start[idx] = eps_start as i64;
                arr.state_diff[idx] = diff;
                arr.state_len[idx] = length as i32;

                arr.boards[idx] = b_np[i];
                arr.available[idx] = av_np[i];
                arr.actions[idx] = a_np[i];
                arr.piece_ids[idx] = pid_np[i];
                arr.rewards[idx] = r_np[i];
                arr.policies[idx] = pol_np[i];
                arr.values[idx] = v_np[i];
            }
        }

        let mut add_indices = Vec::with_capacity(length);
        let mut add_diff_penalties = Vec::with_capacity(length);
        for i in 0..length {
            add_indices.push((eps_start + i) % cap);
            add_diff_penalties.push(diff_penalty);
        }
        self.state.per.add_batch(&add_indices, &add_diff_penalties);

        self.state.global_write_idx.store(next_write_idx, Ordering::Release);

        {
            let mut eps = match self.state.episodes.lock() {
                Ok(lock) => lock,
                Err(e) => e.into_inner(),
            };
            eps.push(EpisodeMeta {
                global_start_idx: eps_start,
                length,
                difficulty: diff,
                score,
            });

            let mut valid_eps = Vec::new();
            for ep in eps.iter() {
                if ep.global_start_idx + cap >= next_write_idx {
                    valid_eps.push(ep.clone());
                }
            }
            *eps = valid_eps;
        }

        let mut recent = match self.state.recent_scores.lock() {
            Ok(lock) => lock,
            Err(e) => e.into_inner(),
        };
        recent.push(score);
        self.state.completed_games.fetch_add(1, Ordering::Relaxed);

        let cur_num = self.state.num_states.load(Ordering::Relaxed);
        self.state.num_states.store(cap.min(cur_num + length), Ordering::Relaxed);
    }

    pub fn sample_batch(&self, batch_size: usize, device: Device) -> Option<BatchTensors> {
        let (samples, weights_arr) = match self.state.per.sample(batch_size, self.state.capacity) {
            Some((s, w)) => (s, w),
            None => return None,
        };

        let unroll = self.state.unroll_steps;
        let mut b_states = vec![0.0f32; batch_size * 20 * 96];
        let mut b_acts = vec![0i64; batch_size * unroll];
        let mut b_pids = vec![0i64; batch_size * unroll];
        let mut b_rews = vec![0.0f32; batch_size * unroll];
        let mut b_t_pols = vec![0.0f32; batch_size * (unroll + 1) * 288];
        let mut b_t_vals = vec![0.0f32; batch_size * (unroll + 1)];
        let mut b_m_vals = vec![0.0f32; batch_size * (unroll + 1)];
        let mut b_t_states = vec![0.0f32; batch_size * unroll * 20 * 96];
        let mut b_masks = vec![0.0f32; batch_size * (unroll + 1)];
        let mut b_weights = vec![0.0f32; batch_size];

        let mut indices = Vec::with_capacity(batch_size);

        {
            let arr = match self.state.arrays.read() {
                Ok(lock) => lock,
                Err(e) => e.into_inner(),
            };

            for (b, &(circ_idx, _)) in samples.iter().enumerate() {
                b_weights[b] = weights_arr[b] as f32;

                let st = arr.state_start[circ_idx];
                let ln = arr.state_len[circ_idx];
                let global_state_index = if st != -1 {
                    let offset = (circ_idx as i64 - st).rem_euclid(self.state.capacity as i64);
                    if offset < ln as i64 {
                        (st + offset) as usize
                    } else {
                        st as usize
                    }
                } else {
                    0
                };

                indices.push(global_state_index);

                let start_global = arr.state_start[global_state_index % self.state.capacity];
                let ep_end_global = if start_global != -1 {
                    (start_global + arr.state_len[global_state_index % self.state.capacity] as i64) as usize
                } else {
                    global_state_index + 1
                };

                let safe_before = self.state.global_write_idx.load(Ordering::Acquire);

                drop(arr);
                let feat = self.state.get_features(global_state_index);
                let arr = match self.state.arrays.read() {
                    Ok(lock) => lock,
                    Err(e) => e.into_inner(),
                };

                for i in 0..20 {
                    for j in 0..96 {
                        b_states[b * 20 * 96 + i * 96 + j] = feat[i * 96 + j];
                    }
                }

                for offset in 0..=(unroll) {
                    let curr_global = global_state_index + offset;
                    let curr_idx = curr_global % self.state.capacity;

                    if curr_global < ep_end_global {
                        b_masks[b * (unroll + 1) + offset] = 1.0;
                        if offset > 0 {
                            let prev_idx = (curr_global - 1) % self.state.capacity;
                            b_acts[b * unroll + offset - 1] = arr.actions[prev_idx];
                            b_pids[b * unroll + offset - 1] = arr.piece_ids[prev_idx];
                            b_rews[b * unroll + offset - 1] = arr.rewards[prev_idx];

                            drop(arr);
                            let t_feat = self.state.get_features(curr_global);
                            let arr_read = match self.state.arrays.read() {
                                Ok(lock) => lock,
                                Err(e) => e.into_inner(),
                            };
                            for i in 0..20 {
                                for j in 0..96 {
                                    b_t_states[(b * unroll + offset - 1) * 20 * 96 + i * 96 + j] = t_feat[i * 96 + j];
                                }
                            }
                            // Rebound arr since drop(arr) occurred
                            #[allow(unused_variables)]
                            let _ = arr_read;
                        }
                        
                        let arr = match self.state.arrays.read() {
                            Ok(lock) => lock,
                            Err(e) => e.into_inner(),
                        };

                        let policy_slice = arr.policies[curr_idx];
                        for j in 0..288 {
                            b_t_pols[b * (unroll + 1) * 288 + offset * 288 + j] = policy_slice[j];
                        }
                        b_m_vals[b * (unroll + 1) + offset] = arr.values[curr_idx];

                        let bootstrap_global = curr_global + self.state.td_steps;
                        let gamma = 0.99f32;
                        let mut val = 0.0;

                        let limit = bootstrap_global.min(ep_end_global);

                        for i in 0..(limit - curr_global) {
                            let r_idx = (curr_global + i) % self.state.capacity;
                            val += arr.rewards[r_idx] * gamma.powi(i as i32);
                        }

                        if bootstrap_global < ep_end_global {
                            val += arr.values[bootstrap_global % self.state.capacity]
                                * gamma.powi(self.state.td_steps as i32);
                        }
                        b_t_vals[b * (unroll + 1) + offset] = val;
                    } else {
                        b_masks[b * (unroll + 1) + offset] = 0.0;
                        b_t_vals[b * (unroll + 1) + offset] = 0.0;
                        b_m_vals[b * (unroll + 1) + offset] = 0.0;
                        for j in 0..288 {
                            b_t_pols[b * (unroll + 1) * 288 + offset * 288 + j] = 1.0 / 288.0;
                        }
                    }
                }

                let active_after = self.state.global_write_active_idx.load(Ordering::Acquire);
                let max_global_read = global_state_index + unroll + self.state.td_steps;
                let min_global_read = global_state_index.saturating_sub(8);

                let not_fully_written = max_global_read >= safe_before;
                let overwritten = min_global_read + self.state.capacity <= active_after;

                if not_fully_written || overwritten {
                    b_weights[b] = 0.0;
                }
            }
        }

        Some(BatchTensors {
            b_states: Tensor::from_slice(&b_states).view((batch_size as i64, 20, 96)).to_device(device),
            b_acts: Tensor::from_slice(&b_acts).view((batch_size as i64, unroll as i64)).to_device(device),
            b_pids: Tensor::from_slice(&b_pids).view((batch_size as i64, unroll as i64)).to_device(device),
            b_rews: Tensor::from_slice(&b_rews).view((batch_size as i64, unroll as i64)).to_device(device).nan_to_num(0.0, Some(0.0), Some(0.0)),
            b_t_pols: Tensor::from_slice(&b_t_pols).view((batch_size as i64, (unroll + 1) as i64, 288)).to_device(device).nan_to_num(0.0, Some(0.0), Some(0.0)),
            b_t_vals: Tensor::from_slice(&b_t_vals).view((batch_size as i64, (unroll + 1) as i64)).to_device(device).nan_to_num(0.0, Some(0.0), Some(0.0)),
            b_m_vals: Tensor::from_slice(&b_m_vals).view((batch_size as i64, (unroll + 1) as i64)).to_device(device).nan_to_num(0.0, Some(0.0), Some(0.0)),
            b_t_states: Tensor::from_slice(&b_t_states).view((batch_size as i64, unroll as i64, 20, 96)).to_device(device),
            b_masks: Tensor::from_slice(&b_masks).view((batch_size as i64, (unroll + 1) as i64)).to_device(device),
            b_weights: Tensor::from_slice(&b_weights).view((batch_size as i64,)).to_device(device),
            indices,
        })
    }

    pub fn update_priorities(&self, indices: &[usize], priorities: &[f64]) {
        let current_diff = self.state.current_diff.load(Ordering::Relaxed);
        let mut circ_indices = Vec::with_capacity(indices.len());
        let mut diff_penalties = Vec::with_capacity(indices.len());

        let arr = match self.state.arrays.read() {
            Ok(lock) => lock,
            Err(e) => e.into_inner(),
        };

        for &global_state_idx in indices {
            let circ_idx = global_state_idx % self.state.capacity;
            let st = arr.state_start[circ_idx];
            if st != -1 {
                let diff = arr.state_diff[circ_idx];
                diff_penalties.push(10f64.powf(-(current_diff - diff).abs() as f64));
                circ_indices.push(circ_idx);
            } else {
                diff_penalties.push(0.0);
                circ_indices.push(circ_idx);
            }
        }

        self.state.per.update_priorities(&circ_indices, &diff_penalties, priorities);
    }
}
