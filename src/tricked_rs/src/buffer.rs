use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4};
use std::sync::{Arc, RwLock};
use std::thread;
use crate::sumtree::SegmentTree;
use crate::board::GameStateExt;
use crate::features::extract_feature_native;
use bytemuck;
use zmq;

#[derive(Clone)]
pub struct EpisodeMeta {
    pub global_start_idx: usize,
    pub length: usize,
    pub difficulty: i32,
    pub score: f32,
}

pub struct SharedState {
    pub capacity: usize,
    pub unroll_steps: usize,
    pub td_steps: usize,
    pub episodes: Vec<EpisodeMeta>,
    pub current_diff: i32,
    pub global_write_idx: usize,
    pub num_states: usize,
    
    pub boards: Vec<[u64; 2]>,
    pub available: Vec<[i32; 3]>,
    pub actions: Vec<i64>,
    pub piece_ids: Vec<i64>,
    pub rewards: Vec<f32>,
    pub policies: Vec<[f32; 288]>,
    pub values: Vec<f32>,
    
    pub state_start: Vec<i64>,
    pub state_diff: Vec<i32>,
    pub state_len: Vec<i32>,
    
    pub per: crate::sumtree::PrioritizedReplay,
    
    pub recent_scores: Vec<f32>,
    pub completed_games: usize,
}

#[pyclass]
pub struct NativeReplayBuffer {
    state: Arc<RwLock<SharedState>>,
}

#[pymethods]
impl NativeReplayBuffer {
    #[new]
    #[pyo3(signature = (capacity, unroll_steps, td_steps, pull_port=None))]
    pub fn new(capacity: usize, unroll_steps: usize, td_steps: usize, pull_port: Option<String>) -> Self {
        let pull_port = pull_port.unwrap_or_else(|| "tcp://127.0.0.1:5556".to_string());
        let state = SharedState {
            capacity,
            unroll_steps,
            td_steps,
            episodes: Vec::new(),
            current_diff: 1,
            global_write_idx: 0,
            num_states: 0,
            
            boards: vec![[0, 0]; capacity],
            available: vec![[0, 0, 0]; capacity],
            actions: vec![0; capacity],
            piece_ids: vec![0; capacity],
            rewards: vec![0.0; capacity],
            policies: vec![[0.0; 288]; capacity],
            values: vec![0.0; capacity],
            
            state_start: vec![-1; capacity],
            state_diff: vec![0; capacity],
            state_len: vec![0; capacity],
            
            per: crate::sumtree::PrioritizedReplay::new(capacity, 0.6, 0.4),
            
            recent_scores: Vec::new(),
            completed_games: 0,
        };
        
        let state_arc = Arc::new(RwLock::new(state));
        
        let state_clone = Arc::clone(&state_arc);
        thread::spawn(move || {
            zmq_listener_loop(state_clone, pull_port);
        });
        
        Self { state: state_arc }
    }
    
    pub fn get_length(&self) -> usize {
        self.state.read().unwrap().num_states
    }
    
    pub fn get_global_write_idx(&self) -> usize {
        self.state.read().unwrap().global_write_idx
    }
    
    pub fn get_and_clear_metrics(&self) -> (Vec<f32>, f32, f32, f32) {
        let mut state = self.state.write().unwrap();
        if state.recent_scores.is_empty() {
            return (vec![], 0.0, 0.0, 0.0);
        }
        let scores = std::mem::take(&mut state.recent_scores);
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = sorted[sorted.len() / 2];
        let max = *sorted.last().unwrap();
        let sum: f32 = scores.iter().sum();
        let avg = sum / scores.len() as f32;
        (scores, med, max, avg)
    }

    pub fn sample_batch<'py>(&self, py: Python<'py>, batch_size: usize) -> Option<(
        Bound<'py, PyArray3<f32>>, // states: [B, 20, 96]
        Bound<'py, PyArray2<i64>>, // acts: [B, steps]
        Bound<'py, PyArray2<i64>>, // pids: [B, steps]
        Bound<'py, PyArray2<f32>>, // rews: [B, steps]
        Bound<'py, PyArray3<f32>>, // t_pols: [B, steps+1, 288]
        Bound<'py, PyArray2<f32>>, // t_vals: [B, steps+1]
        Bound<'py, PyArray2<f32>>, // m_vals: [B, steps+1]
        Bound<'py, PyArray4<f32>>, // t_states: [B, steps, 20, 96]
        Bound<'py, PyArray2<f32>>, // masks: [B, steps+1]
        Vec<(usize, usize)>, // indices: [(0, global_idx)]
        Bound<'py, PyArray1<f32>>, // weights: [B]
    )> {
        let state = self.state.read().unwrap();
        
        let (samples, weights) = match state.per.sample(batch_size, state.capacity) {
            Some((s, w)) => (s, w),
            None => return None,
        };
        
        let unroll = state.unroll_steps;
        
        let mut b_states = ndarray::Array3::<f32>::zeros((batch_size, 20, 96));
        let mut b_acts = ndarray::Array2::<i64>::zeros((batch_size, unroll));
        let mut b_pids = ndarray::Array2::<i64>::zeros((batch_size, unroll));
        let mut b_rews = ndarray::Array2::<f32>::zeros((batch_size, unroll));
        let mut b_t_pols = ndarray::Array3::<f32>::zeros((batch_size, unroll + 1, 288));
        let mut b_t_vals = ndarray::Array2::<f32>::zeros((batch_size, unroll + 1));
        let mut b_m_vals = ndarray::Array2::<f32>::zeros((batch_size, unroll + 1));
        let mut b_t_states = ndarray::Array4::<f32>::zeros((batch_size, unroll, 20, 96));
        let mut b_masks = ndarray::Array2::<f32>::zeros((batch_size, unroll + 1));
        let mut b_weights = ndarray::Array1::<f32>::zeros(batch_size);
        let mut indices = Vec::with_capacity(batch_size);
        
        for (b, &(circ_idx, _)) in samples.iter().enumerate() {
            b_weights[b] = weights[b];
            
            let st = state.state_start[circ_idx];
            let ln = state.state_len[circ_idx];
            let global_state_index = if st != -1 {
                let offset = (circ_idx as i64 - st).rem_euclid(state.capacity as i64);
                if offset < ln as i64 {
                    (st + offset) as usize
                } else {
                    st as usize
                }
            } else {
                0
            };
            
            indices.push((0, global_state_index));
            
            let start_global = state.state_start[global_state_index % state.capacity];
            let ep_end_global = if start_global != -1 {
                (start_global + state.state_len[global_state_index % state.capacity] as i64) as usize
            } else {
                global_state_index + 1
            };
            
            let feat = get_features(&state, global_state_index);
            for i in 0..20 {
                for j in 0..96 {
                    b_states[[b, i, j]] = feat[i * 96 + j];
                }
            }
            
            for offset in 0..=(unroll) {
                let curr_global = global_state_index + offset;
                let curr_idx = curr_global % state.capacity;
                
                if curr_global < ep_end_global {
                    b_masks[[b, offset]] = 1.0;
                    if offset > 0 {
                        let prev_idx = (curr_global - 1) % state.capacity;
                        b_acts[[b, offset - 1]] = state.actions[prev_idx];
                        b_pids[[b, offset - 1]] = state.piece_ids[prev_idx];
                        b_rews[[b, offset - 1]] = state.rewards[prev_idx];
                        
                        let t_feat = get_features(&state, curr_global);
                        for i in 0..20 {
                            for j in 0..96 {
                                b_t_states[[b, offset - 1, i, j]] = t_feat[i * 96 + j];
                            }
                        }
                    }
                    
                    for j in 0..288 {
                        b_t_pols[[b, offset, j]] = state.policies[curr_idx][j];
                    }
                    b_m_vals[[b, offset]] = state.values[curr_idx];
                    
                    let bootstrap_global = curr_global + state.td_steps;
                    let gamma = 0.99f32;
                    let mut val = 0.0;
                    let limit = bootstrap_global.min(ep_end_global);
                    
                    for i in 0..(limit - curr_global) {
                        let r_idx = (curr_global + i) % state.capacity;
                        val += state.rewards[r_idx] * gamma.powi(i as i32);
                    }
                    
                    if bootstrap_global < ep_end_global {
                        val += state.values[bootstrap_global % state.capacity] * gamma.powi(state.td_steps as i32);
                    }
                    b_t_vals[[b, offset]] = val;
                } else {
                    b_masks[[b, offset]] = 0.0;
                    b_t_vals[[b, offset]] = 0.0;
                    b_m_vals[[b, offset]] = 0.0;
                    for j in 0..288 {
                        b_t_pols[[b, offset, j]] = 1.0 / 288.0;
                    }
                }
            }
        }
        
        Some((
            b_states.into_pyarray_bound(py),
            b_acts.into_pyarray_bound(py),
            b_pids.into_pyarray_bound(py),
            b_rews.into_pyarray_bound(py),
            b_t_pols.into_pyarray_bound(py),
            b_t_vals.into_pyarray_bound(py),
            b_m_vals.into_pyarray_bound(py),
            b_t_states.into_pyarray_bound(py),
            b_masks.into_pyarray_bound(py),
            indices,
            b_weights.into_pyarray_bound(py),
        ))
    }
    
    pub fn update_priorities(&self, indices: Vec<usize>, priorities: Vec<f64>) {
        let mut state = self.state.write().unwrap();
        let mut circ_indices = Vec::with_capacity(indices.len());
        let mut diff_penalties = Vec::with_capacity(indices.len());
        
        for &global_state_idx in &indices {
            let circ_idx = global_state_idx % state.capacity;
            let st = state.state_start[circ_idx];
            if st != -1 {
                let diff = state.state_diff[circ_idx];
                diff_penalties.push(10f64.powf(-(state.current_diff - diff).abs() as f64));
                circ_indices.push(circ_idx);
            } else {
                diff_penalties.push(0.0); // Dummy, shouldn't hit
                circ_indices.push(circ_idx);
            }
        }
        
        state.per.update_priorities(&circ_indices, &diff_penalties, &priorities);
    }
    
    pub fn start_reanalyzer(&self, sub_port: String) {
        let state_clone = Arc::clone(&self.state);
        thread::spawn(move || {
            let device = tch::Device::Cuda(0);
            let mut model: Option<tch::CModule> = None;
            
            let ctx = zmq::Context::new();
            let subscriber = ctx.socket(zmq::SUB).unwrap();
            subscriber.connect(&sub_port).unwrap();
            subscriber.set_subscribe(b"").unwrap();
            
            loop {
                if let Ok(msg) = subscriber.recv_bytes(zmq::DONTWAIT) {
                    if let Ok(new_model) = tch::CModule::load_data_on_device(&mut msg.as_slice(), device) {
                        model = Some(new_model);
                    }
                }
                
                let m = match model {
                    Some(ref m) => m,
                    None => {
                        thread::sleep(std::time::Duration::from_secs(1));
                        continue;
                    }
                };
                
                let (cap, curr_idx) = {
                    let st = state_clone.read().unwrap();
                    (st.capacity, st.global_write_idx)
                };
                
                if curr_idx < 1000 {
                    thread::sleep(std::time::Duration::from_secs(1));
                    continue;
                }
                
                let head_circ = curr_idx % cap;
                let mut safe_idxs = Vec::new();
                let batch_size = 256;
                let mut rng = rand::thread_rng();
                use rand::Rng;
                
                for _ in 0..(batch_size * 5) {
                    let idx = rng.gen_range(0..curr_idx);
                    let idx_circ = idx % cap;
                    let dist = (idx_circ as i64 - head_circ as i64).abs();
                    let dist = dist.min(cap as i64 - dist);
                    if dist > 100 {
                        safe_idxs.push(idx);
                    }
                    if safe_idxs.len() == batch_size {
                        break;
                    }
                }
                
                if safe_idxs.len() < batch_size {
                    thread::sleep(std::time::Duration::from_millis(100));
                    continue;
                }
                
                let mut s_tensors = Vec::new();
                for &idx in &safe_idxs {
                    let st = state_clone.read().unwrap();
                    let feat = get_features(&st, idx);
                    s_tensors.push(tch::Tensor::from_slice(&feat).view([20, 96]));
                }
                
                let s_batch = tch::Tensor::stack(&s_tensors, 0).to_device(device);
                tch::no_grad(|| {
                    if let Ok(outputs) = m.method_is("initial_inference_jit", &[tch::IValue::from(s_batch)]) {
                        let tuple = if let tch::IValue::Tuple(t) = outputs { t } else { return; };
                        let value_batch = if let tch::IValue::Tensor(t) = &tuple[1] { t.copy() } else { return; }.to_device(tch::Device::Cpu).contiguous();
                        let policy_batch = if let tch::IValue::Tensor(t) = &tuple[2] { t.copy() } else { return; };
                        let policy_probs = policy_batch.softmax(-1, tch::Kind::Float).to_device(tch::Device::Cpu).contiguous();
                        
                        let value_f32: Vec<f32> = value_batch.view([-1]).try_into().unwrap_or_default();
                        let policy_f32: Vec<f32> = policy_probs.view([-1]).try_into().unwrap_or_default();
                        
                        if value_f32.len() == safe_idxs.len() {
                            let mut st = state_clone.write().unwrap();
                            for (i, &idx) in safe_idxs.iter().enumerate() {
                                if st.global_write_idx > idx + st.capacity {
                                    continue; // This state was overwritten in the ring buffer while we were evaluating!
                                }
                                let circ_idx = idx % st.capacity;
                                st.values[circ_idx] = 0.5 * st.values[circ_idx] + 0.5 * value_f32[i];
                                for j in 0..288 {
                                    st.policies[circ_idx][j] = 0.5 * st.policies[circ_idx][j] + 0.5 * policy_f32[i * 288 + j];
                                }
                            }
                        }
                    }
                });
                
                thread::sleep(std::time::Duration::from_millis(50));
            }
        });
    }
}

fn get_features(state: &SharedState, global_idx: usize) -> Vec<f32> {
    let idx = global_idx % state.capacity;
    let start_global = state.state_start[idx];
    if start_global == -1 {
        return vec![0.0; 20 * 96];
    }
    
    let diff = state.state_diff[idx];
    let board = ((state.boards[idx][1] as u128) << 64) | (state.boards[idx][0] as u128);
    let mut avail = vec![];
    for &a in &state.available[idx] {
        avail.push(a);
    }
    
    let gstate = GameStateExt::new(Some(avail.clone()), board, 0, diff, avail.iter().filter(|&&x| x != -1).count() as i32);
    
    let mut history = vec![];
    for i in 1..=8 {
        if global_idx >= i {
            let prev_global = global_idx - i;
            if prev_global >= start_global as usize {
                let p_idx = prev_global % state.capacity;
                history.push(((state.boards[p_idx][1] as u128) << 64) | (state.boards[p_idx][0] as u128));
            } else {
                break;
            }
        } else {
            break;
        }
    }
    history.reverse();
    
    let mut actions = vec![];
    for i in 1..=4 {
        if global_idx >= i {
            let prev_global = global_idx - i;
            if prev_global >= start_global as usize {
                let p_idx = prev_global % state.capacity;
                actions.push(state.actions[p_idx] as i32);
            } else {
                break;
            }
        } else {
            break;
        }
    }
    actions.reverse();
    
    extract_feature_native(&gstate, Some(history), Some(actions), diff)
}

fn zmq_listener_loop(state_arc: Arc<RwLock<SharedState>>, pull_port: String) {
    let ctx = zmq::Context::new();
    let puller = ctx.socket(zmq::PULL).unwrap();
    puller.bind(&pull_port).unwrap();
    
    let mut message = zmq::Message::new();
    loop {
        if puller.recv(&mut message, 0).is_ok() {
            let payload = message.as_ref();
            if payload.len() < 72 { continue; }
            
            let diff = i32::from_le_bytes(payload[0..4].try_into().unwrap());
            let score = f32::from_le_bytes(payload[4..8].try_into().unwrap());
            let length = u64::from_le_bytes(payload[8..16].try_into().unwrap()) as usize;
            
            if length == 0 { continue; }
            
            let b_len = u64::from_le_bytes(payload[16..24].try_into().unwrap()) as usize;
            let av_len = u64::from_le_bytes(payload[24..32].try_into().unwrap()) as usize;
            let a_len = u64::from_le_bytes(payload[32..40].try_into().unwrap()) as usize;
            let pid_len = u64::from_le_bytes(payload[40..48].try_into().unwrap()) as usize;
            let r_len = u64::from_le_bytes(payload[48..56].try_into().unwrap()) as usize;
            let pol_len = u64::from_le_bytes(payload[56..64].try_into().unwrap()) as usize;
            let v_len = u64::from_le_bytes(payload[64..72].try_into().unwrap()) as usize;
            
            let mut offset = 72;
            let b_slice = &payload[offset..offset + b_len]; offset += b_len;
            let av_slice = &payload[offset..offset + av_len]; offset += av_len;
            let a_slice = &payload[offset..offset + a_len]; offset += a_len;
            let pid_slice = &payload[offset..offset + pid_len]; offset += pid_len;
            let r_slice = &payload[offset..offset + r_len]; offset += r_len;
            let pol_slice = &payload[offset..offset + pol_len]; offset += pol_len;
            let v_slice = &payload[offset..offset + v_len];
            
            let b_np: &[u64] = bytemuck::cast_slice(b_slice);
            let av_np: &[i32] = bytemuck::cast_slice(av_slice);
            let a_np: &[i64] = bytemuck::cast_slice(a_slice);
            let pid_np: &[i64] = bytemuck::cast_slice(pid_slice);
            let r_np: &[f32] = bytemuck::cast_slice(r_slice);
            let pol_np: &[f32] = bytemuck::cast_slice(pol_slice);
            let v_np: &[f32] = bytemuck::cast_slice(v_slice);
            
            let mut state = state_arc.write().unwrap();
            
            let eps_start = state.global_write_idx;
            state.global_write_idx += length;
            
            if state.episodes.is_empty() || diff != state.current_diff {
                state.current_diff = diff;
            }
            
            let diff_penalty = 10f64.powf(-(state.current_diff - diff).abs() as f64);
            
            let cap = state.capacity;
            
            for i in 0..length {
                let idx = (eps_start + i) % cap;
                state.state_start[idx] = eps_start as i64;
                state.state_diff[idx] = diff;
                state.state_len[idx] = length as i32;
                state.per.add(idx, diff_penalty);
                
                state.boards[idx][0] = b_np[i * 2];
                state.boards[idx][1] = b_np[i * 2 + 1];
                state.available[idx][0] = av_np[i * 3];
                state.available[idx][1] = av_np[i * 3 + 1];
                state.available[idx][2] = av_np[i * 3 + 2];
                state.actions[idx] = a_np[i];
                state.piece_ids[idx] = pid_np[i];
                state.rewards[idx] = r_np[i];
                for j in 0..288 {
                    state.policies[idx][j] = pol_np[i * 288 + j];
                }
                state.values[idx] = v_np[i];
            }
            
            state.episodes.push(EpisodeMeta {
                global_start_idx: eps_start,
                length,
                difficulty: diff,
                score,
            });
            state.recent_scores.push(score);
            state.completed_games += 1;
            
            let mut valid_eps = Vec::new();
            let mut num_valid_states = 0;
            let latest_global = eps_start + length;
            
            // Just copy over relevant episodes.
            // If the episode's start is within the capacity, it hasn't been overwritten.
            for ep in &state.episodes {
                if ep.global_start_idx + state.capacity >= latest_global {
                    valid_eps.push(ep.clone());
                    num_valid_states += ep.length;
                }
            }
            state.episodes = valid_eps;
            
            state.num_states = cap.min(state.num_states + length);
        }
    }
}
