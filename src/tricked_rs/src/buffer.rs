use crate::board::GameStateExt;
use crate::features::extract_feature_native;
use bytemuck;
use numpy::{PyReadwriteArray1, PyReadwriteArray2, PyReadwriteArray3, PyReadwriteArray4};
use pyo3::prelude::*;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use zmq;

#[derive(Clone)]
pub struct EpisodeMeta {
    pub global_start_idx: usize,
    pub length: usize,
    pub difficulty: i32,
    pub score: f32,
}

pub struct SharedArrays {
    pub boards: UnsafeCell<Vec<[u64; 2]>>,
    pub available: UnsafeCell<Vec<[i32; 3]>>,
    pub actions: UnsafeCell<Vec<i64>>,
    pub piece_ids: UnsafeCell<Vec<i64>>,
    pub rewards: UnsafeCell<Vec<f32>>,
    pub policies: UnsafeCell<Vec<[f32; 288]>>,
    pub values: UnsafeCell<Vec<f32>>,
    pub state_start: UnsafeCell<Vec<i64>>,
    pub state_diff: UnsafeCell<Vec<i32>>,
    pub state_len: UnsafeCell<Vec<i32>>,
}

unsafe impl Sync for SharedArrays {}

pub struct SharedState {
    pub capacity: usize,
    pub unroll_steps: usize,
    pub td_steps: usize,

    pub current_diff: AtomicI32,
    pub global_write_idx: AtomicUsize,
    pub global_write_active_idx: AtomicUsize,
    pub num_states: AtomicUsize,

    pub arrays: SharedArrays,

    pub per: crate::sumtree::ShardedPrioritizedReplay,

    pub episodes: Mutex<Vec<EpisodeMeta>>,
    pub recent_scores: Mutex<Vec<f32>>,
    pub completed_games: AtomicUsize,
}

impl SharedState {
    pub fn get_features(&self, global_idx: usize) -> Vec<f32> {
        let idx = global_idx % self.capacity;

        let start_global = unsafe { (&*self.arrays.state_start.get())[idx] };
        if start_global == -1 {
            return vec![0.0; 20 * 96];
        }

        let diff = unsafe { (&*self.arrays.state_diff.get())[idx] };
        let b = unsafe { (&*self.arrays.boards.get())[idx] };
        let board = ((b[1] as u128) << 64) | (b[0] as u128);

        let av = unsafe { (&*self.arrays.available.get())[idx] };
        let avail = vec![av[0], av[1], av[2]];

        let gstate = GameStateExt::new(
            Some(avail.clone()),
            board,
            0,
            diff,
            avail.iter().filter(|&&x| x != -1).count() as i32,
        );

        let mut history = vec![];
        for i in 1..=8 {
            if global_idx >= i {
                let prev_global = global_idx - i;
                if prev_global >= start_global as usize {
                    let p_idx = prev_global % self.capacity;
                    let pb = unsafe { (&*self.arrays.boards.get())[p_idx] };
                    history.push(((pb[1] as u128) << 64) | (pb[0] as u128));
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
                    let p_idx = prev_global % self.capacity;
                    actions.push(unsafe { (&*self.arrays.actions.get())[p_idx] } as i32);
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
}

#[pyclass]
pub struct NativeReplayBuffer {
    state: Arc<SharedState>,
}

#[pymethods]
impl NativeReplayBuffer {
    #[new]
    #[pyo3(signature = (capacity, unroll_steps, td_steps, pull_port=None))]
    pub fn new(
        capacity: usize,
        unroll_steps: usize,
        td_steps: usize,
        pull_port: Option<String>,
    ) -> Self {
        let pull_port = pull_port.unwrap_or_else(|| "tcp://127.0.0.1:5556".to_string());
        let state = SharedState {
            capacity,
            unroll_steps,
            td_steps,
            current_diff: AtomicI32::new(1),
            global_write_idx: AtomicUsize::new(0),
            global_write_active_idx: AtomicUsize::new(0),
            num_states: AtomicUsize::new(0),

            arrays: SharedArrays {
                boards: UnsafeCell::new(vec![[0, 0]; capacity]),
                available: UnsafeCell::new(vec![[0, 0, 0]; capacity]),
                actions: UnsafeCell::new(vec![0; capacity]),
                piece_ids: UnsafeCell::new(vec![0; capacity]),
                rewards: UnsafeCell::new(vec![0.0; capacity]),
                policies: UnsafeCell::new(vec![[0.0; 288]; capacity]),
                values: UnsafeCell::new(vec![0.0; capacity]),

                state_start: UnsafeCell::new(vec![-1; capacity]),
                state_diff: UnsafeCell::new(vec![0; capacity]),
                state_len: UnsafeCell::new(vec![0; capacity]),
            },

            per: crate::sumtree::ShardedPrioritizedReplay::new(capacity, 0.6, 0.4, 8),

            episodes: Mutex::new(Vec::new()),
            recent_scores: Mutex::new(Vec::new()),
            completed_games: AtomicUsize::new(0),
        };

        let state_arc = Arc::new(state);
        let state_clone = Arc::clone(&state_arc);

        thread::spawn(move || {
            zmq_listener_loop(state_clone, pull_port);
        });

        Self { state: state_arc }
    }

    pub fn get_length(&self) -> usize {
        self.state.num_states.load(Ordering::Relaxed)
    }

    pub fn get_global_write_idx(&self) -> usize {
        self.state.global_write_idx.load(Ordering::Acquire)
    }

    pub fn get_and_clear_metrics(&self) -> (Vec<f32>, f32, f32, f32) {
        let mut recent = self.state.recent_scores.lock().unwrap();
        if recent.is_empty() {
            return (vec![], 0.0, 0.0, 0.0);
        }
        let scores = std::mem::take(&mut *recent);
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = sorted[sorted.len() / 2];
        let max = *sorted.last().unwrap();
        let sum: f32 = scores.iter().sum();
        let avg = sum / scores.len() as f32;
        (scores, med, max, avg)
    }

    #[pyo3(signature = (batch_size, b_states, b_acts, b_pids, b_rews, b_t_pols, b_t_vals, b_m_vals, b_t_states, b_masks, b_weights))]
    pub fn sample_batch<'py>(
        &self,
        py: Python<'py>,
        batch_size: usize,
        mut b_states: PyReadwriteArray3<f32>,
        mut b_acts: PyReadwriteArray2<i64>,
        mut b_pids: PyReadwriteArray2<i64>,
        mut b_rews: PyReadwriteArray2<f32>,
        mut b_t_pols: PyReadwriteArray3<f32>,
        mut b_t_vals: PyReadwriteArray2<f32>,
        mut b_m_vals: PyReadwriteArray2<f32>,
        mut b_t_states: PyReadwriteArray4<f32>,
        mut b_masks: PyReadwriteArray2<f32>,
        mut b_weights: PyReadwriteArray1<f32>,
    ) -> Option<Vec<(usize, usize)>> {
        let (samples, weights) = {
            match self.state.per.sample(batch_size, self.state.capacity) {
                Some((s, w)) => (s, w),
                None => return None,
            }
        };

        let unroll = self.state.unroll_steps;

        let mut out_states = b_states.as_array_mut();
        let mut out_acts = b_acts.as_array_mut();
        let mut out_pids = b_pids.as_array_mut();
        let mut out_rews = b_rews.as_array_mut();
        let mut out_t_pols = b_t_pols.as_array_mut();
        let mut out_t_vals = b_t_vals.as_array_mut();
        let mut out_m_vals = b_m_vals.as_array_mut();
        let mut out_t_states = b_t_states.as_array_mut();
        let mut out_masks = b_masks.as_array_mut();
        let mut out_weights = b_weights.as_array_mut();

        let mut indices = Vec::with_capacity(batch_size);

        py.allow_threads(|| {
            for (b, &(circ_idx, _)) in samples.iter().enumerate() {
                out_weights[b] = weights[b];

                let st = unsafe { (&*self.state.arrays.state_start.get())[circ_idx] };
                let ln = unsafe { (&*self.state.arrays.state_len.get())[circ_idx] };
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

                // Add index, skipping Python reference inside background
                indices.push((0, global_state_index));

                let start_global = unsafe {
                    (&*self.state.arrays.state_start.get())
                        [global_state_index % self.state.capacity]
                };
                let ep_end_global = if start_global != -1 {
                    (start_global
                        + unsafe {
                            (&*self.state.arrays.state_len.get())
                                [global_state_index % self.state.capacity]
                        } as i64) as usize
                } else {
                    global_state_index + 1
                };

                let safe_before = self.state.global_write_idx.load(Ordering::Acquire);

                let feat = self.state.get_features(global_state_index);
                for i in 0..20 {
                    for j in 0..96 {
                        out_states[[b, i, j]] = feat[i * 96 + j];
                    }
                }

                for offset in 0..=(unroll) {
                    let curr_global = global_state_index + offset;
                    let curr_idx = curr_global % self.state.capacity;

                    if curr_global < ep_end_global {
                        out_masks[[b, offset]] = 1.0;
                        if offset > 0 {
                            let prev_idx = (curr_global - 1) % self.state.capacity;
                            out_acts[[b, offset - 1]] =
                                unsafe { (&*self.state.arrays.actions.get())[prev_idx] };
                            out_pids[[b, offset - 1]] =
                                unsafe { (&*self.state.arrays.piece_ids.get())[prev_idx] };
                            out_rews[[b, offset - 1]] =
                                unsafe { (&*self.state.arrays.rewards.get())[prev_idx] };

                            let t_feat = self.state.get_features(curr_global);
                            for i in 0..20 {
                                for j in 0..96 {
                                    out_t_states[[b, offset - 1, i, j]] = t_feat[i * 96 + j];
                                }
                            }
                        }

                        let policy_slice =
                            unsafe { (&*self.state.arrays.policies.get())[curr_idx] };
                        for j in 0..288 {
                            out_t_pols[[b, offset, j]] = policy_slice[j];
                        }
                        out_m_vals[[b, offset]] =
                            unsafe { (&*self.state.arrays.values.get())[curr_idx] };

                        let bootstrap_global = curr_global + self.state.td_steps;
                        let gamma = 0.99f32;
                        let mut val = 0.0;

                        let limit = bootstrap_global.min(ep_end_global);

                        for i in 0..(limit - curr_global) {
                            let r_idx = (curr_global + i) % self.state.capacity;
                            val += unsafe { (&*self.state.arrays.rewards.get())[r_idx] }
                                * gamma.powi(i as i32);
                        }

                        if bootstrap_global < ep_end_global {
                            val += unsafe {
                                (&*self.state.arrays.values.get())
                                    [bootstrap_global % self.state.capacity]
                            } * gamma.powi(self.state.td_steps as i32);
                        }
                        out_t_vals[[b, offset]] = val;
                    } else {
                        out_masks[[b, offset]] = 0.0;
                        out_t_vals[[b, offset]] = 0.0;
                        out_m_vals[[b, offset]] = 0.0;
                        for j in 0..288 {
                            out_t_pols[[b, offset, j]] = 1.0 / 288.0;
                        }
                    }
                }

                let active_after = self.state.global_write_active_idx.load(Ordering::Acquire);
                let max_global_read = global_state_index + unroll + self.state.td_steps;
                let min_global_read = global_state_index.saturating_sub(8); // get_features reads 8 states of history

                let not_fully_written = max_global_read >= safe_before;
                let overwritten = min_global_read + self.state.capacity <= active_after;

                if not_fully_written || overwritten {
                    out_weights[b] = 0.0; // Mask out the sample, it's torn
                }
            }
        });

        Some(indices)
    }

    pub fn update_priorities(&self, indices: Vec<usize>, priorities: Vec<f64>) {
        let current_diff = self.state.current_diff.load(Ordering::Relaxed);
        let mut circ_indices = Vec::with_capacity(indices.len());
        let mut diff_penalties = Vec::with_capacity(indices.len());

        for &global_state_idx in &indices {
            let circ_idx = global_state_idx % self.state.capacity;
            let st = unsafe { (&*self.state.arrays.state_start.get())[circ_idx] };
            if st != -1 {
                let diff = unsafe { (&*self.state.arrays.state_diff.get())[circ_idx] };
                diff_penalties.push(10f64.powf(-(current_diff - diff).abs() as f64));
                circ_indices.push(circ_idx);
            } else {
                diff_penalties.push(0.0);
                circ_indices.push(circ_idx);
            }
        }

        self.state
            .per
            .update_priorities(&circ_indices, &diff_penalties, &priorities);
    }

    pub fn start_reanalyzer(&self, redis_url: String) {
        let state_clone = Arc::clone(&self.state);
        thread::spawn(move || {
            let device = tch::Device::Cuda(0);
            let mut model: Option<tch::CModule> = None;

            let client = redis::Client::open(redis_url).unwrap();
            let mut con = client.get_connection().unwrap();
            let mut pubsub = con.as_pubsub();
            let _ = pubsub.subscribe("model_updates");
            let mut dl_con = client.get_connection().unwrap();

            if let Ok(bytes) = redis::cmd("GET")
                .arg("model_weights")
                .query::<Vec<u8>>(&mut dl_con)
            {
                if let Ok(new_model) =
                    tch::CModule::load_data_on_device(&mut bytes.as_slice(), device)
                {
                    model = Some(new_model);
                }
            }

            loop {
                if let Ok(_) = pubsub.get_message() {
                    if let Ok(bytes) = redis::cmd("GET")
                        .arg("model_weights")
                        .query::<Vec<u8>>(&mut dl_con)
                    {
                        if let Ok(new_model) =
                            tch::CModule::load_data_on_device(&mut bytes.as_slice(), device)
                        {
                            model = Some(new_model);
                        }
                    }
                }

                let m = match model {
                    Some(ref m) => m,
                    None => {
                        thread::sleep(std::time::Duration::from_secs(1));
                        continue;
                    }
                };

                let cap = state_clone.capacity;
                let curr_idx = state_clone.global_write_idx.load(Ordering::Acquire);

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
                    let feat = state_clone.get_features(idx);
                    s_tensors.push(tch::Tensor::from_slice(&feat).view([20, 96]));
                }

                let s_batch = tch::Tensor::stack(&s_tensors, 0)
                    .to_device(device)
                    .to_kind(tch::Kind::Half);
                tch::no_grad(|| {
                    if let Ok(outputs) =
                        m.method_is("initial_inference_jit", &[tch::IValue::from(s_batch)])
                    {
                        let tuple = if let tch::IValue::Tuple(t) = outputs {
                            t
                        } else {
                            return;
                        };
                        let value_batch = if let tch::IValue::Tensor(t) = &tuple[1] {
                            t.copy()
                        } else {
                            return;
                        }
                        .to_kind(tch::Kind::Float)
                        .to_device(tch::Device::Cpu)
                        .contiguous();
                        let policy_batch = if let tch::IValue::Tensor(t) = &tuple[2] {
                            t.copy()
                        } else {
                            return;
                        }
                        .to_kind(tch::Kind::Float);
                        let policy_probs = policy_batch
                            .softmax(-1, tch::Kind::Float)
                            .to_device(tch::Device::Cpu)
                            .contiguous();

                        let value_f32: Vec<f32> =
                            value_batch.view([-1]).try_into().unwrap_or_default();
                        let policy_f32: Vec<f32> =
                            policy_probs.view([-1]).try_into().unwrap_or_default();

                        if value_f32.len() == safe_idxs.len() {
                            for (i, &idx) in safe_idxs.iter().enumerate() {
                                let active_writer =
                                    state_clone.global_write_active_idx.load(Ordering::Acquire);
                                if active_writer > idx + state_clone.capacity {
                                    continue;
                                }
                                let circ_idx = idx % state_clone.capacity;
                                unsafe {
                                    (&mut *state_clone.arrays.values.get())[circ_idx] = 0.5
                                        * (&*state_clone.arrays.values.get())[circ_idx]
                                        + 0.5 * value_f32[i];
                                    for j in 0..288 {
                                        (&mut *state_clone.arrays.policies.get())[circ_idx][j] = 0.5
                                            * (&*state_clone.arrays.policies.get())[circ_idx][j]
                                            + 0.5 * policy_f32[i * 288 + j];
                                    }
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

macro_rules! safe_cast {
    ($slice:expr, $ty:ty) => {
        match bytemuck::try_cast_slice::<_, $ty>($slice) {
            Ok(s) => std::borrow::Cow::Borrowed(s),
            Err(_) => {
                let mut v = vec![0 as $ty; $slice.len() / std::mem::size_of::<$ty>()];
                bytemuck::cast_slice_mut::<_, u8>(&mut v).copy_from_slice($slice);
                std::borrow::Cow::Owned(v)
            }
        }
    };
}

fn zmq_listener_loop(state_arc: Arc<SharedState>, pull_port: String) {
    let ctx = zmq::Context::new();
    let puller = ctx.socket(zmq::PULL).unwrap();
    puller.bind(&pull_port).unwrap();

    let mut message = zmq::Message::new();
    loop {
        if puller.recv(&mut message, 0).is_ok() {
            let payload = message.as_ref();
            if payload.len() < 72 {
                continue;
            }

            let diff = i32::from_le_bytes(payload[0..4].try_into().unwrap());
            let score = f32::from_le_bytes(payload[4..8].try_into().unwrap());
            let length = u64::from_le_bytes(payload[8..16].try_into().unwrap()) as usize;

            if length == 0 {
                continue;
            }

            let b_len = u64::from_le_bytes(payload[16..24].try_into().unwrap()) as usize;
            let av_len = u64::from_le_bytes(payload[24..32].try_into().unwrap()) as usize;
            let a_len = u64::from_le_bytes(payload[32..40].try_into().unwrap()) as usize;
            let pid_len = u64::from_le_bytes(payload[40..48].try_into().unwrap()) as usize;
            let r_len = u64::from_le_bytes(payload[48..56].try_into().unwrap()) as usize;
            let pol_len = u64::from_le_bytes(payload[56..64].try_into().unwrap()) as usize;
            let v_len = u64::from_le_bytes(payload[64..72].try_into().unwrap()) as usize;

            let mut offset = 72;
            let b_slice = &payload[offset..offset + b_len];
            offset += b_len;
            let av_slice = &payload[offset..offset + av_len];
            offset += av_len;
            let a_slice = &payload[offset..offset + a_len];
            offset += a_len;
            let pid_slice = &payload[offset..offset + pid_len];
            offset += pid_len;
            let r_slice = &payload[offset..offset + r_len];
            offset += r_len;
            let pol_slice = &payload[offset..offset + pol_len];
            offset += pol_len;
            let v_slice = &payload[offset..offset + v_len];

            let b_np = safe_cast!(b_slice, u64);
            let av_np = safe_cast!(av_slice, i32);
            let a_np = safe_cast!(a_slice, i64);
            let pid_np = safe_cast!(pid_slice, i64);
            let r_np = safe_cast!(r_slice, f32);
            let pol_np = safe_cast!(pol_slice, f32);
            let v_np = safe_cast!(v_slice, f32);

            // Obtain writing indexes lock-free
            let eps_start = state_arc.global_write_idx.load(Ordering::Relaxed);
            let state = &state_arc;
            let cap = state.capacity;
            let current_diff = state.current_diff.load(Ordering::Relaxed);
            let next_write_idx = eps_start + length;
            state
                .global_write_active_idx
                .store(next_write_idx, Ordering::Release);

            if current_diff == 0 || diff != current_diff {
                state.current_diff.store(diff, Ordering::Relaxed);
            }
            let current_diff = state.current_diff.load(Ordering::Relaxed);
            let diff_penalty = 10f64.powf(-(current_diff - diff).abs() as f64);

            // First we write all raw arrays concurrently without updating global index
            unsafe {
                for i in 0..length {
                    let idx = (eps_start + i) % cap;
                    (&mut *state.arrays.state_start.get())[idx] = eps_start as i64;
                    (&mut *state.arrays.state_diff.get())[idx] = diff;
                    (&mut *state.arrays.state_len.get())[idx] = length as i32;

                    (&mut *state.arrays.boards.get())[idx][0] = b_np[i * 2];
                    (&mut *state.arrays.boards.get())[idx][1] = b_np[i * 2 + 1];
                    (&mut *state.arrays.available.get())[idx][0] = av_np[i * 3];
                    (&mut *state.arrays.available.get())[idx][1] = av_np[i * 3 + 1];
                    (&mut *state.arrays.available.get())[idx][2] = av_np[i * 3 + 2];
                    (&mut *state.arrays.actions.get())[idx] = a_np[i];
                    (&mut *state.arrays.piece_ids.get())[idx] = pid_np[i];
                    (&mut *state.arrays.rewards.get())[idx] = r_np[i];
                    for j in 0..288 {
                        (&mut *state.arrays.policies.get())[idx][j] = pol_np[i * 288 + j];
                    }
                    (&mut *state.arrays.values.get())[idx] = v_np[i];
                }
            }

            // NOW we update the Prioritized Replay under lock
            let mut add_indices = Vec::with_capacity(length);
            let mut add_diff_penalties = Vec::with_capacity(length);
            for i in 0..length {
                add_indices.push((eps_start + i) % cap);
                add_diff_penalties.push(diff_penalty);
            }
            state.per.add_batch(&add_indices, &add_diff_penalties);

            // NOW WE EXPOSE THE DATA VIA RELEASE
            let next_write_idx = eps_start + length;
            state
                .global_write_idx
                .store(next_write_idx, Ordering::Release);

            {
                let mut eps = state.episodes.lock().unwrap();
                eps.push(EpisodeMeta {
                    global_start_idx: eps_start,
                    length,
                    difficulty: diff,
                    score,
                });

                let mut valid_eps = Vec::new();
                for ep in eps.iter() {
                    if ep.global_start_idx + state.capacity >= next_write_idx {
                        valid_eps.push(ep.clone());
                    }
                }
                *eps = valid_eps;
            }

            let mut recent = state.recent_scores.lock().unwrap();
            recent.push(score);
            state.completed_games.fetch_add(1, Ordering::Relaxed);

            let cur_num = state.num_states.load(Ordering::Relaxed);
            state
                .num_states
                .store(cap.min(cur_num + length), Ordering::Relaxed);
        }
    }
}
