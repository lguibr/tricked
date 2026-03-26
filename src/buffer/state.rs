use crate::board::GameStateExt;
use crate::features::extract_feature_native;
use std::sync::atomic::{AtomicI32, AtomicUsize};
use std::sync::{Mutex, RwLock};

#[derive(Clone, Debug)]
pub struct EpisodeMeta {
    pub global_start_idx: usize,
    pub length: usize,
    pub difficulty: i32,
    pub score: f32,
}

pub struct StorageArrays {
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
}

impl StorageArrays {
    pub fn new(capacity: usize) -> Self {
        Self {
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
        }
    }
}

pub struct SharedState {
    pub capacity: usize,
    pub unroll_steps: usize,
    pub td_steps: usize,

    pub current_diff: AtomicI32,
    pub global_write_idx: AtomicUsize,
    pub global_write_active_idx: AtomicUsize,
    pub num_states: AtomicUsize,

    pub arrays: RwLock<StorageArrays>,
    pub per: crate::sumtree::ShardedPrioritizedReplay,

    pub episodes: Mutex<Vec<EpisodeMeta>>,
    pub recent_scores: Mutex<Vec<f32>>,
    pub completed_games: AtomicUsize,
}

impl SharedState {
    pub fn get_features(&self, global_idx: usize) -> Vec<f32> {
        let idx = global_idx % self.capacity;
        let arr = match self.arrays.read() {
            Ok(lock) => lock,
            Err(e) => e.into_inner(),
        };

        let start_global = arr.state_start[idx];
        if start_global == -1 {
            return vec![0.0; 20 * 96];
        }

        let diff = arr.state_diff[idx];
        let b = arr.boards[idx];
        let board = ((b[1] as u128) << 64) | (b[0] as u128);

        let av = arr.available[idx];
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
                    let pb = arr.boards[p_idx];
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
                    actions.push(arr.actions[p_idx] as i32);
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
