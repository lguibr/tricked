use crate::board::GameStateExt;
use crate::features::extract_feature_native;
use std::sync::atomic::{AtomicI32, AtomicUsize};
use std::sync::{Mutex, RwLock};

pub struct ShardedStorageArrays {
    pub shards: Vec<RwLock<StorageArrays>>,
    pub num_shards: usize,
}

impl ShardedStorageArrays {
    pub fn new(capacity: usize, num_shards: usize) -> Self {
        let mut shards = Vec::with_capacity(num_shards);
        let shard_capacity = capacity / num_shards + 1;
        for _ in 0..num_shards {
            shards.push(RwLock::new(StorageArrays::new(shard_capacity)));
        }
        Self { shards, num_shards }
    }

    #[inline]
    pub fn read_idx<T>(&self, circ_idx: usize, f: impl FnOnce(&StorageArrays, usize) -> T) -> T {
        let shard_idx = circ_idx % self.num_shards;
        let internal_idx = circ_idx / self.num_shards;
        let arr = match self.shards[shard_idx].read() {
            Ok(lock) => lock,
            Err(e) => e.into_inner(),
        };
        f(&arr, internal_idx)
    }

    #[inline]
    pub fn write_idx<T>(
        &self,
        circ_idx: usize,
        f: impl FnOnce(&mut StorageArrays, usize) -> T,
    ) -> T {
        let shard_idx = circ_idx % self.num_shards;
        let internal_idx = circ_idx / self.num_shards;
        let mut arr = match self.shards[shard_idx].write() {
            Ok(lock) => lock,
            Err(e) => e.into_inner(),
        };
        f(&mut arr, internal_idx)
    }
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
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

    pub arrays: ShardedStorageArrays,
    pub per: crate::sumtree::ShardedPrioritizedReplay,

    pub episodes: Mutex<Vec<EpisodeMeta>>,
    pub recent_scores: Mutex<Vec<f32>>,
    pub completed_games: AtomicUsize,
}

impl SharedState {
    pub fn get_features(&self, global_idx: usize) -> Vec<f32> {
        let circ_idx = global_idx % self.capacity;
        let shard_idx = circ_idx % self.arrays.num_shards;
        let internal_idx = circ_idx / self.arrays.num_shards;

        let arr = match self.arrays.shards[shard_idx].read() {
            Ok(lock) => lock,
            Err(e) => e.into_inner(),
        };

        let start_global = arr.state_start[internal_idx];
        if start_global == -1 {
            return vec![0.0; 20 * 96];
        }

        let diff = arr.state_diff[internal_idx];
        let b = arr.boards[internal_idx];
        let board = ((b[1] as u128) << 64) | (b[0] as u128);

        let av = arr.available[internal_idx];
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
                    let p_circ = prev_global % self.capacity;
                    let p_shard = p_circ % self.arrays.num_shards;
                    let p_internal = p_circ / self.arrays.num_shards;
                    if p_shard == shard_idx {
                        let pb = arr.boards[p_internal];
                        history.push(((pb[1] as u128) << 64) | (pb[0] as u128));
                    } else {
                        let p_arr = match self.arrays.shards[p_shard].read() {
                            Ok(lock) => lock,
                            Err(e) => e.into_inner(),
                        };
                        let pb = p_arr.boards[p_internal];
                        history.push(((pb[1] as u128) << 64) | (pb[0] as u128));
                    }
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
                    let p_circ = prev_global % self.capacity;
                    let p_shard = p_circ % self.arrays.num_shards;
                    let p_internal = p_circ / self.arrays.num_shards;
                    if p_shard == shard_idx {
                        actions.push(arr.actions[p_internal] as i32);
                    } else {
                        let p_arr = match self.arrays.shards[p_shard].read() {
                            Ok(lock) => lock,
                            Err(e) => e.into_inner(),
                        };
                        actions.push(p_arr.actions[p_internal] as i32);
                    }
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
