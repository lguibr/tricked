use crate::board::GameStateExt;
use crate::features::extract_feature_native;
use std::sync::atomic::{AtomicI32, AtomicUsize};
use std::sync::{Mutex, RwLock};

pub struct ShardedStorageArrays {
    pub shards: Vec<RwLock<StorageArrays>>,
    pub shard_count: usize,
}

impl ShardedStorageArrays {
    pub fn new(capacity_limit: usize, configured_shard_count: usize) -> Self {
        let mut allocated_shards = Vec::with_capacity(configured_shard_count);
        let shard_capacity = capacity_limit / configured_shard_count + 1;
        for _ in 0..configured_shard_count {
            allocated_shards.push(RwLock::new(StorageArrays::new(shard_capacity)));
        }
        Self {
            shards: allocated_shards,
            shard_count: configured_shard_count,
        }
    }

    #[inline]
    pub fn read_idx<T>(
        &self,
        circular_index: usize,
        reader_function: impl FnOnce(&StorageArrays, usize) -> T,
    ) -> T {
        let physical_shard_index = circular_index % self.shard_count;
        let internal_shard_index = circular_index / self.shard_count;
        let array_shard = match self.shards[physical_shard_index].read() {
            Ok(lock) => lock,
            Err(poisoned_error) => poisoned_error.into_inner(),
        };
        reader_function(&array_shard, internal_shard_index)
    }

    #[inline]
    pub fn write_idx<T>(
        &self,
        circular_index: usize,
        writer_function: impl FnOnce(&mut StorageArrays, usize) -> T,
    ) -> T {
        let physical_shard_index = circular_index % self.shard_count;
        let internal_shard_index = circular_index / self.shard_count;
        let mut array_shard = match self.shards[physical_shard_index].write() {
            Ok(lock) => lock,
            Err(poisoned_error) => poisoned_error.into_inner(),
        };
        writer_function(&mut array_shard, internal_shard_index)
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
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
    pub fn new(capacity_limit: usize) -> Self {
        Self {
            boards: vec![[0, 0]; capacity_limit],
            available: vec![[0, 0, 0]; capacity_limit],
            actions: vec![0; capacity_limit],
            piece_ids: vec![0; capacity_limit],
            rewards: vec![0.0; capacity_limit],
            policies: vec![[0.0; 288]; capacity_limit],
            values: vec![0.0; capacity_limit],
            state_start: vec![-1; capacity_limit],
            state_diff: vec![0; capacity_limit],
            state_len: vec![0; capacity_limit],
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
    pub fn get_features(&self, target_global_index: usize) -> Vec<f32> {
        let circular_index = target_global_index % self.capacity;
        let physical_shard_index = circular_index % self.arrays.shard_count;
        let internal_shard_index = circular_index / self.arrays.shard_count;

        let memory_shard = match self.arrays.shards[physical_shard_index].read() {
            Ok(lock) => lock,
            Err(poisoned_error) => poisoned_error.into_inner(),
        };

        let logical_start_global = memory_shard.state_start[internal_shard_index];
        if logical_start_global == -1 {
            return vec![0.0; 20 * 96];
        }

        let difficulty_setting = memory_shard.state_diff[internal_shard_index];
        let bitboard_shard = memory_shard.boards[internal_shard_index];
        let reconstructed_board = ((bitboard_shard[1] as u128) << 64) | (bitboard_shard[0] as u128);

        let available_shard = memory_shard.available[internal_shard_index];
        let available_pieces = [available_shard[0], available_shard[1], available_shard[2]];

        let game_state_recreation = GameStateExt::new(
            Some(available_pieces),
            reconstructed_board,
            0,
            difficulty_setting,
            available_pieces
                .iter()
                .filter(|&&piece_identifier| piece_identifier != -1)
                .count() as i32,
        );

        let extracted_history_boards = fetch_historical_boards(
            self,
            target_global_index,
            logical_start_global,
            physical_shard_index,
            &memory_shard,
        );

        let extracted_action_history = fetch_historical_actions(
            self,
            target_global_index,
            logical_start_global,
            physical_shard_index,
            &memory_shard,
        );

        extract_feature_native(
            &game_state_recreation,
            Some(extracted_history_boards),
            Some(extracted_action_history),
            difficulty_setting,
        )
    }
}

fn fetch_historical_boards(
    shared_state: &SharedState,
    target_global_index: usize,
    logical_start_global: i64,
    physical_shard_index: usize,
    active_memory_shard: &StorageArrays,
) -> Vec<u128> {
    let mut history_boards = vec![];
    for timestep_offset in 1..=8 {
        if target_global_index < timestep_offset {
            break;
        }
        let previous_global_index = target_global_index - timestep_offset;
        if previous_global_index < logical_start_global as usize {
            break;
        }

        let previous_circular_index = previous_global_index % shared_state.capacity;
        let previous_physical_shard = previous_circular_index % shared_state.arrays.shard_count;
        let previous_internal_index = previous_circular_index / shared_state.arrays.shard_count;

        if previous_physical_shard == physical_shard_index {
            let previous_bitboard = active_memory_shard.boards[previous_internal_index];
            history_boards
                .push(((previous_bitboard[1] as u128) << 64) | (previous_bitboard[0] as u128));
        } else {
            let previous_memory_shard =
                match shared_state.arrays.shards[previous_physical_shard].read() {
                    Ok(lock) => lock,
                    Err(err) => err.into_inner(),
                };
            let previous_bitboard = previous_memory_shard.boards[previous_internal_index];
            history_boards
                .push(((previous_bitboard[1] as u128) << 64) | (previous_bitboard[0] as u128));
        }
    }
    history_boards.reverse();
    history_boards
}

fn fetch_historical_actions(
    shared_state: &SharedState,
    target_global_index: usize,
    logical_start_global: i64,
    physical_shard_index: usize,
    active_memory_shard: &StorageArrays,
) -> Vec<i32> {
    let mut action_history = vec![];
    for timestep_offset in 1..=4 {
        if target_global_index < timestep_offset {
            break;
        }
        let previous_global_index = target_global_index - timestep_offset;
        if previous_global_index < logical_start_global as usize {
            break;
        }

        let previous_circular_index = previous_global_index % shared_state.capacity;
        let previous_physical_shard = previous_circular_index % shared_state.arrays.shard_count;
        let previous_internal_index = previous_circular_index / shared_state.arrays.shard_count;

        if previous_physical_shard == physical_shard_index {
            action_history.push(active_memory_shard.actions[previous_internal_index] as i32);
        } else {
            let previous_memory_shard =
                match shared_state.arrays.shards[previous_physical_shard].read() {
                    Ok(lock) => lock,
                    Err(err) => err.into_inner(),
                };
            action_history.push(previous_memory_shard.actions[previous_internal_index] as i32);
        }
    }
    action_history.reverse();
    action_history
}

#[cfg(test)]
mod tests {
    use super::*;
    // removed
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_cross_shard_history_reads() {
        let state = SharedState {
            capacity: 4,
            unroll_steps: 1,
            td_steps: 1,
            current_diff: AtomicI32::new(1),
            global_write_idx: AtomicUsize::new(4),
            global_write_active_idx: AtomicUsize::new(4),
            num_states: AtomicUsize::new(4),
            arrays: ShardedStorageArrays::new(4, 2),
            per: crate::sumtree::ShardedPrioritizedReplay::new(4, 0.6, 0.4, 2),
            episodes: Mutex::new(vec![]),
            recent_scores: Mutex::new(vec![]),
            completed_games: AtomicUsize::new(0),
        };

        state.arrays.write_idx(0, |memory_shard, index| {
            memory_shard.state_start[index] = 0;
            memory_shard.boards[index] = [0b1, 0];
        });
        state.arrays.write_idx(1, |memory_shard, index| {
            memory_shard.state_start[index] = 0;
            memory_shard.boards[index] = [0b10, 0];
        });
        state.arrays.write_idx(2, |memory_shard, index| {
            memory_shard.state_start[index] = 0;
            memory_shard.boards[index] = [0b100, 0];
        });

        let features = state.get_features(2);

        let memory_offset_1 = 96;
        assert_eq!(
            features[memory_offset_1 + 1],
            1.0,
            "Cross-shard history read failed"
        );

        let memory_offset_2 = 2 * 96;
        assert_eq!(
            features[memory_offset_2], 1.0,
            "Same-shard history read failed"
        );
    }

    #[test]
    fn test_torn_read_prevention() {
        let storage_arrays = Arc::new(ShardedStorageArrays::new(100, 4));
        let storage_arrays_clone = Arc::clone(&storage_arrays);

        let thread_writer = thread::spawn(move || {
            for index in 0..10_000 {
                storage_arrays_clone.write_idx(5, |memory_shard, physical_index| {
                    memory_shard.state_start[physical_index] = index as i64;
                    memory_shard.state_diff[physical_index] = index;
                });
            }
        });

        let thread_reader = thread::spawn(move || {
            for _ in 0..10_000 {
                storage_arrays.read_idx(5, |memory_shard, physical_index| {
                    let logical_start = memory_shard.state_start[physical_index];
                    let difficulty_setting = memory_shard.state_diff[physical_index];
                    if logical_start != -1 {
                        assert_eq!(
                            logical_start as i32, difficulty_setting,
                            "Torn read detected: start and diff arrays desynchronized"
                        );
                    }
                });
            }
        });

        thread_writer.join().unwrap();
        thread_reader.join().unwrap();
    }
}
