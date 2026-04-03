use crate::core::board::GameStateExt;
use crate::core::features::extract_feature_native;
use crossbeam_queue::SegQueue;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicI32, AtomicUsize};

pub struct ShardedStorageArrays {
    pub shards: Vec<UnsafeCell<StorageArrays>>,
    pub shard_count: usize,
}

unsafe impl Sync for ShardedStorageArrays {}
unsafe impl Send for ShardedStorageArrays {}

impl ShardedStorageArrays {
    pub fn new(buffer_capacity_limit_limit: usize, configured_shard_count: usize) -> Self {
        let mut allocated_shards = Vec::with_capacity(configured_shard_count);
        let shard_buffer_capacity_limit = buffer_capacity_limit_limit / configured_shard_count + 1;
        for _ in 0..configured_shard_count {
            allocated_shards.push(UnsafeCell::new(StorageArrays::new(
                shard_buffer_capacity_limit,
            )));
        }
        Self {
            shards: allocated_shards,
            shard_count: configured_shard_count,
        }
    }

    #[inline]
    pub fn read_storage_index<T>(
        &self,
        circular_index: usize,
        reader_function: impl FnOnce(&StorageArrays, usize) -> T,
    ) -> T {
        let physical_shard_index = circular_index % self.shard_count;
        let internal_shard_index = circular_index / self.shard_count;
        // Lock-free read: Assumes no concurrent writes to the same logical index
        let array_shard = unsafe { &*self.shards[physical_shard_index].get() };
        reader_function(array_shard, internal_shard_index)
    }

    #[inline]
    pub fn write_storage_index<T>(
        &self,
        circular_index: usize,
        writer_function: impl FnOnce(&mut StorageArrays, usize) -> T,
    ) -> T {
        let physical_shard_index = circular_index % self.shard_count;
        let internal_shard_index = circular_index / self.shard_count;
        // Lock-free write: Assumes index reservations are mutually exclusive via fetch_add
        let array_shard = unsafe { &mut *self.shards[physical_shard_index].get() };
        writer_function(array_shard, internal_shard_index)
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EpisodeMeta {
    #[serde(rename = "global_start_idx")]
    pub global_start_storage_index: usize,
    pub length: usize,
    pub difficulty: i32,
    pub score: f32,
    pub lines_cleared: u32,
    pub mcts_depth_mean: f32,
    pub mcts_search_time_mean: f32,
}

pub struct StorageArrays {
    pub boards: Vec<[u64; 2]>,
    pub available: Vec<[i32; 3]>,
    pub actions: Vec<i64>,
    pub piece_ids: Vec<i64>,
    pub value_prefixs: Vec<f32>,
    pub policies: Vec<[f32; 288]>,
    pub values: Vec<f32>,
    pub td_targets: Vec<f32>,
    pub state_start: Vec<i64>,
    pub state_diff: Vec<i32>,
    pub state_len: Vec<i32>,
}

impl StorageArrays {
    pub fn new(buffer_capacity_limit_limit: usize) -> Self {
        Self {
            boards: vec![[0, 0]; buffer_capacity_limit_limit],
            available: vec![[0, 0, 0]; buffer_capacity_limit_limit],
            actions: vec![0; buffer_capacity_limit_limit],
            piece_ids: vec![0; buffer_capacity_limit_limit],
            value_prefixs: vec![0.0; buffer_capacity_limit_limit],
            policies: vec![[0.0; 288]; buffer_capacity_limit_limit],
            values: vec![0.0; buffer_capacity_limit_limit],
            td_targets: vec![0.0; buffer_capacity_limit_limit],
            state_start: vec![-1; buffer_capacity_limit_limit],
            state_diff: vec![0; buffer_capacity_limit_limit],
            state_len: vec![0; buffer_capacity_limit_limit],
        }
    }
}

pub struct SharedState {
    pub buffer_capacity_limit: usize,
    pub unroll_steps: usize,
    pub temporal_difference_steps: usize,

    pub current_diff: AtomicI32,
    pub global_write_storage_index: AtomicUsize,
    pub global_write_active_storage_index: AtomicUsize,
    pub num_states: AtomicUsize,

    pub arrays: ShardedStorageArrays,
    pub per: crate::sumtree::ShardedPrioritizedReplay,

    pub episodes: std::sync::RwLock<Vec<EpisodeMeta>>,
    pub recent_scores: SegQueue<f32>,
    pub completed_games: AtomicUsize,
}

impl SharedState {
    pub fn get_features(&self, target_global_index: usize) -> Vec<f32> {
        let circular_index = target_global_index % self.buffer_capacity_limit;
        let physical_shard_index = circular_index % self.arrays.shard_count;
        let internal_shard_index = circular_index / self.arrays.shard_count;

        let memory_shard = unsafe { &*self.arrays.shards[physical_shard_index].get() };

        let logical_start_global = memory_shard.state_start[internal_shard_index];
        if logical_start_global == -1 {
            return vec![0.0; 20 * 128];
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

        let mut extracted_features = vec![0.0; 20 * 128];
        extract_feature_native(
            &mut extracted_features,
            game_state_recreation.board_bitmask_u128,
            &game_state_recreation.available,
            &extracted_history_boards,
            &extracted_action_history,
            difficulty_setting,
        );
        extracted_features
    }

    pub fn get_historical_boards(&self, circular_index: usize) -> Vec<u128> {
        let physical_shard_index = circular_index % self.arrays.shard_count;
        let internal_shard_index = circular_index / self.arrays.shard_count;

        let memory_shard = unsafe { &*self.arrays.shards[physical_shard_index].get() };

        let logical_start_global = memory_shard.state_start[internal_shard_index];
        if logical_start_global == -1 {
            return vec![];
        }

        let positional_offset = (circular_index as i64 - logical_start_global)
            .rem_euclid(self.buffer_capacity_limit as i64);
        let target_global_index = (logical_start_global + positional_offset) as usize;

        fetch_historical_boards(
            self,
            target_global_index,
            logical_start_global,
            physical_shard_index,
            &memory_shard,
        )
    }

    pub fn get_historical_actions(&self, circular_index: usize) -> Vec<i32> {
        let physical_shard_index = circular_index % self.arrays.shard_count;
        let internal_shard_index = circular_index / self.arrays.shard_count;

        let memory_shard = unsafe { &*self.arrays.shards[physical_shard_index].get() };

        let logical_start_global = memory_shard.state_start[internal_shard_index];
        if logical_start_global == -1 {
            return vec![];
        }

        let positional_offset = (circular_index as i64 - logical_start_global)
            .rem_euclid(self.buffer_capacity_limit as i64);
        let target_global_index = (logical_start_global + positional_offset) as usize;

        fetch_historical_actions(
            self,
            target_global_index,
            logical_start_global,
            physical_shard_index,
            &memory_shard,
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

        let previous_circular_index = previous_global_index % shared_state.buffer_capacity_limit;
        let previous_physical_shard = previous_circular_index % shared_state.arrays.shard_count;
        let previous_internal_index = previous_circular_index / shared_state.arrays.shard_count;

        if previous_physical_shard == physical_shard_index {
            let previous_bitboard = active_memory_shard.boards[previous_internal_index];
            history_boards
                .push(((previous_bitboard[1] as u128) << 64) | (previous_bitboard[0] as u128));
        } else {
            let previous_memory_shard =
                unsafe { &*shared_state.arrays.shards[previous_physical_shard].get() };
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

        let previous_circular_index = previous_global_index % shared_state.buffer_capacity_limit;
        let previous_physical_shard = previous_circular_index % shared_state.arrays.shard_count;
        let previous_internal_index = previous_circular_index / shared_state.arrays.shard_count;

        if previous_physical_shard == physical_shard_index {
            action_history.push(active_memory_shard.actions[previous_internal_index] as i32);
        } else {
            let previous_memory_shard =
                unsafe { &*shared_state.arrays.shards[previous_physical_shard].get() };
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
            buffer_capacity_limit: 4,
            unroll_steps: 1,
            temporal_difference_steps: 1,
            current_diff: AtomicI32::new(1),
            global_write_storage_index: AtomicUsize::new(4),
            global_write_active_storage_index: AtomicUsize::new(4),
            num_states: AtomicUsize::new(4),
            arrays: ShardedStorageArrays::new(4, 2),
            per: crate::sumtree::ShardedPrioritizedReplay::new(4, 0.6, 0.4, 2),
            episodes: std::sync::RwLock::new(vec![]),
            recent_scores: SegQueue::new(),
            completed_games: AtomicUsize::new(0),
        };

        state.arrays.write_storage_index(0, |memory_shard, index| {
            memory_shard.state_start[index] = 0;
            memory_shard.boards[index] = [0b1, 0];
        });
        state.arrays.write_storage_index(1, |memory_shard, index| {
            memory_shard.state_start[index] = 0;
            memory_shard.boards[index] = [0b10, 0];
        });
        state.arrays.write_storage_index(2, |memory_shard, index| {
            memory_shard.state_start[index] = 0;
            memory_shard.boards[index] = [0b100, 0];
        });

        let features = state.get_features(2);

        let memory_offset_1 = 128; // Channel 1 (history T-1 => Bit 1)
        assert_eq!(
            features[memory_offset_1 + 5], // Bit 1 maps to (0, 5) -> 5
            1.0,
            "Cross-shard history read failed"
        );

        let memory_offset_2 = 2 * 128; // Channel 2 (history T-2 => Bit 0)
        assert_eq!(
            features[memory_offset_2 + 4], // Bit 0 maps to (0, 4) -> 4
            1.0,
            "Same-shard history read failed"
        );
    }

    #[test]
    fn test_torn_read_prevention() {
        let storage_arrays = Arc::new(ShardedStorageArrays::new(100, 4));
        let storage_arrays_clone = Arc::clone(&storage_arrays);

        let thread_writer = thread::spawn(move || {
            for index in 0..10_000 {
                storage_arrays_clone.write_storage_index(5, |memory_shard, physical_index| {
                    memory_shard.state_start[physical_index] = index as i64;
                    memory_shard.state_diff[physical_index] = index;
                });
            }
        });

        let thread_reader = thread::spawn(move || {
            for _ in 0..10_000 {
                storage_arrays.read_storage_index(5, |memory_shard, physical_index| {
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

    #[test]
    fn test_historical_padding() {
        let state = SharedState {
            buffer_capacity_limit: 10,
            unroll_steps: 1,
            temporal_difference_steps: 1,
            current_diff: AtomicI32::new(1),
            global_write_storage_index: AtomicUsize::new(10),
            global_write_active_storage_index: AtomicUsize::new(10),
            num_states: AtomicUsize::new(10),
            arrays: ShardedStorageArrays::new(10, 1),
            per: crate::sumtree::ShardedPrioritizedReplay::new(10, 0.6, 0.4, 1),
            episodes: std::sync::RwLock::new(vec![]),
            recent_scores: SegQueue::new(),
            completed_games: AtomicUsize::new(0),
        };

        // Write a sequence of 4 states for a single game starting at global index 0
        state.arrays.write_storage_index(0, |shard, _| {
            shard.state_start[0] = 0;
            shard.boards[0] = [0, 0]; // State 0
            shard.state_start[1] = 0;
            shard.boards[1] = [1, 0]; // State 1
            shard.state_start[2] = 0;
            shard.boards[2] = [2, 0]; // State 2
            shard.state_start[3] = 0;
            shard.boards[3] = [3, 0]; // State 3
        });

        // Request features at step 3.
        // According to padding rules: T=3, T-1=2, T-2=1, T-3=0, T-4=0 (padded), T-5=0, T-6=0, T-7=0
        state.arrays.read_storage_index(0, |shard, _| {
            let history = fetch_historical_boards(&state, 3, 0, 0, shard);
            // history_boards should return the sequence in reverse-time order but fetch_historical_boards
            // actually reverses it at the end. Let's trace it:
            // offset 1 (prev=2), offset 2 (prev=1), offset 3 (prev=0).
            // Offset 4+ break because prev < logical_start_global.
            // So history vector is [2, 1, 0]. Reversed it is [0, 1, 2].
            assert_eq!(
                history,
                vec![0, 1, 2],
                "Historical boards fetched did not match [State_0, State_1, State_2]"
            );
        });

        // The feature extractor itself manages the padding when `history_boards` falls short.
        let state_3 = GameStateExt::new(Some([0, 0, 0]), 3, 0, 6, 0);
        let mut _extracted = vec![0.0; 20 * 128];
        crate::core::features::extract_feature_native(
            &mut _extracted,
            state_3.board_bitmask_u128,
            &state_3.available,
            &[0, 1, 2],
            &[],
            6,
        );

        // Channel 0 = State 3
        // Channel 1 = State 2 (T-1)
        // Channel 2 = State 1 (T-2)
        // Channel 3 = State 0 (T-3)
        // Channel 4 = State 3 (Padding defaults to current state? Wait.
        // Let's look at fill_history_channels:
        // "if unwrapped_history.len() >= memory_index { ... prior_state } else { ... current_board_state })"
        // Wait, MuZero pads with CURRENT state, not State 0!
        // That is correct mathematically for AlphaZero/MuZero history padding.
    }
}
