use crossbeam_channel::Sender;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tch::Device;
use crate::train::buffer::{BatchTensors, ReplayBuffer};

pub fn spawn_prefetch_thread(
    prefetch_replay_buffer: Arc<ReplayBuffer>,
    prefetch_active_flag: Arc<AtomicBool>,
    prefetch_tx: Sender<BatchTensors>,
    prefetch_device: Device,
    prefetch_batch_size: usize,
    unroll_steps: usize,
    prefetch_max_steps: f64,
) {
    let _ = std::thread::Builder::new()
        .name("prefetch".into())
        .spawn(move || {
            const BUFFER_COUNT: usize = 8;
            let mut pinned_arenas: Vec<_> = (0..BUFFER_COUNT)
                .map(|_| {
                    crate::train::arena::PinnedBatchTensors::new(
                        prefetch_batch_size,
                        unroll_steps,
                        prefetch_device,
                    )
                })
                .collect();
            let mut gpu_arenas: Vec<_> = (0..BUFFER_COUNT)
                .map(|_| {
                    crate::train::arena::GpuBatchTensors::new(
                        prefetch_batch_size,
                        unroll_steps,
                        prefetch_device,
                    )
                })
                .collect();

            let mut cycle = 0;

            while prefetch_active_flag.load(Ordering::Relaxed) {
                if prefetch_replay_buffer.get_length() < prefetch_batch_size {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    continue;
                }
                let current_step = prefetch_replay_buffer
                    .state
                    .completed_games
                    .load(Ordering::Relaxed)
                    as f64;
                let beta =
                    (0.4 + 0.6 * (current_step / prefetch_max_steps.max(100_000.0))).min(1.0);

                if let Some(mut batch) =
                    prefetch_replay_buffer.sample_batch(prefetch_batch_size, beta)
                {
                    let idx = cycle % BUFFER_COUNT;
                    pinned_arenas[idx].copy_from_unpinned(&batch);
                    gpu_arenas[idx].copy_from_pinned(&pinned_arenas[idx]);

                    batch.state_features_batch = gpu_arenas[idx].state_features.shallow_clone();
                    batch.actions_batch = gpu_arenas[idx].actions.shallow_clone();
                    batch.piece_identifiers_batch =
                        gpu_arenas[idx].piece_identifiers.shallow_clone();
                    batch.value_prefixs_batch = gpu_arenas[idx].value_prefixs.shallow_clone();
                    batch.target_policies_batch = gpu_arenas[idx].target_policies.shallow_clone();
                    batch.target_values_batch = gpu_arenas[idx].target_values.shallow_clone();
                    batch.model_values_batch = gpu_arenas[idx].model_values.shallow_clone();
                    batch.raw_unrolled_boards_batch =
                        gpu_arenas[idx].raw_unrolled_boards.shallow_clone();
                    batch.raw_unrolled_histories_batch =
                        gpu_arenas[idx].raw_unrolled_histories.shallow_clone();
                    batch.loss_masks_batch = gpu_arenas[idx].loss_masks.shallow_clone();
                    batch.importance_weights_batch =
                        gpu_arenas[idx].importance_weights.shallow_clone();

                    if prefetch_tx.send(batch).is_err() {
                        break;
                    }
                    cycle += 1;
                } else {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
            }
        });
}
