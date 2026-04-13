use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::PyObject;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};
use std::sync::Arc;
use std::sync::RwLock;

use crate::config::Config;

use crate::env::worker::self_play;
use crate::queue::FixedInferenceQueue;
use crate::train::buffer::core::ReplayBuffer;

#[pyclass]
pub struct TrickedEngine {
    pub replay_buffer: Arc<ReplayBuffer>,
    pub inference_queue: Arc<FixedInferenceQueue>,
    pub configuration: Arc<Config>,
    pub active_training_flag: Arc<AtomicBool>,
    pub shared_heatmap: Arc<RwLock<[f32; 96]>>,
    pub global_difficulty: Arc<AtomicI32>,
    pub global_gumbel_scale_multiplier: Arc<AtomicU32>,
}

#[pymethods]
impl TrickedEngine {
    #[new]
    pub fn new(
        capacity: usize,
        config_bytes: &[u8],
        run_id: String,
        run_name: String,
        run_type: String,
    ) -> PyResult<Self> {
        let configuration = Config::from_bytes(config_bytes).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Strict Protobuf Decode Error: {}", e))
        })?;

        let buffer = ReplayBuffer::new(
            configuration.optimizer.buffer_capacity_limit,
            configuration.optimizer.unroll_steps,
            configuration.optimizer.temporal_difference_steps,
            configuration.optimizer.train_batch_size,
            Some(format!("backend/workspace/runs/{}/artifacts", run_id)),
            configuration.optimizer.discount_factor,
            configuration.optimizer.td_lambda,
            configuration.optimizer.prioritized_replay_alpha,
            configuration.optimizer.prioritized_replay_beta,
            run_id,
            run_name,
            run_type,
        );

        Ok(Self {
            replay_buffer: Arc::new(buffer),
            inference_queue: FixedInferenceQueue::new(capacity.min(65536), 16),
            configuration: Arc::new(configuration),
            active_training_flag: Arc::new(AtomicBool::new(true)),
            shared_heatmap: Arc::new(RwLock::new([0.0; 96])),
            global_difficulty: Arc::new(AtomicI32::new(3)),
            global_gumbel_scale_multiplier: Arc::new(AtomicU32::new(100)),
        })
    }

    pub fn start_workers(
        &self,
        num_workers: usize,
        initial_model_path: String,
        recurrent_model_path: String,
        use_cuda: bool,
    ) {
        self.active_training_flag.store(true, Ordering::SeqCst);
        self.inference_queue
            .active_producers
            .store(num_workers, Ordering::SeqCst);

        let reanalyze_ratio = self.configuration.optimizer.reanalyze_ratio;
        let reanalyze_workers = (num_workers as f32 * reanalyze_ratio).floor() as usize;
        let self_play_workers = num_workers - reanalyze_workers;

        // Spawn Self-Play Workers
        for worker_id in 0..self_play_workers {
            let thread_configuration = Arc::clone(&self.configuration);
            let thread_evaluation_sender = Arc::clone(&self.inference_queue);
            let thread_replay_buffer = Arc::clone(&self.replay_buffer);
            let thread_active_flag = Arc::clone(&self.active_training_flag);
            let thread_heatmap = Arc::clone(&self.shared_heatmap);
            let thread_difficulty = Arc::clone(&self.global_difficulty);
            let thread_gumbel_multiplier = Arc::clone(&self.global_gumbel_scale_multiplier);

            let _ = std::thread::Builder::new()
                .name(format!("mcts-worker-{}", worker_id))
                .spawn(move || {
                    while thread_active_flag.load(Ordering::Relaxed) {
                        self_play::game_loop(self_play::GameLoopExecutionParameters {
                            configuration: Arc::clone(&thread_configuration),
                            evaluation_transmitter: Arc::clone(&thread_evaluation_sender),
                            experience_buffer: Arc::clone(&thread_replay_buffer),
                            worker_id,
                            active_flag: Arc::clone(&thread_active_flag),
                            shared_heatmap: Arc::clone(&thread_heatmap),
                            global_difficulty: Arc::clone(&thread_difficulty),
                            global_gumbel_scale_multiplier: Arc::clone(&thread_gumbel_multiplier),
                        });
                    }
                });
        }

        // Spawn Reanalyze Workers
        for worker_id in self_play_workers..num_workers {
            let thread_configuration = Arc::clone(&self.configuration);
            let thread_evaluation_sender = Arc::clone(&self.inference_queue);
            let thread_replay_buffer = Arc::clone(&self.replay_buffer);
            let thread_active_flag = Arc::clone(&self.active_training_flag);

            let _ = std::thread::Builder::new()
                .name(format!("reanalyze-worker-{}", worker_id))
                .spawn(move || {
                    crate::env::reanalyze::reanalyze_worker_loop(
                        thread_configuration,
                        thread_evaluation_sender,
                        thread_replay_buffer,
                        worker_id,
                        thread_active_flag,
                    );
                });
        }

        let inf_queue = Arc::clone(&self.inference_queue);
        let inf_config = Arc::clone(&self.configuration);
        let inf_flag = Arc::clone(&self.active_training_flag);

        std::thread::Builder::new()
            .name("rust-inference-worker".into())
            .spawn(move || {
                crate::inference_worker::r_inference_worker(
                    inf_queue,
                    inf_config,
                    inf_flag,
                    &initial_model_path,
                    &recurrent_model_path,
                    use_cuda,
                );
            })
            .expect("Failed to spawn inference native thread");
    }

    pub fn stop_workers(&self) {
        self.active_training_flag.store(false, Ordering::SeqCst);
        self.inference_queue
            .active_producers
            .store(0, Ordering::SeqCst);
    }

    pub fn poll_inference(
        &self,
        py: Python,
        max_batch_size: usize,
        timeout_ms: u64,
    ) -> PyResult<PyObject> {
        let (initial_batch, recurrent_batch) = match self
            .inference_queue
            .pop_batch_timeout(max_batch_size, std::time::Duration::from_millis(timeout_ms))
        {
            Ok(b) => b,
            Err(_) => return Ok(py.None()),
        };

        if initial_batch.is_empty() && recurrent_batch.is_empty() {
            return Ok(py.None());
        }

        let dict = pyo3::types::PyDict::new_bound(py);

        if !initial_batch.is_empty() {
            let mut initial_ids = Vec::with_capacity(initial_batch.len());
            for guard in initial_batch {
                initial_ids.push(guard.disarm());
            }

            dict.set_item("initial_ids", initial_ids.clone().into_pyarray_bound(py))?;

            // Generate dense batches
            let b_size = initial_ids.len();
            let mut dense_boards = vec![0i64; b_size * 2];
            let mut dense_avail = vec![1i32; b_size * 3];
            let mut dense_hist = vec![0i64; b_size * 14];
            let mut dense_acts = vec![0i32; b_size * 3];
            let mut dense_diff = vec![0i32; b_size];
            let mut initial_leaf_cache = vec![0u32; b_size];

            unsafe {
                let boards_ref = &*self.inference_queue.initial_boards_pinned.get();
                let avail_ref = &*self.inference_queue.initial_avail_pinned.get();
                let hist_ref = &*self.inference_queue.initial_hist_pinned.get();
                let acts_ref = &*self.inference_queue.initial_acts_pinned.get();
                let diff_ref = &*self.inference_queue.initial_diff_pinned.get();

                for (i, &slot) in initial_ids.iter().enumerate() {
                    dense_boards[i * 2..i * 2 + 2]
                        .copy_from_slice(&boards_ref[slot * 2..slot * 2 + 2]);
                    dense_avail[i * 3..i * 3 + 3]
                        .copy_from_slice(&avail_ref[slot * 3..slot * 3 + 3]);
                    dense_hist[i * 14..i * 14 + 14]
                        .copy_from_slice(&hist_ref[slot * 14..slot * 14 + 14]);
                    dense_acts[i * 3..i * 3 + 3].copy_from_slice(&acts_ref[slot * 3..slot * 3 + 3]);
                    dense_diff[i] = diff_ref[slot];

                    if let Some((req, _)) = &*self.inference_queue.metadata[slot].get() {
                        initial_leaf_cache[i] = req.leaf_cache_index;
                    }
                }
            }

            dict.set_item("initial_boards", dense_boards.into_pyarray_bound(py))?;
            dict.set_item("initial_avail", dense_avail.into_pyarray_bound(py))?;
            dict.set_item("initial_hist", dense_hist.into_pyarray_bound(py))?;
            dict.set_item("initial_acts", dense_acts.into_pyarray_bound(py))?;
            dict.set_item("initial_diff", dense_diff.into_pyarray_bound(py))?;
            dict.set_item(
                "initial_leaf_cache_index",
                initial_leaf_cache.into_pyarray_bound(py),
            )?;
        }

        if !recurrent_batch.is_empty() {
            let mut recurrent_ids_raw = Vec::with_capacity(recurrent_batch.len());
            for guard in recurrent_batch {
                recurrent_ids_raw.push(guard.disarm());
            }

            dict.set_item(
                "recurrent_slots",
                recurrent_ids_raw.clone().into_pyarray_bound(py),
            )?;

            let b_size = recurrent_ids_raw.len();
            let mut dense_actions = vec![0i64; b_size];
            let mut dense_ids = vec![0i64; b_size];
            let mut recurrent_parent_cache = vec![0u32; b_size];
            let mut recurrent_leaf_cache = vec![0u32; b_size];

            unsafe {
                let actions_ref = &*self.inference_queue.recurrent_actions_pinned.get();
                let ids_ref = &*self.inference_queue.recurrent_ids_pinned.get();

                for (i, &slot) in recurrent_ids_raw.iter().enumerate() {
                    dense_actions[i] = actions_ref[slot];
                    dense_ids[i] = ids_ref[slot];

                    if let Some((req, _)) = &*self.inference_queue.metadata[slot].get() {
                        recurrent_parent_cache[i] = req.parent_cache_index;
                        recurrent_leaf_cache[i] = req.leaf_cache_index;
                    }
                }
            }

            dict.set_item("recurrent_actions", dense_actions.into_pyarray_bound(py))?;
            dict.set_item("recurrent_ids", dense_ids.into_pyarray_bound(py))?;
            dict.set_item(
                "recurrent_parent_cache_index",
                recurrent_parent_cache.into_pyarray_bound(py),
            )?;
            dict.set_item(
                "recurrent_leaf_cache_index",
                recurrent_leaf_cache.into_pyarray_bound(py),
            )?;
        }

        Ok(dict.into())
    }

    #[pyo3(signature = (max_batch_size, timeout_ms, i_boards_ptr, i_avail_ptr, i_hist_ptr, i_acts_ptr, i_diff_ptr, i_leaf_ptr, r_acts_ptr, r_ids_ptr, r_parent_ptr, r_leaf_ptr))]
    pub fn poll_inference_direct(
        &self,
        py: Python,
        max_batch_size: usize,
        timeout_ms: u64,
        i_boards_ptr: usize,
        i_avail_ptr: usize,
        i_hist_ptr: usize,
        i_acts_ptr: usize,
        i_diff_ptr: usize,
        i_leaf_ptr: usize,
        r_acts_ptr: usize,
        r_ids_ptr: usize,
        r_parent_ptr: usize,
        r_leaf_ptr: usize,
    ) -> PyResult<PyObject> {
        let (initial_batch, recurrent_batch) = match self
            .inference_queue
            .pop_batch_timeout(max_batch_size, std::time::Duration::from_millis(timeout_ms))
        {
            Ok(b) => b,
            Err(_) => return Ok(py.None()),
        };

        if initial_batch.is_empty() && recurrent_batch.is_empty() {
            return Ok(py.None());
        }

        let dict = pyo3::types::PyDict::new_bound(py);

        if !initial_batch.is_empty() {
            let mut initial_ids = Vec::with_capacity(initial_batch.len());
            for guard in initial_batch {
                initial_ids.push(guard.disarm());
            }

            unsafe {
                let out_boards =
                    std::slice::from_raw_parts_mut(i_boards_ptr as *mut i64, max_batch_size * 2);
                let out_avail =
                    std::slice::from_raw_parts_mut(i_avail_ptr as *mut i32, max_batch_size * 3);
                let out_hist =
                    std::slice::from_raw_parts_mut(i_hist_ptr as *mut i64, max_batch_size * 14);
                let out_acts =
                    std::slice::from_raw_parts_mut(i_acts_ptr as *mut i32, max_batch_size * 3);
                let out_diff =
                    std::slice::from_raw_parts_mut(i_diff_ptr as *mut i32, max_batch_size);
                let out_leaf =
                    std::slice::from_raw_parts_mut(i_leaf_ptr as *mut u32, max_batch_size);

                let boards_ref = &*self.inference_queue.initial_boards_pinned.get();
                let avail_ref = &*self.inference_queue.initial_avail_pinned.get();
                let hist_ref = &*self.inference_queue.initial_hist_pinned.get();
                let acts_ref = &*self.inference_queue.initial_acts_pinned.get();
                let diff_ref = &*self.inference_queue.initial_diff_pinned.get();

                for (i, &slot) in initial_ids.iter().enumerate() {
                    out_boards[i * 2..i * 2 + 2]
                        .copy_from_slice(&boards_ref[slot * 2..slot * 2 + 2]);
                    out_avail[i * 3..i * 3 + 3].copy_from_slice(&avail_ref[slot * 3..slot * 3 + 3]);
                    out_hist[i * 14..i * 14 + 14]
                        .copy_from_slice(&hist_ref[slot * 14..slot * 14 + 14]);
                    out_acts[i * 3..i * 3 + 3].copy_from_slice(&acts_ref[slot * 3..slot * 3 + 3]);
                    out_diff[i] = diff_ref[slot];

                    if let Some((req, _)) = &*self.inference_queue.metadata[slot].get() {
                        out_leaf[i] = req.leaf_cache_index;
                    }
                }
            }

            dict.set_item("initial_ids", initial_ids.into_pyarray_bound(py))?;
        }

        if !recurrent_batch.is_empty() {
            let mut recurrent_ids_raw = Vec::with_capacity(recurrent_batch.len());
            for guard in recurrent_batch {
                recurrent_ids_raw.push(guard.disarm());
            }

            unsafe {
                let out_acts =
                    std::slice::from_raw_parts_mut(r_acts_ptr as *mut i64, max_batch_size);
                let out_ids = std::slice::from_raw_parts_mut(r_ids_ptr as *mut i64, max_batch_size);
                let out_parent =
                    std::slice::from_raw_parts_mut(r_parent_ptr as *mut u32, max_batch_size);
                let out_leaf =
                    std::slice::from_raw_parts_mut(r_leaf_ptr as *mut u32, max_batch_size);

                let actions_ref = &*self.inference_queue.recurrent_actions_pinned.get();
                let ids_ref = &*self.inference_queue.recurrent_ids_pinned.get();

                for (i, &slot) in recurrent_ids_raw.iter().enumerate() {
                    out_acts[i] = actions_ref[slot];
                    out_ids[i] = ids_ref[slot];

                    if let Some((req, _)) = &*self.inference_queue.metadata[slot].get() {
                        out_parent[i] = req.parent_cache_index;
                        out_leaf[i] = req.leaf_cache_index;
                    }
                }
            }

            dict.set_item("recurrent_slots", recurrent_ids_raw.into_pyarray_bound(py))?;
        }

        Ok(dict.into())
    }

    #[pyo3(signature = (initial_ids, initial_policies=None, initial_values=None, recurrent_ids=vec![], recurrent_policies=None, recurrent_values=None, recurrent_rewards=None))]
    pub fn submit_inference_results(
        &self,
        _py: Python,
        initial_ids: Vec<usize>,
        initial_policies: Option<&pyo3::prelude::Bound<'_, numpy::PyArray2<f32>>>,
        initial_values: Option<&pyo3::prelude::Bound<'_, numpy::PyArray1<f32>>>,
        recurrent_ids: Vec<usize>,
        recurrent_policies: Option<&pyo3::prelude::Bound<'_, numpy::PyArray2<f32>>>,
        recurrent_values: Option<&pyo3::prelude::Bound<'_, numpy::PyArray1<f32>>>,
        recurrent_rewards: Option<&pyo3::prelude::Bound<'_, numpy::PyArray1<f32>>>,
    ) -> PyResult<()> {
        let channel = &self.inference_queue;

        if !initial_ids.is_empty() {
            let p_arr = unsafe { initial_policies.unwrap().as_array() };
            let v_arr = unsafe { initial_values.unwrap().as_array() };

            for (i, &slot) in initial_ids.iter().enumerate() {
                unsafe {
                    let maybe_req = &mut *channel.metadata[slot].get();
                    if let Some((req, _time)) = maybe_req.take() {
                        let policy_slice = p_arr.row(i).to_slice().unwrap();
                        let value = *v_arr.get(i).unwrap();

                        let mut arr = [0.0; 288];
                        arr.copy_from_slice(policy_slice);

                        req.mailbox
                            .write_and_notify(crate::mcts::EvaluationResponse {
                                child_prior_probabilities_tensor: arr,
                                value_prefix: 0.0,
                                value,
                                node_index: req.node_index,
                                generation: req.generation,
                            });
                        let _ = channel.free_slots.push(slot);
                    }
                }
            }
        }

        if !recurrent_ids.is_empty() {
            let p_arr = unsafe { recurrent_policies.unwrap().as_array() };
            let v_arr = unsafe { recurrent_values.unwrap().as_array() };
            let r_arr = unsafe { recurrent_rewards.unwrap().as_array() };

            for (i, &slot) in recurrent_ids.iter().enumerate() {
                unsafe {
                    let maybe_req = &mut *channel.metadata[slot].get();
                    if let Some((req, _time)) = maybe_req.take() {
                        let policy_slice = p_arr.row(i).to_slice().unwrap();
                        let value = *v_arr.get(i).unwrap();
                        let reward = *r_arr.get(i).unwrap();

                        let mut arr = [0.0; 288];
                        arr.copy_from_slice(policy_slice);

                        req.mailbox
                            .write_and_notify(crate::mcts::EvaluationResponse {
                                child_prior_probabilities_tensor: arr,
                                value_prefix: reward,
                                value,
                                node_index: req.node_index,
                                generation: req.generation,
                            });
                        let _ = channel.free_slots.push(slot);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn get_buffer_length(&self) -> usize {
        self.replay_buffer.get_length()
    }

    pub fn get_telemetry(&self, py: Python) -> PyResult<PyObject> {
        let metrics = self.replay_buffer.get_and_clear_metrics();

        let dict = pyo3::types::PyDict::new_bound(py);

        // Calculate dynamic winrate from recent scores (-1.0 to 1.0)
        let win_rate = if !metrics.cloned_scores.is_empty() {
            let wins = metrics.cloned_scores.iter().filter(|&&s| s > 0.0).count() as f32;
            wins / metrics.cloned_scores.len() as f32
        } else {
            0.0
        };

        dict.set_item("mcts_depth_mean", metrics.mcts_depth_mean)?;
        dict.set_item("mcts_search_time_mean", metrics.mcts_time_mean)?;
        dict.set_item("game_score_mean", metrics.game_score_mean)?;
        dict.set_item("game_score_med", metrics.game_score_med)?;
        dict.set_item("game_score_max", metrics.game_score_max)?;
        dict.set_item("game_score_min", metrics.game_score_min)?;
        dict.set_item("game_lines_cleared", metrics.game_lines_cleared)?;
        dict.set_item(
            "difficulty",
            self.global_difficulty
                .load(std::sync::atomic::Ordering::Relaxed) as f32,
        )?;
        dict.set_item("game_count", metrics.cloned_scores.len())?;
        dict.set_item("win_rate", win_rate)?;
        dict.set_item("action_space_entropy", 0.0)?; // Placedholder or from state

        // Extract heatmap
        let heatmap = self.shared_heatmap.read().unwrap();
        let heatmap_vec = heatmap[..].to_vec();
        dict.set_item("spatial_heatmap", heatmap_vec)?;

        // Append Advanced Dynamic Queue Metrics
        let init_ready = self.inference_queue.initial_ready.len() as f32;
        let rec_ready = self.inference_queue.recurrent_ready.len() as f32;
        let limit = self.configuration.hardware.inference_batch_size_limit as f32;
        let q_sat = if limit > 0.0 {
            (init_ready + rec_ready) / limit
        } else {
            0.0
        };

        let sps_tps = 1.0; // Simulated ratio for SPS vs TPS tracking

        let lat_sum = self
            .inference_queue
            .latency_sum_nanos
            .swap(0, std::sync::atomic::Ordering::Relaxed);
        let lat_count = self
            .inference_queue
            .latency_count
            .swap(0, std::sync::atomic::Ordering::Relaxed);
        let q_lat_us = if lat_count > 0 {
            (lat_sum / lat_count) as f32 / 1000.0
        } else {
            0.0
        };

        let tree_cont = self.replay_buffer.state.per.get_and_reset_contention() as f32 / 1000.0;

        dict.set_item("queue_saturation_ratio", q_sat)?;
        dict.set_item("queue_latency_us", q_lat_us)?;
        dict.set_item("sumtree_contention_us", tree_cont)?;
        dict.set_item("sps_vs_tps", sps_tps)?;

        Ok(dict.into())
    }

    pub fn sample_batch(&self, py: Python, batch_size: usize) -> PyResult<PyObject> {
        if let Some(mut batch) = self.replay_buffer.sample_batch(batch_size, 1.0) {
            let dict = pyo3::types::PyDict::new_bound(py);

            dict.set_item(
                "board_states",
                batch.board_states_batch.into_pyarray_bound(py),
            )?;
            dict.set_item(
                "board_histories",
                batch.board_histories_batch.into_pyarray_bound(py),
            )?;
            dict.set_item(
                "board_available",
                batch.board_available_batch.into_pyarray_bound(py),
            )?;
            dict.set_item(
                "board_historical_acts",
                batch.board_historical_acts_batch.into_pyarray_bound(py),
            )?;
            dict.set_item("board_diff", batch.board_diff_batch.into_pyarray_bound(py))?;
            dict.set_item("actions", batch.actions_batch.into_pyarray_bound(py))?;
            dict.set_item(
                "piece_identifiers",
                batch.piece_identifiers_batch.into_pyarray_bound(py),
            )?;
            dict.set_item(
                "value_prefixs",
                batch.value_prefixs_batch.into_pyarray_bound(py),
            )?;
            dict.set_item(
                "target_policies",
                batch.target_policies_batch.into_pyarray_bound(py),
            )?;
            dict.set_item(
                "target_values",
                batch.target_values_batch.into_pyarray_bound(py),
            )?;
            dict.set_item(
                "model_values",
                batch.model_values_batch.into_pyarray_bound(py),
            )?;
            dict.set_item(
                "raw_unrolled_boards",
                batch.raw_unrolled_boards_batch.into_pyarray_bound(py),
            )?;
            dict.set_item(
                "raw_unrolled_histories",
                batch.raw_unrolled_histories_batch.into_pyarray_bound(py),
            )?;
            dict.set_item(
                "raw_unrolled_available",
                batch.raw_unrolled_available_batch.into_pyarray_bound(py),
            )?;
            dict.set_item(
                "raw_unrolled_actions",
                batch.raw_unrolled_actions_batch.into_pyarray_bound(py),
            )?;
            dict.set_item(
                "raw_unrolled_diff",
                batch.raw_unrolled_diff_batch.into_pyarray_bound(py),
            )?;
            dict.set_item("loss_masks", batch.loss_masks_batch.into_pyarray_bound(py))?;
            dict.set_item(
                "importance_weights",
                batch.importance_weights_batch.into_pyarray_bound(py),
            )?;

            if let Some(arena) = batch.arena.take() {
                let ptr = Box::into_raw(Box::new(arena)) as usize;
                dict.set_item("arena_ptr", ptr)?;
            }

            return Ok(dict.into());
        }

        Ok(py.None())
    }

    pub fn release_batch_arena(&self, ptr: usize) {
        if ptr != 0 {
            let arena =
                unsafe { Box::from_raw(ptr as *mut crate::train::buffer::core::SampleArena) };
            self.replay_buffer.return_arena(*arena);
        }
    }
}
