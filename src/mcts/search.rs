use crate::core::board::GameStateExt;
use crate::node::{get_valid_action_mask, select_child, LatentNode};
use rand::Rng;
use std::collections::HashMap;

use super::evaluator::{EvaluationRequest, EvaluationResponse, NetworkEvaluator};
use super::tree::{allocate_node, expand_root_node, initialize_search_tree, MctsTree};

pub struct MctsParams<'a> {
    pub raw_policy_probabilities: &'a [f32],
    pub root_cache_index: u32,
    pub maximum_allowed_nodes_in_search_tree: u32,
    pub worker_id: usize,
    pub game_state: &'a GameStateExt,
    pub total_simulations: usize,
    pub max_gumbel_k_samples: usize,
    pub gumbel_noise_scale: f32,
    pub previous_tree: Option<MctsTree>,
    pub last_executed_action: Option<i32>,
    pub neural_evaluator: &'a dyn NetworkEvaluator,
    pub evaluation_request_transmitter: crossbeam_channel::Sender<EvaluationResponse>,
    pub evaluation_response_receiver: &'a crossbeam_channel::Receiver<EvaluationResponse>,
    pub active_flag: std::sync::Arc<std::sync::RwLock<bool>>,
    pub _seed: Option<u64>,
}

#[hotpath::measure]
pub fn mcts_search(params: MctsParams) -> Result<(i32, HashMap<i32, i32>, f32, MctsTree), String> {
    let MctsParams {
        raw_policy_probabilities,
        root_cache_index,
        maximum_allowed_nodes_in_search_tree,
        worker_id,
        game_state,
        total_simulations,
        max_gumbel_k_samples,
        gumbel_noise_scale,
        previous_tree,
        last_executed_action,
        neural_evaluator,
        evaluation_request_transmitter,
        evaluation_response_receiver,
        active_flag,
        _seed,
    } = params;
    let (normalized_probabilities, valid_mask, valid_actions) =
        normalize_policy_distributions(raw_policy_probabilities, game_state);

    if valid_actions.is_empty() {
        let tree = initialize_search_tree(
            previous_tree,
            last_executed_action,
            maximum_allowed_nodes_in_search_tree,
            total_simulations,
        );
        return Ok((-1, HashMap::new(), 0.0, tree));
    }

    let mut tree = initialize_search_tree(
        previous_tree,
        last_executed_action,
        maximum_allowed_nodes_in_search_tree,
        total_simulations,
    );
    let root_index = tree.root_index;

    expand_root_node(&mut tree, root_cache_index, &normalized_probabilities);

    let k_dynamic_samples = calculate_dynamic_k_samples(max_gumbel_k_samples, valid_actions.len());
    if k_dynamic_samples <= 1 {
        let mut visit_distribution = HashMap::new();
        visit_distribution.insert(valid_actions[0], 1);
        let val = tree.arena[root_index].value();
        return Ok((valid_actions[0], visit_distribution, val, tree));
    }

    let mut candidate_actions = valid_actions.clone();
    let gumbel_noisy_logits = inject_gumbel_noise(
        &mut tree.arena,
        root_index,
        &candidate_actions,
        &normalized_probabilities,
        gumbel_noise_scale,
    );

    candidate_actions.sort_by(|&action_a, &action_b| {
        gumbel_noisy_logits[action_b as usize]
            .partial_cmp(&gumbel_noisy_logits[action_a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    candidate_actions.truncate(k_dynamic_samples);

    execute_sequential_halving(
        &mut tree,
        &mut candidate_actions,
        total_simulations,
        &gumbel_noisy_logits,
        game_state,
        neural_evaluator,
        worker_id,
        evaluation_request_transmitter,
        evaluation_response_receiver,
        &active_flag,
    )?;

    compute_final_action_distribution(tree, valid_mask, candidate_actions, gumbel_noisy_logits)
}

#[hotpath::measure]
fn normalize_policy_distributions(
    raw_policy_probabilities: &[f32],
    game_state: &GameStateExt,
) -> (Vec<f32>, [bool; 288], Vec<i32>) {
    let valid_action_mask = get_valid_action_mask(game_state);
    let mut normalized_probabilities = Vec::with_capacity(288);
    let mut valid_actions = Vec::new();
    let mut valid_action_count = 0;

    for (index, &probability) in raw_policy_probabilities.iter().enumerate() {
        if valid_action_mask[index] {
            normalized_probabilities.push(probability.max(1e-8));
            valid_actions.push(index as i32);
            valid_action_count += 1;
        } else {
            normalized_probabilities.push(0.0);
        }
    }

    let sum_probabilities: f32 = normalized_probabilities.iter().sum();
    assert!(!sum_probabilities.is_nan(), "Policy mask sum is NaN!");

    if sum_probabilities > 0.0 {
        for probability in normalized_probabilities.iter_mut() {
            *probability /= sum_probabilities;
        }
        let check_sum: f32 = normalized_probabilities.iter().sum();
        assert!(
            (check_sum - 1.0).abs() < 1e-4,
            "Normalized prior policies must sum to 1.0. Actual: {}",
            check_sum
        );
    } else if valid_action_count > 0 {
        for (index, probability) in normalized_probabilities.iter_mut().enumerate() {
            if valid_action_mask[index] {
                *probability = 1.0 / (valid_action_count as f32);
            }
        }
    }

    (normalized_probabilities, valid_action_mask, valid_actions)
}

#[hotpath::measure]
fn calculate_dynamic_k_samples(max_gumbel_k_samples: usize, valid_action_count: usize) -> usize {
    let empty_board_density = 1.0 - (valid_action_count as f32 / 288.0);
    let mut k_dynamic_samples =
        4i32 + ((max_gumbel_k_samples as f32 - 4.0) * empty_board_density) as i32;
    if k_dynamic_samples < 2 {
        k_dynamic_samples = 2;
    }
    (k_dynamic_samples as usize).min(valid_action_count)
}

#[hotpath::measure]
fn inject_gumbel_noise(
    arena: &mut [LatentNode],
    root_index: usize,
    candidate_actions: &[i32],
    normalized_probabilities: &[f32],
    gumbel_noise_scale: f32,
) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut gumbel_noisy_logits = vec![f32::NEG_INFINITY; 288];

    for &action_index in candidate_actions {
        let uniform_random_sample: f32 = rng.gen_range(1e-6..=(1.0 - 1e-6));
        let gumbel_noise_value = -(-(uniform_random_sample.ln())).ln();
        assert!(!gumbel_noise_value.is_nan(), "Gumbel noise is NaN");

        let action_usize = action_index as usize;
        let child_index = arena[root_index].get_child(arena, action_index);

        if child_index != usize::MAX {
            arena[child_index].gumbel_noise = gumbel_noise_value;
            let log_probability = (normalized_probabilities[action_usize] + 1e-8).ln();
            gumbel_noisy_logits[action_usize] =
                log_probability + (gumbel_noise_value * gumbel_noise_scale);
        }
    }
    gumbel_noisy_logits
}

#[hotpath::measure]
#[allow(clippy::too_many_arguments)]
fn execute_sequential_halving(
    tree: &mut MctsTree,
    candidate_actions: &mut Vec<i32>,
    total_simulations: usize,
    gumbel_noisy_logits: &[f32],
    game_state: &GameStateExt,
    neural_evaluator: &dyn NetworkEvaluator,
    worker_id: usize,
    evaluation_request_transmitter: crossbeam_channel::Sender<EvaluationResponse>,
    evaluation_response_receiver: &crossbeam_channel::Receiver<EvaluationResponse>,
    active_flag: &std::sync::Arc<std::sync::RwLock<bool>>,
) -> Result<(), String> {
    let candidate_count = candidate_actions.len();
    let total_halving_phases = if candidate_count > 1 {
        (candidate_count as f32).log2().ceil() as usize
    } else {
        0
    };

    let mut remaining_simulations = total_simulations;
    let mut remaining_phases = total_halving_phases;

    for _phase in 0..total_halving_phases {
        let current_candidate_count = candidate_actions.len();
        if current_candidate_count <= 1 || remaining_phases == 0 {
            break;
        }

        let mut visits_per_candidate =
            (remaining_simulations / remaining_phases) / current_candidate_count;
        if visits_per_candidate == 0 {
            visits_per_candidate = 1;
        }

        expand_and_evaluate_candidates(
            tree,
            candidate_actions,
            visits_per_candidate,
            game_state,
            neural_evaluator,
            worker_id,
            evaluation_request_transmitter.clone(),
            evaluation_response_receiver,
            active_flag,
        )?;

        let root_index = tree.root_index;
        prune_candidates(
            &tree.arena,
            root_index,
            candidate_actions,
            gumbel_noisy_logits,
        );

        remaining_simulations =
            remaining_simulations.saturating_sub(visits_per_candidate * current_candidate_count);
        remaining_phases -= 1;
    }

    Ok(())
}

#[hotpath::measure]
#[allow(clippy::too_many_arguments)]
fn expand_and_evaluate_candidates(
    tree: &mut MctsTree,
    candidate_actions: &[i32],
    visits_per_candidate: usize,
    game_state: &GameStateExt,
    neural_evaluator: &dyn NetworkEvaluator,
    worker_id: usize,
    evaluation_request_transmitter: crossbeam_channel::Sender<EvaluationResponse>,
    evaluation_response_receiver: &crossbeam_channel::Receiver<EvaluationResponse>,
    active_flag: &std::sync::Arc<std::sync::RwLock<bool>>,
) -> Result<(), String> {
    let mut eval_batch = arrayvec::ArrayVec::<EvaluationRequest, 256>::new();
    let mut batch_paths = arrayvec::ArrayVec::<arrayvec::ArrayVec<usize, 64>, 256>::new();

    let root_index = tree.root_index;

    for &candidate_action in candidate_actions {
        for _ in 0..visits_per_candidate {
            let (search_path, leaf_node_index, successfully_traversed) =
                traverse_tree_to_leaf(&tree.arena, root_index, candidate_action);

            if !successfully_traversed {
                continue;
            }

            for &node_idx in &search_path {
                tree.arena[node_idx].visits += 1;
                tree.arena[node_idx].value_sum -= 1.0;
            }

            let parent_index = search_path[search_path.len() - 3];
            let last_action_taken = search_path[search_path.len() - 2];
            let slot_index = last_action_taken / 96;
            let position_bit_index = last_action_taken % 96;
            let mut piece_identifier = game_state.available[slot_index];
            if piece_identifier == -1 {
                piece_identifier = 0;
            }

            let piece_action_identifier = piece_identifier * 96 + (position_bit_index as i32);
            let prev_idx = tree.arena[parent_index].hidden_state_index;
            let new_idx = tree.free_list.pop().unwrap();

            tree.arena[leaf_node_index].hidden_state_index = new_idx;

            let evaluation_request = EvaluationRequest {
                is_initial: false,
                board_bitmask: 0,
                available_pieces: [-1; 3],
                recent_board_history: [0; 8],
                history_len: 0,
                recent_action_history: [0; 4],
                action_history_len: 0,
                difficulty: 6,
                piece_action: piece_action_identifier as i64,
                piece_id: piece_identifier as i64,
                node_index: leaf_node_index,
                worker_id,
                parent_cache_index: prev_idx,
                leaf_cache_index: new_idx,
                evaluation_request_transmitter: evaluation_request_transmitter.clone(),
            };

            batch_paths.push(search_path);
            eval_batch.push(evaluation_request);
        }
    }

    let active_requests = eval_batch.len();
    if active_requests > 0 {
        if let Err(error) = neural_evaluator.send_batch(eval_batch) {
            return Err(format!("Failed sending eval request: {}", error));
        }

        process_evaluation_responses(
            tree,
            evaluation_response_receiver,
            active_requests as u32,
            batch_paths,
            active_flag,
        )?;
    }
    Ok(())
}

#[hotpath::measure]
fn traverse_tree_to_leaf(
    arena: &[LatentNode],
    root_index: usize,
    candidate_action: i32,
) -> (arrayvec::ArrayVec<usize, 64>, usize, bool) {
    let mut search_path = arrayvec::ArrayVec::new();
    search_path.push(root_index);
    let mut current_node_index = root_index;

    let immediate_child_index = arena[current_node_index].get_child(arena, candidate_action);
    if immediate_child_index == usize::MAX {
        return (search_path, current_node_index, false);
    }

    search_path.push(candidate_action as usize);
    search_path.push(immediate_child_index);
    current_node_index = immediate_child_index;

    while arena[current_node_index].is_topologically_expanded {
        let (best_action, next_node_index) = select_child(arena, current_node_index, false);
        if next_node_index == usize::MAX {
            break;
        }
        search_path.push(best_action as usize);
        search_path.push(next_node_index);
        current_node_index = next_node_index;
    }

    (search_path, current_node_index, true)
}

#[hotpath::measure]
fn process_evaluation_responses(
    tree: &mut MctsTree,
    receiver_rx: &crossbeam_channel::Receiver<EvaluationResponse>,
    active_requests: u32,
    batch_paths: arrayvec::ArrayVec<arrayvec::ArrayVec<usize, 64>, 256>,
    active_flag: &std::sync::Arc<std::sync::RwLock<bool>>,
) -> Result<(), String> {
    for _ in 0..active_requests {
        let evaluation_response = loop {
            if !*active_flag.read().unwrap() {
                return Err("Training stopped".to_string());
            }
            match receiver_rx.recv_timeout(std::time::Duration::from_millis(100)) {
                Ok(resp) => break resp,
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
                Err(_) => return Err("Channel disconnected".to_string()),
            }
        };

        let leaf_node_index = evaluation_response.node_index;
        let search_path = batch_paths
            .iter()
            .find(|path| *path.last().unwrap() == leaf_node_index)
            .unwrap();

        tree.arena[leaf_node_index].reward = evaluation_response.reward;
        tree.arena[leaf_node_index].is_topologically_expanded = true;

        let mut prev_child = u32::MAX;
        let mut first_child = u32::MAX;

        for (action_index, &probability) in evaluation_response
            .child_prior_probabilities_tensor
            .iter()
            .enumerate()
        {
            if probability > 0.0 {
                let new_node_index = allocate_node(tree, probability, action_index as i16);
                if first_child == u32::MAX {
                    first_child = new_node_index;
                } else {
                    tree.arena[prev_child as usize].next_sibling = new_node_index;
                }
                prev_child = new_node_index;
            }
        }
        tree.arena[leaf_node_index].first_child = first_child;

        let mut backprop_value = evaluation_response.value;

        for index in (0..search_path.len()).step_by(2).rev() {
            let node_index = search_path[index];
            tree.arena[node_index].value_sum += 1.0;
            tree.arena[node_index].value_sum += backprop_value;
            backprop_value = tree.arena[node_index].reward + 0.99 * backprop_value;
        }
    }
    Ok(())
}

#[hotpath::measure]
fn prune_candidates(
    arena: &[LatentNode],
    root_index: usize,
    candidate_actions: &mut Vec<i32>,
    gumbel_noisy_logits: &[f32],
) {
    let mut candidates_with_nodes: Vec<(i32, usize)> = candidate_actions
        .iter()
        .map(|&a| (a, arena[root_index].get_child(arena, a)))
        .filter(|&(_, idx)| idx != usize::MAX)
        .collect();

    candidates_with_nodes.sort_by(|&(action_a, index_a), &(action_b, index_b)| {
        let node_a = &arena[index_a];
        let node_b = &arena[index_b];

        let q_value_a = node_a.reward + 0.99 * node_a.value();
        let q_value_b = node_b.reward + 0.99 * node_b.value();

        let exploration_scale_a = 50.0 / ((node_a.visits + 1) as f32);
        let score_a = gumbel_noisy_logits[action_a as usize] + (exploration_scale_a * q_value_a);

        let exploration_scale_b = 50.0 / ((node_b.visits + 1) as f32);
        let score_b = gumbel_noisy_logits[action_b as usize] + (exploration_scale_b * q_value_b);

        score_b
            .partial_cmp(&score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let items_to_drop = candidates_with_nodes.len() / 2;
    candidates_with_nodes.truncate(candidates_with_nodes.len() - items_to_drop);

    *candidate_actions = candidates_with_nodes.into_iter().map(|(a, _)| a).collect();
}

#[hotpath::measure]
fn compute_final_action_distribution(
    tree: MctsTree,
    valid_action_mask: [bool; 288],
    candidate_actions: Vec<i32>,
    gumbel_noisy_logits: Vec<f32>,
) -> Result<(i32, HashMap<i32, i32>, f32, MctsTree), String> {
    let arena = &tree.arena;
    let root_index = tree.root_index;
    let mut evaluated_candidates = Vec::new();
    for (action_index, &is_valid) in valid_action_mask.iter().enumerate() {
        let child_index = arena[root_index].get_child(arena, action_index as i32);
        if child_index != usize::MAX && arena[child_index].visits > 0 && is_valid {
            evaluated_candidates.push((action_index as i32, child_index));
        }
    }

    if evaluated_candidates.is_empty() {
        let mut uniform_visits = HashMap::new();
        uniform_visits.insert(candidate_actions[0], 1);
        let val = arena[root_index].value();
        return Ok((candidate_actions[0], uniform_visits, val, tree));
    }

    let mut q_values = Vec::new();
    let mut maximum_q_value = f32::NEG_INFINITY;
    let mut minimum_q_value = f32::INFINITY;

    for &(_action_index, child_index) in &evaluated_candidates {
        let q_value = arena[child_index].reward + 0.99 * arena[child_index].value();
        q_values.push(q_value);
        if q_value > maximum_q_value {
            maximum_q_value = q_value;
        }
        if q_value < minimum_q_value {
            minimum_q_value = q_value;
        }
    }

    let mut visit_distribution = HashMap::new();
    for &(action_index, child_index) in &evaluated_candidates {
        visit_distribution.insert(action_index, arena[child_index].visits);
    }

    let mut optimal_action = candidate_actions[0];
    let mut optimal_action_score = f32::NEG_INFINITY;

    let mut max_visit = 0;
    let mut sum_visit = 0;
    for &(_action_index, child_index) in &evaluated_candidates {
        let visits = arena[child_index].visits;
        sum_visit += visits;
        if visits > max_visit {
            max_visit = visits;
        }
    }

    let exploration_scale = (50.0 + max_visit as f32) / (sum_visit as f32 + 1e-8);

    for &(action_index, child_index) in &evaluated_candidates {
        let q_value = arena[child_index].reward + 0.99 * arena[child_index].value();
        let completed_gumbel_score =
            gumbel_noisy_logits[action_index as usize] + exploration_scale * q_value;

        if completed_gumbel_score > optimal_action_score {
            optimal_action_score = completed_gumbel_score;
            optimal_action = action_index;
        }
    }

    let val = arena[root_index].value();
    Ok((optimal_action, visit_distribution, val, tree))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::GameStateExt;
    use crate::mcts::evaluator::{CustomEvaluator, MockEvaluator};
    use crate::node::LatentNode;

    #[test]
    fn test_q_value_min_max_normalization() {
        let q_values = [10.0, 50.0, 100.0];
        let min_q = q_values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_q = q_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let normalized: Vec<f32> = q_values
            .iter()
            .map(|&q| {
                if max_q > min_q {
                    (q - min_q) / (max_q - min_q)
                } else {
                    0.5
                }
            })
            .collect();

        assert!((normalized[0] - 0.0).abs() < 1e-6);
        assert!((normalized[1] - 0.4444444).abs() < 1e-4);
        assert!((normalized[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gumbel_noise_distribution() {
        let mut mock_arena = vec![LatentNode::new(0.0, 0)];
        mock_arena.push(LatentNode::new(0.33, 1));
        mock_arena.push(LatentNode::new(0.33, 2));
        mock_arena.push(LatentNode::new(0.34, 3));
        mock_arena[0].first_child = 1;
        mock_arena[1].next_sibling = 2;
        mock_arena[2].next_sibling = 3;

        let actions = vec![1, 2, 3];
        let mut probs = vec![0.0; 288];
        probs[1] = 0.33;
        probs[2] = 0.33;
        probs[3] = 0.34;

        let mut sum = 0.0;
        let mut sq_sum = 0.0;
        let n = 200_000;

        for _ in 0..n {
            let res = inject_gumbel_noise(&mut mock_arena, 0, &actions, &probs, 1.0);
            for action in &actions {
                let log_prob = (probs[*action as usize] + 1e-8).ln();
                let noise = res[*action as usize] - log_prob;
                sum += noise;
                sq_sum += noise * noise;
            }
        }

        let mean = sum / (n as f32 * 3.0);
        let std_dev = (sq_sum / (n as f32 * 3.0) - (mean * mean)).sqrt();
        let variance = std_dev * std_dev;

        assert!((mean - 0.5772).abs() < 0.1, "Gumbel mean is off: {}", mean);
        assert!(
            (variance - 1.6449).abs() < 0.2,
            "Gumbel variance is off: {}",
            variance
        );
    }

    #[test]
    fn test_valid_action_mask() {
        let mut state = GameStateExt::new(Some([0, 1, 2]), 0, 0, 6, 0);
        let mask = get_valid_action_mask(&state);
        assert!(mask.contains(&true));

        state.terminal = true;
        let terminal_mask = get_valid_action_mask(&state);
        assert!(!terminal_mask.contains(&true));
    }

    #[test]
    fn test_sequential_halving_visits() {
        let evaluator = MockEvaluator;
        let state = GameStateExt::new(Some([0, 1, 2]), 0, 0, 6, 0);

        let (answer_tx, answer_rx) = crossbeam_channel::unbounded();

        let mut policy_probs = vec![0.0; 288];
        let mask = get_valid_action_mask(&state);
        let mut valid_count = 0;
        for i in 0..mask.len() {
            if mask[i] {
                policy_probs[i] = 1.0;
                valid_count += 1;
            }
        }
        for p in policy_probs.iter_mut() {
            *p /= valid_count as f32;
        }

        let simulations = 50;
        let k = 8;

        let (_best_action, visits, _value, _tree) = mcts_search(MctsParams {
            root_cache_index: 0,
            maximum_allowed_nodes_in_search_tree: 50000,
            worker_id: 0,
            raw_policy_probabilities: &policy_probs,
            game_state: &state,
            total_simulations: simulations,
            max_gumbel_k_samples: k,
            gumbel_noise_scale: 1.0,
            previous_tree: None,
            last_executed_action: None,
            neural_evaluator: &evaluator,
            evaluation_request_transmitter: answer_tx,
            evaluation_response_receiver: &answer_rx,
            active_flag: std::sync::Arc::new(std::sync::RwLock::new(true)),
            _seed: None,
        })
        .unwrap();

        let total_visits: i32 = visits.values().sum();
        assert!(
            (50..=200).contains(&total_visits),
            "Total visits should scale relative to requested simulations. Was: {}",
            total_visits
        );

        let mut visit_counts: Vec<i32> = visits.values().cloned().collect();
        visit_counts.sort_unstable_by(|a, b| b.cmp(a));
        assert!(visit_counts[0] > 8, "Sequential Halving correctly concentrates visits on top candidates, even if uniform prior.");
    }

    #[test]
    fn test_terminal_state_value_masking() {
        let evaluator = CustomEvaluator {
            reward: 1.0,
            value: 0.5,
        };
        let state = GameStateExt::new(Some([0, 1, 2]), 0, 0, 6, 0);
        let (answer_tx, answer_rx) = crossbeam_channel::unbounded();
        let mut policy_probs = vec![0.0; 288];
        let mask = get_valid_action_mask(&state);
        for i in 0..mask.len() {
            if mask[i] {
                policy_probs[i] = 1.0;
            }
        }
        let (_best_action, _visits, _value, tree) = mcts_search(MctsParams {
            root_cache_index: 0,
            maximum_allowed_nodes_in_search_tree: 50000,
            worker_id: 0,
            raw_policy_probabilities: &policy_probs,
            game_state: &state,
            total_simulations: 10,
            max_gumbel_k_samples: 8,
            gumbel_noise_scale: 1.0,
            previous_tree: None,
            last_executed_action: None,
            neural_evaluator: &evaluator,
            evaluation_request_transmitter: answer_tx,
            evaluation_response_receiver: &answer_rx,
            active_flag: std::sync::Arc::new(std::sync::RwLock::new(true)),
            _seed: None,
        })
        .unwrap();

        let root = &tree.arena[tree.root_index];
        let mut checked_any = false;
        let mut child_idx = root.first_child;
        while child_idx != u32::MAX {
            if tree.arena[child_idx as usize].visits == 1 {
                let child = &tree.arena[child_idx as usize];
                let expected = child.value();
                assert!(
                    (expected - 0.5).abs() < 1e-5 || (expected + 1.0).abs() < 1e-5 || (expected - 1.0).abs() < 1e-5,
                    "Child with 1 visit should contain expected network value or terminal mask! Found: {}",
                    expected
                );
                checked_any = true;
            }
            child_idx = tree.arena[child_idx as usize].next_sibling;
        }
        assert!(checked_any, "No children were evaluated.");
    }

    #[test]
    fn test_gumbel_distribution() {
        let mut rng = rand::thread_rng();
        let mut sum = 0.0;
        let n = 10_000;
        for _ in 0..n {
            let uniform_random_sample: f32 = rng.gen_range(1e-6..=(1.0 - 1e-6));
            let gumbel_noise_value = -(-(uniform_random_sample.ln())).ln();
            sum += gumbel_noise_value;
        }
        let mean = sum / (n as f32);
        assert!((mean - 0.5772).abs() < 0.05, "Gumbel mean off: {}", mean);
    }
}
