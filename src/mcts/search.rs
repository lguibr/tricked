use crate::core::board::GameStateExt;
use crate::mcts::gumbel::{
    calculate_dynamic_k_samples, compute_final_action_distribution, execute_sequential_halving,
    inject_gumbel_noise,
};
use crate::node::get_valid_action_mask;
use std::collections::HashMap;

use super::evaluator::{EvaluationResponse, NetworkEvaluator};
use super::tree::{expand_root_node, initialize_search_tree, MctsTree};

pub struct MctsParams<'a> {
    pub raw_policy_probabilities: &'a [f32],
    pub root_cache_index: u32,
    pub max_tree_nodes: u32,
    pub max_cache_slots: u32,
    pub worker_id: usize,
    pub game_state: &'a GameStateExt,
    pub total_simulations: usize,
    pub max_gumbel_k_samples: usize,
    pub gumbel_noise_scale: f32,
    pub training_steps: usize,
    pub neural_evaluator: &'a dyn NetworkEvaluator,
    pub evaluation_request_transmitter: crossbeam_channel::Sender<EvaluationResponse>,
    pub evaluation_response_receiver: &'a crossbeam_channel::Receiver<EvaluationResponse>,
    pub active_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    pub _seed: Option<u64>,
}

#[hotpath::measure]
pub fn mcts_search(params: MctsParams) -> Result<(i32, HashMap<i32, i32>, f32, MctsTree), String> {
    let MctsParams {
        raw_policy_probabilities,
        root_cache_index,
        max_tree_nodes,
        max_cache_slots,
        worker_id,
        game_state,
        total_simulations,
        max_gumbel_k_samples,
        gumbel_noise_scale,
        training_steps,
        neural_evaluator,
        evaluation_request_transmitter,
        evaluation_response_receiver,
        active_flag,
        _seed,
    } = params;
    let (normalized_probabilities, valid_mask, valid_actions) =
        normalize_policy_distributions(raw_policy_probabilities, game_state);

    if valid_actions.is_empty() {
        let tree = initialize_search_tree(max_tree_nodes, max_cache_slots, total_simulations);
        return Ok((-1, HashMap::new(), 0.0, tree));
    }

    let mut tree = initialize_search_tree(max_tree_nodes, max_cache_slots, total_simulations);
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
        &tree.arena.0,
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
        training_steps,
    )?;

    compute_final_action_distribution(
        tree,
        valid_mask,
        candidate_actions,
        gumbel_noisy_logits,
        training_steps,
    )
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
