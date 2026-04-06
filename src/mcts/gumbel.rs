use crate::core::board::GameStateExt;
use crate::mcts::evaluator::{EvaluationResponse, NetworkEvaluator};
use crate::mcts::tree::MctsTree;
use crate::mcts::tree_ops::expand_and_evaluate_candidates;
use crate::node::LatentNode;
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoroshiro128PlusPlus;
use std::collections::HashMap;
use std::sync::atomic::Ordering;

thread_local! {
    static FAST_RNG: std::cell::RefCell<Xoroshiro128PlusPlus> =
        std::cell::RefCell::new(Xoroshiro128PlusPlus::seed_from_u64(rand::thread_rng().gen()));
}

#[hotpath::measure]
pub fn calculate_dynamic_k_samples(
    max_gumbel_k_samples: usize,
    valid_action_count: usize,
) -> usize {
    let empty_board_density = 1.0 - (valid_action_count as f32 / 288.0);
    let mut k_dynamic_samples =
        4i32 + ((max_gumbel_k_samples as f32 - 4.0) * empty_board_density) as i32;
    if k_dynamic_samples < 2 {
        k_dynamic_samples = 2;
    }
    (k_dynamic_samples as usize).min(valid_action_count)
}

#[hotpath::measure]
pub(crate) fn inject_gumbel_noise(
    arena: &[LatentNode],
    root_index: usize,
    candidate_actions: &[i32],
    normalized_probabilities: &[f32],
    gumbel_noise_scale: f32,
) -> Vec<f32> {
    let mut gumbel_noisy_logits = vec![f32::NEG_INFINITY; 288];

    FAST_RNG.with(|rng| {
        let mut r = rng.borrow_mut();
        for &action_index in candidate_actions {
            let action_usize = action_index as usize;
            let child_index = arena[root_index].get_child(arena, action_index);

            if child_index != usize::MAX {
                let uniform_random_sample: f32 = (*r).gen_range(1e-6..=(1.0 - 1e-6));
                let gumbel_noise_value = -(-(uniform_random_sample.ln())).ln();
                assert!(!gumbel_noise_value.is_nan(), "Gumbel noise is NaN");

                arena[child_index]
                    .gumbel_noise
                    .store(gumbel_noise_value, Ordering::SeqCst);
                let log_probability = (normalized_probabilities[action_usize] + 1e-8).ln();
                gumbel_noisy_logits[action_usize] =
                    log_probability + (gumbel_noise_value * gumbel_noise_scale);
            }
        }
    });
    gumbel_noisy_logits
}

#[hotpath::measure]
#[allow(clippy::too_many_arguments)]
pub fn execute_sequential_halving(
    tree: &mut MctsTree,
    candidate_actions: &mut Vec<i32>,
    total_simulations: usize,
    gumbel_noisy_logits: &[f32],
    game_state: &GameStateExt,
    neural_evaluator: &dyn NetworkEvaluator,
    worker_id: usize,
    evaluation_request_transmitter: crossbeam_channel::Sender<EvaluationResponse>,
    evaluation_response_receiver: &crossbeam_channel::Receiver<EvaluationResponse>,
    active_flag: &std::sync::Arc<std::sync::atomic::AtomicBool>,
    training_steps: usize,
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
            training_steps,
        );

        remaining_simulations =
            remaining_simulations.saturating_sub(visits_per_candidate * current_candidate_count);
        remaining_phases -= 1;
    }

    Ok(())
}

#[hotpath::measure]
pub fn prune_candidates(
    arena: &[LatentNode],
    root_index: usize,
    candidate_actions: &mut Vec<i32>,
    gumbel_noisy_logits: &[f32],
    training_steps: usize,
) {
    let decay = (1.0 - (training_steps as f32 / 100_000.0)).max(0.1);
    let base_scale = 50.0 * decay;

    let mut candidates_with_nodes: Vec<(i32, usize)> = candidate_actions
        .iter()
        .map(|&a| (a, arena[root_index].get_child(arena, a)))
        .filter(|&(_, idx)| idx != usize::MAX)
        .collect();

    candidates_with_nodes.sort_by(|&(action_a, index_a), &(action_b, index_b)| {
        let node_a = &arena[index_a];
        let node_b = &arena[index_b];

        let q_value_a = node_a.value_prefix.load(Ordering::Relaxed) + 0.99 * node_a.value();
        let q_value_b = node_b.value_prefix.load(Ordering::Relaxed) + 0.99 * node_b.value();

        let exploration_scale_a = base_scale / ((node_a.visits.load(Ordering::Relaxed) + 1) as f32);
        let score_a = gumbel_noisy_logits[action_a as usize] + (exploration_scale_a * q_value_a);

        let exploration_scale_b = base_scale / ((node_b.visits.load(Ordering::Relaxed) + 1) as f32);
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
pub fn compute_final_action_distribution(
    tree: MctsTree,
    valid_action_mask: [bool; 288],
    candidate_actions: Vec<i32>,
    gumbel_noisy_logits: Vec<f32>,
    training_steps: usize,
) -> Result<(i32, HashMap<i32, i32>, f32, MctsTree), String> {
    let arena = &tree.arena;
    let root_index = tree.root_index;
    let mut evaluated_candidates = Vec::new();
    for (action_index, &is_valid) in valid_action_mask.iter().enumerate() {
        let child_index = arena[root_index].get_child(arena, action_index as i32);
        if child_index != usize::MAX
            && arena[child_index].visits.load(Ordering::Relaxed) > 0
            && is_valid
        {
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
        let q_value = arena[child_index].value_prefix.load(Ordering::Relaxed)
            + 0.99 * arena[child_index].value();
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
        visit_distribution.insert(
            action_index,
            arena[child_index].visits.load(Ordering::Relaxed),
        );
    }

    let mut optimal_action = candidate_actions[0];
    let mut optimal_action_score = f32::NEG_INFINITY;

    let mut max_visit = 0;
    let mut sum_visit = 0;
    for &(_action_index, child_index) in &evaluated_candidates {
        let visits = arena[child_index].visits.load(Ordering::Relaxed);
        sum_visit += visits;
        if visits > max_visit {
            max_visit = visits;
        }
    }

    let decay = (1.0 - (training_steps as f32 / 100_000.0)).max(0.1);
    let base_scale = 50.0 * decay;
    let exploration_scale = (base_scale + max_visit as f32) / (sum_visit as f32 + 1e-8);

    for &(action_index, child_index) in &evaluated_candidates {
        let q_value = arena[child_index].value_prefix.load(Ordering::Relaxed)
            + 0.99 * arena[child_index].value();
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
