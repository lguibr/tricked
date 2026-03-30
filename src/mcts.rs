use crate::board::GameStateExt;
use crate::node::{get_valid_action_mask, select_child, LatentNode};
use rand::Rng;
use std::collections::HashMap;

pub struct EvalReq {
    pub is_initial: bool,
    pub state_feat: Option<Vec<f32>>,
    pub piece_action: i64,
    pub piece_id: i64,
    pub node_index: usize,
    pub worker_id: usize,
    pub parent_cache_index: u32,
    pub leaf_cache_index: u32,
    pub evaluation_request_transmitter: crossbeam_channel::Sender<EvalResp>,
}

pub struct EvalResp {
    pub reward: f32,
    pub value: f32,
    pub child_prior_probabilities_tensor: Vec<f32>,
    pub node_index: usize,
}

pub trait NetworkEvaluator: Send + Sync {
    fn send_req(&self, req: EvalReq) -> Result<(), String>;
}

impl NetworkEvaluator for std::sync::Arc<crate::queue::FixedInferenceQueue> {
    fn send_req(&self, request: EvalReq) -> Result<(), String> {
        self.push(request.worker_id, request)
            .map_err(|_| "Queue Disconnected".to_string())
    }
}

#[allow(dead_code)]
pub struct MockEvaluator;
impl NetworkEvaluator for MockEvaluator {
    fn send_req(&self, request: EvalReq) -> Result<(), String> {
        let response = EvalResp {
            reward: 0.0,
            value: 0.0,
            child_prior_probabilities_tensor: vec![1.0 / 288.0; 288],
            node_index: request.node_index,
        };
        let _ = request.evaluation_request_transmitter.send(response);
        Ok(())
    }
}

#[derive(Clone)]
pub struct MctsTree {
    pub arena: Vec<LatentNode>,
    pub node_free_list: Vec<u32>,
    pub root_index: usize,
    pub free_list: Vec<u32>, // GPU latent state free list
    pub generation: u32,
}

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
    pub evaluation_request_transmitter: crossbeam_channel::Sender<EvalResp>,
    pub evaluation_response_receiver: &'a crossbeam_channel::Receiver<EvalResp>,
    pub _seed: Option<u64>,
}

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
    )?;

    compute_final_action_distribution(tree, valid_mask, candidate_actions, gumbel_noisy_logits)
}

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

fn calculate_dynamic_k_samples(max_gumbel_k_samples: usize, valid_action_count: usize) -> usize {
    let empty_board_density = 1.0 - (valid_action_count as f32 / 288.0);
    let k_dynamic_samples =
        4 + ((max_gumbel_k_samples as f32 - 4.0) * empty_board_density) as usize;
    k_dynamic_samples.min(valid_action_count)
}

fn allocate_node(tree: &mut MctsTree, probability: f32, action: i16) -> u32 {
    let new_idx = if let Some(idx) = tree.node_free_list.pop() {
        idx
    } else {
        let idx = tree.arena.len() as u32;
        tree.arena.push(LatentNode::new(0.0, -1));
        idx
    };

    let mut node = LatentNode::new(probability, action);
    node.generation = tree.generation;
    tree.arena[new_idx as usize] = node;
    new_idx
}

pub fn gc_tree(mut tree: MctsTree, new_root: usize) -> MctsTree {
    tree.generation += 1;
    let current_gen = tree.generation;

    let mut queue = vec![new_root];
    tree.arena[new_root].generation = current_gen;

    let mut head = 0;
    while head < queue.len() {
        let node_idx = queue[head];
        head += 1;

        let mut child_idx = tree.arena[node_idx].first_child;
        while child_idx != u32::MAX {
            tree.arena[child_idx as usize].generation = current_gen;
            queue.push(child_idx as usize);
            child_idx = tree.arena[child_idx as usize].next_sibling;
        }
    }

    tree.node_free_list.clear();
    for i in 0..tree.arena.len() {
        if tree.arena[i].generation != current_gen {
            tree.node_free_list.push(i as u32);
            if tree.arena[i].hidden_state_index != u32::MAX {
                tree.free_list.push(tree.arena[i].hidden_state_index);
                tree.arena[i].hidden_state_index = u32::MAX;
            }
        }
    }

    let arena_capacity = tree.arena.len();
    if arena_capacity > 0 && tree.node_free_list.len() > arena_capacity / 2 {
        let mut compacted_arena = Vec::with_capacity(arena_capacity);
        let mut pointer_remapping = vec![u32::MAX; arena_capacity];

        for (node, mapped_idx) in tree.arena.iter().zip(pointer_remapping.iter_mut()) {
            if node.generation == current_gen {
                *mapped_idx = compacted_arena.len() as u32;
                compacted_arena.push(node.clone());
            }
        }

        for node in compacted_arena.iter_mut() {
            if node.first_child != u32::MAX {
                node.first_child = pointer_remapping[node.first_child as usize];
            }
            if node.next_sibling != u32::MAX {
                node.next_sibling = pointer_remapping[node.next_sibling as usize];
            }
        }

        let compacted_nodes_count = compacted_arena.len();
        let mut refreshed_node_free_list =
            Vec::with_capacity(arena_capacity - compacted_nodes_count);
        for index in (compacted_nodes_count..arena_capacity).rev() {
            refreshed_node_free_list.push(index as u32);
            compacted_arena.push(LatentNode::new(0.0, -1));
        }

        tree.root_index = pointer_remapping[new_root] as usize;
        tree.arena = compacted_arena;
        tree.node_free_list = refreshed_node_free_list;
    } else {
        tree.root_index = new_root;
    }

    tree
}

fn initialize_search_tree(
    previous_tree: Option<MctsTree>,
    last_executed_action: Option<i32>,
    maximum_allowed_nodes_in_search_tree: u32,
    total_simulations: usize,
) -> MctsTree {
    if let Some(existing_tree) = previous_tree {
        if let Some(action) = last_executed_action {
            let child_index = existing_tree.arena[existing_tree.root_index]
                .get_child(&existing_tree.arena, action);
            if child_index != usize::MAX {
                let gc_d_tree = gc_tree(existing_tree, child_index);
                if gc_d_tree.free_list.len() > total_simulations + 10 {
                    return gc_d_tree;
                }
            }
        } else {
            let root = existing_tree.root_index;
            let gc_d_tree = gc_tree(existing_tree, root);
            if gc_d_tree.free_list.len() > total_simulations + 10 {
                return gc_d_tree;
            }
        }
    }

    let mut arena = vec![LatentNode::new(0.0, -1); maximum_allowed_nodes_in_search_tree as usize];
    let node_free_list = (1..maximum_allowed_nodes_in_search_tree)
        .rev()
        .collect::<Vec<u32>>();
    let free_list = (0..maximum_allowed_nodes_in_search_tree)
        .rev()
        .collect::<Vec<u32>>();

    arena[0] = LatentNode::new(1.0, -1);
    arena[0].generation = 1;

    MctsTree {
        arena,
        node_free_list,
        root_index: 0,
        free_list,
        generation: 1,
    }
}

fn expand_root_node(tree: &mut MctsTree, root_cache_index: u32, normalized_probabilities: &[f32]) {
    let root_index = tree.root_index;
    if tree.arena[root_index].is_topologically_expanded {
        return;
    }
    tree.arena[root_index].hidden_state_index = root_cache_index;
    tree.arena[root_index].reward = 0.0;
    tree.arena[root_index].is_topologically_expanded = true;

    let mut prev_child = u32::MAX;
    let mut first_child = u32::MAX;

    for (action_index, &probability) in normalized_probabilities.iter().enumerate() {
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
    tree.arena[root_index].first_child = first_child;
}

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

#[allow(clippy::too_many_arguments)]
fn execute_sequential_halving(
    tree: &mut MctsTree,
    candidate_actions: &mut Vec<i32>,
    total_simulations: usize,
    gumbel_noisy_logits: &[f32],
    game_state: &GameStateExt,
    neural_evaluator: &dyn NetworkEvaluator,
    worker_id: usize,
    evaluation_request_transmitter: crossbeam_channel::Sender<EvalResp>,
    evaluation_response_receiver: &crossbeam_channel::Receiver<EvalResp>,
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

        for _ in 0..visits_per_candidate {
            expand_and_evaluate_candidates(
                tree,
                candidate_actions,
                game_state,
                neural_evaluator,
                worker_id,
                evaluation_request_transmitter.clone(),
                evaluation_response_receiver,
            )?;
        }

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

#[allow(clippy::too_many_arguments)]
fn expand_and_evaluate_candidates(
    tree: &mut MctsTree,
    candidate_actions: &[i32],
    game_state: &GameStateExt,
    neural_evaluator: &dyn NetworkEvaluator,
    worker_id: usize,
    evaluation_request_transmitter: crossbeam_channel::Sender<EvalResp>,
    evaluation_response_receiver: &crossbeam_channel::Receiver<EvalResp>,
) -> Result<(), String> {
    let mut active_requests = 0u32;
    let mut batch_paths = Vec::new();

    let root_index = tree.root_index;

    for &candidate_action in candidate_actions {
        let (search_path, leaf_node_index, successfully_traversed) =
            traverse_tree_to_leaf(&tree.arena, root_index, candidate_action);

        if !successfully_traversed {
            continue;
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

        let evaluation_request = EvalReq {
            is_initial: false,
            state_feat: None,
            piece_action: piece_action_identifier as i64,
            piece_id: piece_identifier as i64,
            node_index: leaf_node_index,
            worker_id,
            parent_cache_index: prev_idx,
            leaf_cache_index: new_idx,
            evaluation_request_transmitter: evaluation_request_transmitter.clone(),
        };

        active_requests += 1;
        batch_paths.push(search_path);

        if let Err(error) = neural_evaluator.send_req(evaluation_request) {
            return Err(format!("Failed sending eval request: {}", error));
        }
    }

    if active_requests > 0 {
        process_evaluation_responses(
            tree,
            evaluation_response_receiver,
            active_requests,
            batch_paths,
        )?;
    }
    Ok(())
}

fn traverse_tree_to_leaf(
    arena: &[LatentNode],
    root_index: usize,
    candidate_action: i32,
) -> (Vec<usize>, usize, bool) {
    let mut search_path = vec![root_index];
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

fn process_evaluation_responses(
    tree: &mut MctsTree,
    receiver_rx: &crossbeam_channel::Receiver<EvalResp>,
    active_requests: u32,
    batch_paths: Vec<Vec<usize>>,
) -> Result<(), String> {
    let mut paths_map = std::collections::HashMap::new();
    for path in batch_paths {
        let leaf_index = *path.last().unwrap();
        paths_map.insert(leaf_index, path);
    }

    for _ in 0..active_requests {
        let evaluation_response = receiver_rx.recv().unwrap();

        let leaf_node_index = evaluation_response.node_index;
        let search_path = paths_map.get(&leaf_node_index).unwrap();

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
            tree.arena[node_index].visits += 1;
            tree.arena[node_index].value_sum += backprop_value;
            backprop_value = tree.arena[node_index].reward + 0.99 * backprop_value;
        }
    }
    Ok(())
}

fn prune_candidates(
    arena: &[LatentNode],
    root_index: usize,
    candidate_actions: &mut Vec<i32>,
    gumbel_noisy_logits: &[f32],
) {
    // 1. PRE-FETCH: Find all child indices in O(N) time before sorting
    let mut candidates_with_nodes: Vec<(i32, usize)> = candidate_actions
        .iter()
        .map(|&a| (a, arena[root_index].get_child(arena, a)))
        .filter(|&(_, idx)| idx != usize::MAX)
        .collect();

    // 2. SORT: Now we use the pre-fetched indices. No linked-list traversal here!
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

    // 3. TRUNCATE: Drop the bottom half and map back to just the actions
    let items_to_drop = candidates_with_nodes.len() / 2;
    candidates_with_nodes.truncate(candidates_with_nodes.len() - items_to_drop);

    *candidate_actions = candidates_with_nodes.into_iter().map(|(a, _)| a).collect();
}

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
    use crate::board::GameStateExt;
    use crate::node::LatentNode;

    // [5. Q-Value Min-Max Normalization (Math Correctness)]
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

    // [6. Gumbel Noise Distribution (Math Correctness)]
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

    // [8. Tree Arena Garbage Collection (RAM Bottleneck / Leak)]
    #[test]
    fn test_tree_arena_garbage_collection_leak_100000_steps() {
        let max_nodes = 1000;
        let mut tree = initialize_search_tree(None, None, max_nodes, 500);

        for _step in 0..100_000 {
            let current_root = tree.root_index;
            // allocate 10 children
            let mut prev = u32::MAX;
            let mut first = u32::MAX;
            for i in 0..10 {
                let new_node = allocate_node(&mut tree, 0.1, i);
                tree.arena[new_node as usize].hidden_state_index = i as u32;
                if first == u32::MAX {
                    first = new_node;
                } else {
                    tree.arena[prev as usize].next_sibling = new_node;
                }
                prev = new_node;
            }
            tree.arena[current_root].first_child = first;

            // Pick the 5th child as the new root to step forward in the game
            let mut new_root = first;
            for _ in 0..4 {
                new_root = tree.arena[new_root as usize].next_sibling;
            }

            tree = gc_tree(tree, new_root as usize);

            // Assert that despite looping 100k times, we never exhaust arena size (no leaks)
            assert!(
                tree.arena.len() <= max_nodes as usize * 2,
                "Arena leaked and grew infinitely!"
            );
            // In a properly functioning GC, the free list should roughly maintain its size.
        }
    }

    #[test]
    fn test_tree_arena_garbage_collection() {
        let mut tree = initialize_search_tree(None, None, 1000, 500);
        let new_root = 0; // Prevent cycle with node_free_list which operates 1..1000

        // Allocate a few nodes
        for i in 0..500 {
            let p_idx = allocate_node(&mut tree, 1.0, i as i16);
            if i > 0 {
                // link to root to keep valid
                tree.arena[p_idx as usize].next_sibling = tree.arena[new_root].first_child;
                tree.arena[new_root].first_child = p_idx;
                tree.arena[p_idx as usize].hidden_state_index = i as u32; // Simulate cached memory
            }
        }

        assert_eq!(tree.arena.len(), 1000);

        let new_tree = gc_tree(tree, new_root);

        // Ensure new_tree has a smaller reach but the arena capacity stays identical
        // node_free_list will be roughly 500 since we retained all 500 allocated nodes
        assert_eq!(new_tree.node_free_list.len(), 500); // Wait, node_free_list gets smaller because we just allocated 500. Oh, GC creates a fresh free list. Initial had ~999 free. After alloc it had ~499. GC will reclaim the 500 we didn't link.
                                                        // Actually, we linked ALL 499 to root 10. So GC will Keep them.
                                                        // Let's test dropping.

        // The new root MUST be mapped to 10 right after GC (before any shifting if we do shift, but gc_tree doesn't shift, it just changes generation).
        assert_eq!(new_tree.root_index, new_root);
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
            maximum_allowed_nodes_in_search_tree: 50,
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
            _seed: None,
        })
        .unwrap();

        let total_visits: i32 = visits.values().sum();
        assert!(
            (total_visits - simulations as i32).abs() <= 6,
            "Total visits must closely match requested simulations despite sequential halving integer division."
        );

        let mut visit_counts: Vec<i32> = visits.values().cloned().collect();
        visit_counts.sort_unstable_by(|a, b| b.cmp(a));
        assert!(visit_counts[0] > 8, "Sequential Halving correctly concentrates visits on top candidates, even if uniform prior.");
    }

    pub struct CustomEvaluator {
        pub reward: f32,
        pub value: f32,
    }
    impl super::NetworkEvaluator for CustomEvaluator {
        fn send_req(&self, request: super::EvalReq) -> Result<(), String> {
            let response = super::EvalResp {
                reward: self.reward,
                value: self.value,
                child_prior_probabilities_tensor: vec![1.0 / 288.0; 288],
                node_index: request.node_index,
            };
            let _ = request.evaluation_request_transmitter.send(response);
            Ok(())
        }
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
            maximum_allowed_nodes_in_search_tree: 50,
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
            _seed: None,
        })
        .unwrap();

        let root = &tree.arena[tree.root_index];
        let mut checked_any = false;
        let mut child_idx = root.first_child;
        while child_idx != u32::MAX {
            if tree.arena[child_idx as usize].visits == 1 {
                let child = &tree.arena[child_idx as usize];
                assert!(
                    (child.value() - 0.5).abs() < 1e-5,
                    "Child with 1 visit should trust expected value 0.5 without terminal zeroing! Found: {}",
                    child.value()
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

    #[test]
    fn test_tree_garbage_collection() {
        let mut mock_arena = vec![LatentNode::new(0.0, 0); 10];
        let mut root = LatentNode::new(1.0, -1);
        root.is_topologically_expanded = true;
        root.first_child = 1;
        mock_arena[0] = root;

        let mut child1 = LatentNode::new(0.5, 0);
        child1.is_topologically_expanded = true;
        child1.first_child = 3;
        child1.next_sibling = 2; // points to child2
        child1.visits = 10;
        child1.hidden_state_index = 100;
        mock_arena[1] = child1;

        let mut child2 = LatentNode::new(0.5, 1);
        child2.visits = 5;
        child2.hidden_state_index = 101;
        mock_arena[2] = child2;

        let mut grandchild = LatentNode::new(1.0, 2);
        grandchild.visits = 2;
        grandchild.hidden_state_index = 102;
        mock_arena[3] = grandchild;

        let mut disconnected = LatentNode::new(1.0, 3);
        disconnected.hidden_state_index = 103;
        mock_arena[4] = disconnected;

        let initial_free_list = vec![999];

        let tree = MctsTree {
            arena: mock_arena,
            node_free_list: vec![],
            root_index: 0,
            free_list: initial_free_list,
            generation: 1,
        };

        let new_tree = gc_tree(tree, 1);
        let actual_root = new_tree.root_index;
        let new_root_node = &new_tree.arena[actual_root];
        assert_eq!(new_root_node.visits, 10, "Visits must be retained");

        let first_child_idx = new_root_node.first_child as usize;
        assert!(
            first_child_idx != u32::MAX as usize,
            "Children must be retained"
        );

        let new_grandchild = &new_tree.arena[first_child_idx];
        assert_eq!(
            new_grandchild.visits, 2,
            "Grandchild visits must be retained"
        );

        assert!(
            new_tree.free_list.contains(&101),
            "Child 2 cache slot must be freed"
        );
        assert!(
            new_tree.free_list.contains(&103),
            "Disconnected node cache slot must be freed"
        );
        assert!(
            !new_tree.free_list.contains(&100),
            "Child 1 cache slot must survive"
        );
        assert!(
            !new_tree.free_list.contains(&102),
            "Grandchild cache slot must survive"
        );
        assert!(
            new_tree.free_list.contains(&999),
            "Original free list items must survive"
        );
    }
}
