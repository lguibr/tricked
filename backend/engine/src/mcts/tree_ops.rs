use crate::core::board::GameStateExt;
use crate::mcts::evaluator::{EvaluationRequest, EvaluationResponse, NetworkEvaluator};
use crate::mcts::tree::{allocate_node, MctsTree};
use crate::node::{select_child, LatentNode};
use std::sync::atomic::Ordering;

#[hotpath::measure]
#[allow(clippy::too_many_arguments)]
pub fn expand_and_evaluate_candidates(
    tree: &mut MctsTree,
    candidate_actions: &[i32],
    visits_per_candidate: usize,
    game_state: &GameStateExt,
    neural_evaluator: &dyn NetworkEvaluator,
    worker_id: usize,
    active_flag: &std::sync::Arc<std::sync::atomic::AtomicBool>,
    discount_factor: f32,
) -> Result<(), String> {
    let root_index = tree.root_index;

    let total_visits_needed = visits_per_candidate * candidate_actions.len();
    let mut visits_completed = 0;
    let mut in_flight_requests = 0;
    let max_in_flight = 16;

    const EMPTY_IN_FLIGHT: Option<(
        usize,
        arrayvec::ArrayVec<usize, 256>,
        std::sync::Arc<crate::mcts::mailbox::AtomicMailbox<EvaluationResponse>>,
    )> = None;
    let mut in_flight_paths = [EMPTY_IN_FLIGHT; 16];
    let mut eval_batch = arrayvec::ArrayVec::<EvaluationRequest, 256>::new();

    while visits_completed < total_visits_needed {
        while in_flight_requests < max_in_flight
            && (visits_completed + in_flight_requests) < total_visits_needed
        {
            let candidate_idx = (visits_completed + in_flight_requests) % candidate_actions.len();
            let candidate_action = candidate_actions[candidate_idx];

            let (search_path, leaf_node_index, successfully_traversed) =
                traverse_tree_to_leaf(&tree.arena, root_index, candidate_action);

            if !successfully_traversed {
                visits_completed += 1;
                continue;
            }

            for &node_idx in &search_path {
                tree.arena[node_idx]
                    .in_flight
                    .fetch_add(1, Ordering::SeqCst);
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
            let prev_idx = tree.arena[parent_index]
                .hidden_state_index
                .load(Ordering::Relaxed);
            let new_idx = tree.allocated_cache_slots as u32;
            tree.allocated_cache_slots += 1;

            tree.arena[leaf_node_index]
                .hidden_state_index
                .store(new_idx, Ordering::SeqCst);

            let mailbox = std::sync::Arc::new(crate::mcts::mailbox::AtomicMailbox::new());
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
                generation: tree.current_generation,
                worker_id,
                parent_cache_index: prev_idx,
                leaf_cache_index: new_idx,
                mailbox: mailbox.clone(),
            };

            for slot in &mut in_flight_paths {
                if slot.is_none() {
                    *slot = Some((leaf_node_index, search_path, mailbox));
                    break;
                }
            }
            eval_batch.push(evaluation_request);
            in_flight_requests += 1;
        }

        if !eval_batch.is_empty() {
            let current_batch = std::mem::replace(&mut eval_batch, arrayvec::ArrayVec::new());
            if let Err(error) = neural_evaluator.send_batch(current_batch) {
                return Err(format!("Failed sending eval request: {}", error));
            }
        }

        if in_flight_requests > 0 {
            neural_evaluator.mark_blocked();

            let backoff = crossbeam::utils::Backoff::new();
            let mut popped = false;
            let start_wait = std::time::Instant::now();

            loop {
                if !active_flag.load(std::sync::atomic::Ordering::Relaxed) {
                    return Err("Training stopped".to_string());
                }

                if start_wait.elapsed().as_millis() > 5000 {
                    active_flag.store(false, Ordering::SeqCst);
                    return Err("CRITICAL: GPU Inference Timeout Exceeded (5000ms). Engine Deadlock prevented. Aborting.".to_string());
                }

                for index in 0..in_flight_paths.len() {
                    let mut ready_resp = None;
                    if let Some((_, _, ref mailbox)) = in_flight_paths[index] {
                        if mailbox.is_ready() {
                            ready_resp = Some(mailbox.extract());
                        }
                    }

                    if let Some(resp) = ready_resp {
                        neural_evaluator.mark_unblocked();
                        apply_evaluation_response(
                            tree,
                            resp,
                            &mut in_flight_paths,
                            discount_factor,
                        )?;
                        visits_completed += 1;
                        in_flight_requests -= 1;
                        popped = true;
                    }
                }

                if popped {
                    break;
                }

                if backoff.is_completed() {
                    std::thread::sleep(std::time::Duration::from_micros(50));
                } else {
                    backoff.snooze();
                }
            }
        }
    }
    Ok(())
}

#[hotpath::measure]
pub fn traverse_tree_to_leaf(
    arena: &[LatentNode],
    root_index: usize,
    candidate_action: i32,
) -> (arrayvec::ArrayVec<usize, 256>, usize, bool) {
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

    while arena[current_node_index]
        .is_topologically_expanded
        .load(Ordering::Relaxed)
    {
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
#[allow(clippy::type_complexity)]
fn apply_evaluation_response(
    tree: &mut MctsTree,
    evaluation_response: EvaluationResponse,
    in_flight_paths: &mut [Option<(
        usize,
        arrayvec::ArrayVec<usize, 256>,
        std::sync::Arc<crate::mcts::mailbox::AtomicMailbox<EvaluationResponse>>,
    )>],
    discount_factor: f32,
) -> Result<(), String> {
    let leaf_node_index = evaluation_response.node_index;
    let mut search_path = None;
    for slot in in_flight_paths.iter_mut() {
        if let Some((node_idx, _, _)) = slot {
            if *node_idx == leaf_node_index {
                search_path = slot.take().map(|(_, path, _)| path);
                break;
            }
        }
    }
    let search_path = search_path.expect(
        "Engine Error: Received inference response for node not found in in-flight registry.",
    );

    let cvp = evaluation_response.value_prefix;
    tree.arena[leaf_node_index]
        .cumulative_value_prefix
        .store(cvp, Ordering::SeqCst);

    let depth = (search_path.len() - 1) / 2;
    if depth > 0 {
        let parent_idx = search_path[search_path.len() - 3];
        let parent_cvp = tree.arena[parent_idx]
            .cumulative_value_prefix
            .load(Ordering::Relaxed);
        let discount = discount_factor.powi((depth - 1) as i32);
        let step_reward = (cvp - parent_cvp) / discount;
        tree.arena[leaf_node_index]
            .value_prefix
            .store(step_reward, Ordering::SeqCst);
    } else {
        tree.arena[leaf_node_index]
            .value_prefix
            .store(0.0, Ordering::SeqCst);
    }
    tree.arena[leaf_node_index]
        .is_topologically_expanded
        .store(true, Ordering::SeqCst);

    let mut prev_child = u32::MAX;
    let mut first_child = u32::MAX;

    for (action_index, &probability) in evaluation_response
        .child_prior_probabilities_tensor
        .iter()
        .enumerate()
    {
        let new_node_index = allocate_node(tree, probability, action_index as i16);
        if new_node_index == u32::MAX {
            break;
        }
        if first_child == u32::MAX {
            first_child = new_node_index;
        } else {
            tree.arena[prev_child as usize]
                .next_sibling
                .store(new_node_index, Ordering::SeqCst);
        }
        prev_child = new_node_index;
    }
    tree.arena[leaf_node_index]
        .first_child
        .store(first_child, Ordering::SeqCst);

    let mut backprop_value = evaluation_response.value;

    for index in (0..search_path.len()).step_by(2).rev() {
        let node_index = search_path[index];
        tree.arena[node_index]
            .in_flight
            .fetch_sub(1, Ordering::SeqCst);
        tree.arena[node_index].visits.fetch_add(1, Ordering::SeqCst);
        tree.arena[node_index]
            .value_sum
            .fetch_add(backprop_value, Ordering::SeqCst);
        backprop_value = tree.arena[node_index].value_prefix.load(Ordering::Relaxed)
            + discount_factor * backprop_value;
    }
    Ok(())
}
