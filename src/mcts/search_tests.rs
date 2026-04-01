#[cfg(test)]
mod tests {
    use crate::core::board::GameStateExt;
    use crate::mcts::evaluator::{CustomEvaluator, MockEvaluator};
    use crate::mcts::gumbel::inject_gumbel_noise;
    use crate::mcts::{mcts_search, MctsParams};
    use crate::node::get_valid_action_mask;
    use crate::node::LatentNode;
    use rand::Rng;

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
        let mut mock_arena = vec![LatentNode::new(0.0, 0, 0)];
        mock_arena.push(LatentNode::new(0.33, 1, 0));
        mock_arena.push(LatentNode::new(0.33, 2, 0));
        mock_arena.push(LatentNode::new(0.34, 3, 0));
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
                let log_prob = (probs[*action as usize] + 1e-8_f32).ln();
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
