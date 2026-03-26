use tricked_engine::board::GameStateExt;
use tricked_engine::features::extract_feature_native;

#[test]
fn test_mcts_root_expansion() {
    let state = GameStateExt::new(None, 0, 0, 6, 0);
    let feat = extract_feature_native(&state, None, None, 6);

    assert_eq!(feat.len(), 20 * 96);
    assert_eq!(state.pieces_left, 3);
    assert!(!state.terminal);
}

#[test]
fn test_mcts_terminal_value_propagation() {
    use crossbeam_channel::bounded;
    use tricked_engine::mcts::{mcts_search, EvalReq, EvalResp, MctsTree, MockEvaluator};

    struct TerminalMockEvaluator;
    impl tricked_engine::mcts::NetworkEvaluator for TerminalMockEvaluator {
        fn send_req(&self, req: EvalReq) -> Result<(), String> {
            // Predict a noisy value of 0.5, but give a definitive terminal reward of 1.0!
            let resp = EvalResp {
                h_next: vec![0.0; 96],
                reward: 1.0,
                value: 0.5, // This is the noisy value that SHOULD be ignored!
                p_next: vec![1.0 / 288.0; 288],
            };
            let _ = req.tx.send(resp);
            Ok(())
        }
    }

    let evaluator = TerminalMockEvaluator;
    let state = GameStateExt::new(Some(vec![0]), 0, 0, 6, 0); // 1 move left
    let h0 = vec![0.0; 96];
    let mut policy_probs = vec![0.0; 288];
    policy_probs[0] = 1.0;

    let (_, _, root_value, _) = mcts_search(
        &h0,
        &policy_probs,
        &state,
        10,
        8,
        1.0,
        None,
        None,
        &evaluator,
        None,
    )
    .unwrap();

    // The root value should be purely derived from the 1.0 reward.
    // If the 0.5 value wasn't zeroed out, it would be 1.0 + 0.99 * 0.5 = 1.495
    assert!(
        root_value > 0.99 && root_value < 1.01,
        "Expected ~1.0 but got {}",
        root_value
    );
}
