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
