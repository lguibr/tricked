use crate::constants::STANDARD_PIECES;
use crate::GameStateExt;

#[derive(Clone)]
pub struct LatentNode {
    pub visits: i32,
    pub value_sum: f32,
    pub prior: f32,
    pub hidden_state: Option<Vec<f32>>,
    pub reward: f32,
    pub gumbel_noise: f32,
    pub children: [usize; 288],
    pub is_expanded: bool,
}

impl LatentNode {
    pub fn new(prior: f32) -> Self {
        LatentNode {
            visits: 0,
            value_sum: 0.0,
            prior,
            hidden_state: None,
            reward: 0.0,
            gumbel_noise: 0.0,
            children: [usize::MAX; 288],
            is_expanded: false,
        }
    }

    pub fn value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / (self.visits as f32)
        }
    }
}

pub fn get_valid_action_mask(state: &GameStateExt) -> [bool; 288] {
    let mut validity_mask = [false; 288];
    if state.terminal {
        return validity_mask;
    }

    for slot_index in 0..3 {
        let piece_identifier = state.available[slot_index];
        if piece_identifier == -1 {
            continue;
        }
        for (rotation_index, _piece) in STANDARD_PIECES.iter().enumerate() {
            let structural_mask = STANDARD_PIECES[piece_identifier as usize][rotation_index];
            if structural_mask != 0 && (state.board & structural_mask) == 0 {
                let action_index = slot_index * 96 + rotation_index;
                validity_mask[action_index] = true;
            }
        }
    }
    validity_mask
}

pub fn select_child(arena: &[LatentNode], node_index: usize, is_root: bool) -> (i32, usize) {
    let parent_node = &arena[node_index];
    let mut highest_score = f32::NEG_INFINITY;
    let mut highest_action_index = -1;
    let mut highest_child_index = usize::MAX;

    for action_index in 0..288 {
        let child_index = parent_node.children[action_index];
        if child_index == usize::MAX {
            continue;
        }

        let child_node = &arena[child_index];
        let expected_q_value = if child_node.visits == 0 {
            parent_node.value()
        } else {
            child_node.reward + 0.99 * child_node.value()
        };

        let policy_logit = child_node.prior.max(1e-8).ln();
        let action_score = if is_root {
            let gumbel_noise_injected_logit = policy_logit + child_node.gumbel_noise;
            let exploration_scale = 50.0 / ((child_node.visits + 1) as f32);
            gumbel_noise_injected_logit + (exploration_scale * expected_q_value)
        } else {
            policy_logit + expected_q_value
        };

        if action_score > highest_score {
            highest_score = action_score;
            highest_action_index = action_index as i32;
            highest_child_index = child_index;
        }
    }

    (highest_action_index, highest_child_index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GameStateExt;

    #[test]
    fn test_latent_node() {
        let node = LatentNode::new(0.5);
        assert_eq!(node.value(), 0.0);

        let mut node2 = LatentNode::new(0.5);
        node2.visits = 2;
        node2.value_sum = 1.0;
        assert_eq!(node2.value(), 0.5);
    }

    #[test]
    fn test_valid_action_mask() {
        let mut state = GameStateExt::new(Some([0, 0, 0]), 0, 0, 6, 0);
        let mask = get_valid_action_mask(&state);
        assert_eq!(
            mask.len(),
            288,
            "Action mask must be exactly 288 elements long."
        );

        let valid_moves = mask.iter().filter(|&&b| b).count();
        assert!(
            valid_moves > 0 && valid_moves <= 288,
            "Must be valid moves on empty setup."
        );

        state.terminal = true;
        let terminal_mask = get_valid_action_mask(&state);
        assert!(
            !terminal_mask.contains(&true),
            "Terminal state must strictly return a mask of all false."
        );
    }

    #[test]
    fn test_select_child_puct_vs_gumbel() {
        let mut arena = vec![LatentNode::new(1.0)];
        arena[0].children[0] = 1;
        arena[0].children[1] = 2;

        let mut child_a = LatentNode::new(0.5);
        child_a.gumbel_noise = 100.0;

        let mut child_b = LatentNode::new(0.6);
        child_b.gumbel_noise = 0.0;

        arena.push(child_a);
        arena.push(child_b);

        let (internal_action, internal_child) = select_child(&arena, 0, false);
        assert_eq!(
            internal_action, 1,
            "PUCT should have selected child B based on higher prior"
        );
        assert_eq!(internal_child, 2);

        let (root_action, root_child) = select_child(&arena, 0, true);
        assert_eq!(
            root_action, 0,
            "Gumbel should have selected child A based on massive injected noise"
        );
        assert_eq!(root_child, 1);
    }
}
