use crate::core::board::GameStateExt;
use crate::core::constants::STANDARD_PIECES;
use once_cell::sync::Lazy;

pub static COMPACT_PIECE_MASKS: Lazy<Vec<Vec<(usize, u128)>>> = Lazy::new(|| {
    STANDARD_PIECES
        .iter()
        .map(|masks| {
            masks
                .iter()
                .copied()
                .enumerate()
                .filter(|&(_, m)| m != 0)
                .collect()
        })
        .collect()
});

#[derive(Clone)]
pub struct LatentNode {
    pub visits: i32,
    pub value_sum: f32,
    pub policy_logit: f32, // CHANGED: Store the logit, not the raw action_prior_probability
    pub action_prior_probability: f32,
    pub reward: f32,
    pub gumbel_noise: f32,
    pub first_child: u32,
    pub next_sibling: u32,
    pub action: i16,
    pub hidden_state_index: u32,
    pub is_topologically_expanded: bool,
}

impl LatentNode {
    pub fn new(action_prior_probability: f32, action: i16) -> Self {
        LatentNode {
            visits: 0,
            value_sum: 0.0,
            // CHANGED: Compute the expensive logarithm exactly ONCE here
            policy_logit: action_prior_probability.max(1e-8).ln(),
            action_prior_probability,
            reward: 0.0,
            gumbel_noise: 0.0,
            first_child: u32::MAX,
            next_sibling: u32::MAX,
            action,
            hidden_state_index: u32::MAX,
            is_topologically_expanded: false,
        }
    }

    pub fn value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / (self.visits as f32)
        }
    }

    pub fn get_child(&self, arena: &[LatentNode], action: i32) -> usize {
        let mut current_node_pointer = self.first_child;
        while current_node_pointer != u32::MAX {
            if arena[current_node_pointer as usize].action == action as i16 {
                return current_node_pointer as usize;
            }
            current_node_pointer = arena[current_node_pointer as usize].next_sibling;
        }
        usize::MAX
    }
}

pub fn get_valid_action_mask(state: &GameStateExt) -> [bool; 288] {
    let mut valid_action_mask = [false; 288];
    if state.terminal {
        return valid_action_mask; // Dead boards cannot expand further mathematically.
    }

    for slot in 0..3 {
        let piece_identifier = state.available[slot];
        if piece_identifier == -1 {
            continue;
        }

        for &(rotation_index, structural_mask) in
            COMPACT_PIECE_MASKS[piece_identifier as usize].iter()
        {
            if (state.board_bitmask_u128 & structural_mask) == 0 {
                let absolute_action_index = (slot * 96) + rotation_index;
                valid_action_mask[absolute_action_index] = true;
            }
        }
    }
    valid_action_mask
}

pub fn select_child(arena: &[LatentNode], node_index: usize, is_root: bool) -> (i32, usize) {
    let parent_node = &arena[node_index];

    // NORMALIZE Q
    let mut minimum_q_value = f32::INFINITY;
    let mut maximum_q_value = f32::NEG_INFINITY;
    let mut child_index = parent_node.first_child;

    while child_index != u32::MAX {
        let child_node = &arena[child_index as usize];
        let expected_q_value = if child_node.visits == 0 {
            parent_node.value()
        } else {
            child_node.reward + 0.99 * child_node.value()
        };
        if expected_q_value < minimum_q_value {
            minimum_q_value = expected_q_value;
        }
        if expected_q_value > maximum_q_value {
            maximum_q_value = expected_q_value;
        }
        child_index = child_node.next_sibling;
    }

    let mut highest_score = f32::NEG_INFINITY;
    let mut highest_action_index = -1;
    let mut highest_child_index = usize::MAX;

    let mut child_index = parent_node.first_child;
    while child_index != u32::MAX {
        let child_node = &arena[child_index as usize];
        let action_index = child_node.action as i32;

        let raw_expected_q_value = if child_node.visits == 0 {
            parent_node.value()
        } else {
            child_node.reward + 0.99 * child_node.value()
        };

        let normalized_q_value = if maximum_q_value > minimum_q_value {
            (raw_expected_q_value - minimum_q_value) / (maximum_q_value - minimum_q_value)
        } else {
            0.5
        };

        // CHANGED: Instantly read the precomputed logit. No math required!
        let action_score = if is_root {
            let gumbel_noise_injected_logit = child_node.policy_logit + child_node.gumbel_noise;
            let exploration_scale = 50.0 / ((child_node.visits + 1) as f32);
            gumbel_noise_injected_logit + (exploration_scale * normalized_q_value)
        } else {
            let puct_exploration_constant = 1.25;
            let upper_confidence_bound_score = puct_exploration_constant
                * child_node.action_prior_probability
                * ((parent_node.visits as f32).sqrt() / (1.0 + child_node.visits as f32));
            normalized_q_value + upper_confidence_bound_score
        };

        if action_score > highest_score {
            highest_score = action_score;
            highest_action_index = action_index;
            highest_child_index = child_index as usize;
        }
        child_index = child_node.next_sibling;
    }

    (highest_action_index, highest_child_index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::GameStateExt;

    #[test]
    fn test_latent_node() {
        let node = LatentNode::new(0.5, 0);
        assert_eq!(node.value(), 0.0);

        let mut node2 = LatentNode::new(0.5, 0);
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
        let mut arena = vec![
            LatentNode::new(1.0, -1),
            LatentNode::new(0.5, 0),
            LatentNode::new(0.6, 1),
        ];
        arena[0].visits = 10;
        arena[0].first_child = 1;
        arena[1].next_sibling = 2;
        arena[1].gumbel_noise = 100.0;
        arena[2].gumbel_noise = 0.0;

        let (internal_action, internal_child) = select_child(&arena, 0, false);
        assert_eq!(
            internal_action, 1,
            "PUCT should have selected child B based on higher action_prior_probability"
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
