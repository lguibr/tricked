use crate::GameStateExt;
use crate::constants::STANDARD_PIECES;
use std::collections::HashMap;

#[derive(Clone)]
pub struct LatentNode {
    pub visits: i32,
    pub value_sum: f32,
    pub prior: f32,
    pub hidden_state: Option<Vec<f32>>,
    pub reward: f32,
    pub gumbel_noise: f32,
    pub children: HashMap<i32, usize>,
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
            children: HashMap::new(),
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

pub fn get_valid_action_mask(state: &GameStateExt) -> Vec<bool> {
    let mut mask = vec![false; 288];
    if state.terminal {
        return mask;
    }

    for slot in 0..3 {
        let p_id = state.available[slot];
        if p_id == -1 {
            continue;
        }
        for idx in 0..96 {
            let m = STANDARD_PIECES[p_id as usize][idx];
            if m != 0 && (state.board & m) == 0 {
                let action_idx = slot * 96 + idx;
                mask[action_idx] = true;
            }
        }
    }
    mask
}

pub fn select_child(arena: &[LatentNode], node_idx: usize, is_root: bool) -> (i32, usize) {
    let node = &arena[node_idx];
    let mut best_score = std::f32::NEG_INFINITY;
    let mut best_action = -1;
    let mut best_child = usize::MAX;

    for (&action, &child_idx) in &node.children {
        let child = &arena[child_idx];
        let q_value = if child.visits == 0 {
            node.value()
        } else {
            child.reward + 0.99 * child.value()
        };

        let logit = child.prior.max(1e-8).ln();
        let score = if is_root {
            let gumbel_logit = logit + child.gumbel_noise;
            let c_scale = 50.0 / ((child.visits + 1) as f32);
            gumbel_logit + (c_scale * q_value)
        } else {
            logit + q_value
        };

        if score > best_score {
            best_score = score;
            best_action = action;
            best_child = child_idx;
        }
    }

    (best_action, best_child)
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
        let mut state = GameStateExt::new(Some(vec![0, 0, 0]), 0, 0, 6, 0);
        let mask = get_valid_action_mask(&state);
        assert!(mask.contains(&true));

        state.terminal = true;
        let terminal_mask = get_valid_action_mask(&state);
        assert!(!terminal_mask.contains(&true));
    }

    #[test]
    fn test_select_child() {
        let mut arena = vec![LatentNode::new(1.0)];
        arena[0].children.insert(10, 1);
        arena[0].children.insert(20, 2);

        arena.push(LatentNode::new(0.3));
        arena.push(LatentNode::new(0.7));

        let (best_action, best_child) = select_child(&arena, 0, false);
        assert_eq!(best_action, 20);
        assert_eq!(best_child, 2);

        let (root_action, root_child) = select_child(&arena, 0, true);
        assert_eq!(root_action, 20);
        assert_eq!(root_child, 2);
    }
}
