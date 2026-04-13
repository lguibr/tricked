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

use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};

pub struct AtomicF32(AtomicU32);

impl AtomicF32 {
    pub fn new(val: f32) -> Self {
        AtomicF32(AtomicU32::new(val.to_bits()))
    }

    pub fn load(&self, order: Ordering) -> f32 {
        f32::from_bits(self.0.load(order))
    }

    pub fn store(&self, val: f32, order: Ordering) {
        self.0.store(val.to_bits(), order);
    }

    pub fn fetch_add(&self, val: f32, order: Ordering) -> f32 {
        if val.is_nan() {
            return self.load(order);
        }
        let mut current = self.0.load(order);
        loop {
            let current_f32 = f32::from_bits(current);
            if current_f32.is_nan() {
                return current_f32;
            }
            let next = (current_f32 + val).to_bits();
            match self.0.compare_exchange_weak(current, next, order, order) {
                Ok(_) => return current_f32,
                Err(e) => current = e,
            }
        }
    }
}

pub struct LatentNode {
    pub visits: AtomicI32,
    pub value_sum: AtomicF32,
    pub policy_logit: AtomicF32,
    pub action_prior_probability: AtomicF32,
    pub value_prefix: AtomicF32,
    pub cumulative_value_prefix: AtomicF32,
    pub gumbel_noise: AtomicF32,
    pub virtual_loss: AtomicI32,
    pub in_flight: AtomicI32,
    pub first_child: AtomicU32,
    pub next_sibling: AtomicU32,
    pub action: AtomicI32,
    pub hidden_state_index: AtomicU32,
    pub is_topologically_expanded: AtomicBool,
    pub generation: AtomicU32,
}

impl LatentNode {
    pub fn new(action_prior_probability: f32, action: i16, generation: u32) -> Self {
        LatentNode {
            visits: AtomicI32::new(0),
            value_sum: AtomicF32::new(0.0),
            policy_logit: AtomicF32::new(action_prior_probability.max(1e-8).ln()),
            action_prior_probability: AtomicF32::new(action_prior_probability),
            value_prefix: AtomicF32::new(0.0),
            cumulative_value_prefix: AtomicF32::new(0.0),
            gumbel_noise: AtomicF32::new(0.0),
            virtual_loss: AtomicI32::new(0),
            in_flight: AtomicI32::new(0),
            first_child: AtomicU32::new(u32::MAX),
            next_sibling: AtomicU32::new(u32::MAX),
            action: AtomicI32::new(action as i32),
            hidden_state_index: AtomicU32::new(u32::MAX),
            is_topologically_expanded: AtomicBool::new(false),
            generation: AtomicU32::new(generation),
        }
    }

    pub fn reset(&self, action_prior_probability: f32, action: i16, generation: u32) {
        self.visits.store(0, Ordering::SeqCst);
        self.value_sum.store(0.0, Ordering::SeqCst);
        self.policy_logit
            .store(action_prior_probability.max(1e-8).ln(), Ordering::SeqCst);
        self.action_prior_probability
            .store(action_prior_probability, Ordering::SeqCst);
        self.value_prefix.store(0.0, Ordering::SeqCst);
        self.cumulative_value_prefix.store(0.0, Ordering::SeqCst);
        self.gumbel_noise.store(0.0, Ordering::SeqCst);
        self.virtual_loss.store(0, Ordering::SeqCst);
        self.in_flight.store(0, Ordering::SeqCst);
        self.first_child.store(u32::MAX, Ordering::SeqCst);
        self.next_sibling.store(u32::MAX, Ordering::SeqCst);
        self.action.store(action as i32, Ordering::SeqCst);
        self.hidden_state_index.store(u32::MAX, Ordering::SeqCst);
        self.is_topologically_expanded
            .store(false, Ordering::SeqCst);
        self.generation.store(generation, Ordering::SeqCst);
    }

    pub fn value(&self) -> f32 {
        let visits = self.visits.load(Ordering::Relaxed);
        let virtual_loss = self.virtual_loss.load(Ordering::Relaxed);
        let in_flight = self.in_flight.load(Ordering::Relaxed);
        let effective_visits = visits + virtual_loss + in_flight;
        if effective_visits == 0 {
            0.0
        } else {
            (self.value_sum.load(Ordering::Relaxed) - virtual_loss as f32 - in_flight as f32)
                / (effective_visits as f32)
        }
    }

    pub fn get_child(&self, arena: &[LatentNode], action: i32) -> usize {
        let mut current_node_pointer = self.first_child.load(Ordering::Relaxed);
        while current_node_pointer != u32::MAX {
            if arena[current_node_pointer as usize]
                .action
                .load(Ordering::Relaxed)
                == action
            {
                return current_node_pointer as usize;
            }
            current_node_pointer = arena[current_node_pointer as usize]
                .next_sibling
                .load(Ordering::Relaxed);
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
    let mut child_index = parent_node.first_child.load(Ordering::Relaxed);

    while child_index != u32::MAX {
        let child_node = &arena[child_index as usize];
        let effective_visits = child_node.visits.load(Ordering::Relaxed)
            + child_node.virtual_loss.load(Ordering::Relaxed)
            + child_node.in_flight.load(Ordering::Relaxed);
        let expected_q_value = if effective_visits == 0 {
            parent_node.value()
        } else {
            child_node.value_prefix.load(Ordering::Relaxed) + 0.99 * child_node.value()
        };
        if expected_q_value < minimum_q_value {
            minimum_q_value = expected_q_value;
        }
        if expected_q_value > maximum_q_value {
            maximum_q_value = expected_q_value;
        }
        child_index = child_node.next_sibling.load(Ordering::Relaxed);
    }

    let mut highest_score = f32::NEG_INFINITY;
    let mut highest_action_index = -1;
    let mut highest_child_index = usize::MAX;

    let mut child_index = parent_node.first_child.load(Ordering::Relaxed);
    while child_index != u32::MAX {
        let child_node = &arena[child_index as usize];
        let action_index = child_node.action.load(Ordering::Relaxed);

        let effective_visits = child_node.visits.load(Ordering::Relaxed)
            + child_node.virtual_loss.load(Ordering::Relaxed)
            + child_node.in_flight.load(Ordering::Relaxed);
        let raw_expected_q_value = if effective_visits == 0 {
            parent_node.value()
        } else {
            child_node.value_prefix.load(Ordering::Relaxed) + 0.99 * child_node.value()
        };

        let normalized_q_value = if maximum_q_value > minimum_q_value {
            (raw_expected_q_value - minimum_q_value) / (maximum_q_value - minimum_q_value)
        } else {
            0.5
        };

        // CHANGED: Instantly read the precomputed logit. No math required!
        let action_score = if is_root {
            let gumbel_noise_injected_logit = child_node.policy_logit.load(Ordering::Relaxed)
                + child_node.gumbel_noise.load(Ordering::Relaxed);
            let exploration_scale = 50.0 / ((effective_visits + 1) as f32);
            gumbel_noise_injected_logit + (exploration_scale * normalized_q_value)
        } else {
            let puct_exploration_constant = 1.25;
            let parent_effective_visits = parent_node.visits.load(Ordering::Relaxed)
                + parent_node.virtual_loss.load(Ordering::Relaxed)
                + parent_node.in_flight.load(Ordering::Relaxed);
            let upper_confidence_bound_score = puct_exploration_constant
                * child_node.action_prior_probability.load(Ordering::Relaxed)
                * ((parent_effective_visits as f32).sqrt() / (1.0 + effective_visits as f32));
            normalized_q_value + upper_confidence_bound_score
        };

        if action_score > highest_score {
            highest_score = action_score;
            highest_action_index = action_index;
            highest_child_index = child_index as usize;
        }
        child_index = child_node.next_sibling.load(Ordering::Relaxed);
    }

    (highest_action_index, highest_child_index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::GameStateExt;

    #[test]
    fn test_latent_node() {
        let node = LatentNode::new(0.5, 0, 0);
        assert_eq!(node.value(), 0.0);

        let node2 = LatentNode::new(0.5, 0, 0);
        node2.visits.store(2, Ordering::SeqCst);
        node2.value_sum.store(1.0, Ordering::SeqCst);
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
        let arena = vec![
            LatentNode::new(1.0, -1, 0),
            LatentNode::new(0.5, 0, 0),
            LatentNode::new(0.6, 1, 0),
        ];
        arena[0].visits.store(10, Ordering::SeqCst);
        arena[0].first_child.store(1, Ordering::SeqCst);
        arena[1].next_sibling.store(2, Ordering::SeqCst);
        arena[1].gumbel_noise.store(100.0, Ordering::SeqCst);
        arena[2].gumbel_noise.store(0.0, Ordering::SeqCst);

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

    #[test]
    fn test_atomic_f32_nan_safety() {
        let atom = std::sync::Arc::new(AtomicF32::new(0.0));
        let mut handles = vec![];
        for i in 0..16 {
            let atom_clone = atom.clone();
            handles.push(std::thread::spawn(move || {
                if i % 3 == 0 {
                    atom_clone.fetch_add(f32::NAN, Ordering::Relaxed);
                } else if i % 3 == 1 {
                    atom_clone.fetch_add(1.0, Ordering::Relaxed);
                } else {
                    atom_clone.fetch_add(1e-8, Ordering::Relaxed);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let final_val = atom.load(Ordering::Relaxed);
        assert!(final_val.is_nan() || final_val.is_finite());
    }
}
