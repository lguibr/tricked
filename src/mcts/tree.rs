use crate::node::LatentNode;
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[derive(Clone)]
pub struct SharedArena(pub Arc<Vec<LatentNode>>);

unsafe impl Send for SharedArena {}
unsafe impl Sync for SharedArena {}

impl std::ops::Deref for SharedArena {
    type Target = [LatentNode];
    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.0.as_ptr(), self.0.len()) }
    }
}

#[derive(Clone)]
pub struct MctsTree {
    pub arena: SharedArena,
    pub allocated_nodes: usize,
    pub current_generation: u32,
    pub root_index: usize,
    pub max_tree_nodes: u32,
    pub max_cache_slots: u32,
    pub allocated_cache_slots: usize,
}

pub fn allocate_node(tree: &mut MctsTree, probability: f32, action: i16) -> u32 {
    let new_idx = tree.allocated_nodes as u32;
    tree.allocated_nodes += 1;
    if tree.allocated_nodes > tree.max_tree_nodes as usize {
        panic!(
            "MCTS Tree Arena ran out of nodes! Capacity {} is too small for the search depth.",
            tree.max_tree_nodes
        );
    }
    tree.arena[new_idx as usize].reset(probability, action, tree.current_generation);
    new_idx
}

pub fn initialize_search_tree(
    mut max_tree_nodes: u32,
    mut max_cache_slots: u32,
    total_simulations: usize,
) -> MctsTree {
    let safe_min_nodes = (total_simulations as u32 * 288) + 5000;
    if max_tree_nodes < safe_min_nodes {
        max_tree_nodes = safe_min_nodes;
    }
    if max_cache_slots < safe_min_nodes {
        max_cache_slots = safe_min_nodes;
    }

    let capacity = max_tree_nodes as usize;
    let mut arena = Vec::with_capacity(capacity);
    for _ in 0..capacity {
        arena.push(LatentNode::new(0.0, -1, 0));
    }

    arena[0].reset(1.0, -1, 1);

    MctsTree {
        arena: SharedArena(Arc::new(arena)),
        allocated_nodes: 1, // root is at index 0
        current_generation: 1,
        root_index: 0,
        max_tree_nodes,
        max_cache_slots,
        allocated_cache_slots: 1,
    }
}

pub fn expand_root_node(
    tree: &mut MctsTree,
    root_cache_index: u32,
    child_prior_probabilities: &[f32],
) {
    if tree.arena[tree.root_index]
        .is_topologically_expanded
        .load(Ordering::Relaxed)
    {
        return;
    }
    let mut prev_child = u32::MAX;
    let mut first_child = u32::MAX;

    for (action_index, &probability) in child_prior_probabilities.iter().enumerate() {
        let new_node_index = allocate_node(tree, probability, action_index as i16);
        if first_child == u32::MAX {
            first_child = new_node_index;
        } else {
            tree.arena[prev_child as usize]
                .next_sibling
                .store(new_node_index, Ordering::SeqCst);
        }
        prev_child = new_node_index;
    }

    tree.arena[tree.root_index]
        .first_child
        .store(first_child, Ordering::SeqCst);
    tree.arena[tree.root_index]
        .hidden_state_index
        .store(root_cache_index, Ordering::SeqCst);
    tree.arena[tree.root_index]
        .is_topologically_expanded
        .store(true, Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn test_bump_allocator_tree() {
        let mut tree = initialize_search_tree(100, 100, 100);
        assert_eq!(tree.allocated_nodes, 1);

        let node_id = allocate_node(&mut tree, 0.5, 2);
        assert_eq!(node_id, 1);
        assert_eq!(tree.allocated_nodes, 2);

        let node_id_2 = allocate_node(&mut tree, 0.3, 3);
        assert_eq!(node_id_2, 2);
        assert_eq!(tree.allocated_nodes, 3);

        assert_eq!(
            tree.arena[node_id as usize].action.load(Ordering::Relaxed),
            2
        );
        assert_eq!(
            tree.arena[node_id_2 as usize]
                .action
                .load(Ordering::Relaxed),
            3
        );
    }
}
