use crate::node::LatentNode;

#[derive(Clone)]
pub struct MctsTree {
    pub arena: Vec<LatentNode>,
    pub swap_arena: Vec<LatentNode>,
    pub pointer_remapping: Vec<u32>,
    pub arena_alloc_ptr: usize,
    pub root_index: usize,
    pub free_list: Vec<u32>, // GPU latent state free list
    pub maximum_allowed_nodes_in_search_tree: u32,
}

pub fn allocate_node(tree: &mut MctsTree, probability: f32, action: i16) -> u32 {
    let new_idx = tree.arena_alloc_ptr;

    if new_idx >= tree.arena.len() {
        let new_capacity = tree.arena.len().max(10_000) * 2;
        tree.arena.resize(new_capacity, LatentNode::new(0.0, -1));
        tree.swap_arena
            .resize(new_capacity, LatentNode::new(0.0, -1));
        tree.pointer_remapping.resize(new_capacity, u32::MAX);
    }

    tree.arena_alloc_ptr += 1;

    tree.arena[new_idx] = LatentNode::new(probability, action);
    new_idx as u32
}

pub fn gc_tree(mut tree: MctsTree, new_root: usize) -> MctsTree {
    let mut new_alloc_ptr = 0;
    let mut queue = vec![new_root as u32];
    tree.pointer_remapping.fill(u32::MAX);

    // Copy new root
    tree.pointer_remapping[new_root] = new_alloc_ptr as u32;
    tree.swap_arena[new_alloc_ptr] = tree.arena[new_root].clone();
    new_alloc_ptr += 1;

    let mut head = 0;
    while head < queue.len() {
        let old_node_idx = queue[head] as usize;
        let new_node_idx = tree.pointer_remapping[old_node_idx] as usize;
        head += 1;

        let mut child_idx = tree.arena[old_node_idx].first_child;
        let mut prev_new_child_idx = u32::MAX;
        let mut is_first = true;

        while child_idx != u32::MAX {
            let new_child_idx = new_alloc_ptr;
            new_alloc_ptr += 1;

            tree.pointer_remapping[child_idx as usize] = new_child_idx as u32;
            tree.swap_arena[new_child_idx] = tree.arena[child_idx as usize].clone();

            if is_first {
                tree.swap_arena[new_node_idx].first_child = new_child_idx as u32;
                is_first = false;
            } else {
                tree.swap_arena[prev_new_child_idx as usize].next_sibling = new_child_idx as u32;
            }

            tree.swap_arena[new_child_idx].next_sibling = u32::MAX;

            queue.push(child_idx);
            prev_new_child_idx = new_child_idx as u32;
            child_idx = tree.arena[child_idx as usize].next_sibling;
        }
    }

    // Rebuild free_list of GPU cache states
    tree.free_list.clear();
    let mut used_states = vec![false; tree.maximum_allowed_nodes_in_search_tree as usize];
    for i in 0..new_alloc_ptr {
        let state_idx = tree.swap_arena[i].hidden_state_index;
        if state_idx != u32::MAX {
            used_states[state_idx as usize] = true;
        }
    }
    for (i, &used) in used_states.iter().enumerate() {
        if !used {
            tree.free_list.push(i as u32);
        }
    }

    std::mem::swap(&mut tree.arena, &mut tree.swap_arena);
    tree.arena_alloc_ptr = new_alloc_ptr;
    tree.root_index = 0; // The new root is now at index 0

    tree
}

pub fn initialize_search_tree(
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

    let dynamic_capacity = (total_simulations * 300 + 10_000).max(100_000);
    let mut arena = vec![LatentNode::new(0.0, -1); dynamic_capacity];
    let swap_arena = vec![LatentNode::new(0.0, -1); dynamic_capacity];
    let pointer_remapping = vec![u32::MAX; dynamic_capacity];

    let free_list = (0..maximum_allowed_nodes_in_search_tree)
        .rev()
        .collect::<Vec<u32>>();

    arena[0] = LatentNode::new(1.0, -1);

    MctsTree {
        arena,
        swap_arena,
        pointer_remapping,
        arena_alloc_ptr: 1,
        root_index: 0,
        free_list,
        maximum_allowed_nodes_in_search_tree,
    }
}

pub fn expand_root_node(
    tree: &mut MctsTree,
    root_cache_index: u32,
    normalized_probabilities: &[f32],
) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::LatentNode;

    #[test]
    fn test_tree_arena_garbage_collection_leak_100000_steps() {
        let max_nodes = 1000;
        let mut tree = initialize_search_tree(None, None, max_nodes, 500);

        for _step in 0..100_000 {
            let current_root = tree.root_index;
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

            let mut new_root = first;
            for _ in 0..4 {
                new_root = tree.arena[new_root as usize].next_sibling;
            }

            tree = gc_tree(tree, new_root as usize);

            assert!(
                tree.arena_alloc_ptr <= max_nodes as usize * 2,
                "Arena leaked and grew infinitely!"
            );
        }
    }

    #[test]
    fn test_tree_arena_garbage_collection() {
        let mut tree = initialize_search_tree(None, None, 1000, 500);
        let new_root = 0;

        for i in 0..500 {
            let p_idx = allocate_node(&mut tree, 1.0, i as i16);
            if i > 0 {
                tree.arena[p_idx as usize].next_sibling = tree.arena[new_root].first_child;
                tree.arena[new_root].first_child = p_idx;
                tree.arena[p_idx as usize].hidden_state_index = i as u32;
            }
        }

        let new_tree = gc_tree(tree, new_root);

        assert_eq!(new_tree.arena_alloc_ptr, 500);
        assert_eq!(new_tree.root_index, 0);
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
        child1.next_sibling = 2;
        child1.visits = 10;
        child1.hidden_state_index = 5;
        mock_arena[1] = child1;

        let mut child2 = LatentNode::new(0.5, 1);
        child2.visits = 5;
        child2.hidden_state_index = 6;
        mock_arena[2] = child2;

        let mut grandchild = LatentNode::new(1.0, 2);
        grandchild.visits = 2;
        grandchild.hidden_state_index = 7;
        mock_arena[3] = grandchild;

        let mut disconnected = LatentNode::new(1.0, 3);
        disconnected.hidden_state_index = 8;
        mock_arena[4] = disconnected;

        let initial_free_list = vec![9];

        let tree = MctsTree {
            arena: mock_arena,
            swap_arena: vec![LatentNode::new(0.0, -1); 1000],
            pointer_remapping: vec![u32::MAX; 1000],
            arena_alloc_ptr: 5,
            root_index: 0,
            free_list: initial_free_list,
            maximum_allowed_nodes_in_search_tree: 1000,
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
            new_tree.free_list.contains(&6),
            "Child 2 cache slot must be freed"
        );
        assert!(
            new_tree.free_list.contains(&8),
            "Disconnected node cache slot must be freed"
        );
        assert!(
            !new_tree.free_list.contains(&5),
            "Child 1 cache slot must survive"
        );
        assert!(
            !new_tree.free_list.contains(&7),
            "Grandchild cache slot must survive"
        );
        assert!(
            new_tree.free_list.contains(&9),
            "Original free list items must survive"
        );
    }
}
