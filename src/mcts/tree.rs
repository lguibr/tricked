use crate::node::LatentNode;

#[derive(Clone)]
pub struct MctsTree {
    pub arena: Vec<LatentNode>,
    pub node_free_list: Vec<u32>,
    pub gpu_cache_free_list: Vec<u32>,
    pub current_generation: u32,
    pub root_index: usize,
    pub maximum_allowed_nodes_in_search_tree: u32,
}

pub fn allocate_node(tree: &mut MctsTree, probability: f32, action: i16) -> u32 {
    let new_idx = tree.node_free_list.pop().expect("Tree out of nodes!");
    tree.arena[new_idx as usize] = LatentNode::new(probability, action, tree.current_generation);
    new_idx
}

pub fn advance_root(mut tree: MctsTree, new_root: usize) -> MctsTree {
    tree.current_generation += 1;
    let gen = tree.current_generation;
    tree.root_index = new_root;

    // BFS to update generations
    let mut queue = vec![new_root as u32];
    let mut head = 0;
    tree.arena[new_root].generation = gen;

    while head < queue.len() {
        let node_idx = queue[head] as usize;
        head += 1;

        let mut child_idx = tree.arena[node_idx].first_child;
        while child_idx != u32::MAX {
            tree.arena[child_idx as usize].generation = gen;
            queue.push(child_idx);
            child_idx = tree.arena[child_idx as usize].next_sibling;
        }
    }

    // Rebuild free lists
    tree.node_free_list.clear();
    tree.gpu_cache_free_list.clear();

    let mut used_gpu_states = vec![false; tree.maximum_allowed_nodes_in_search_tree as usize];

    // Arena is pre-allocated up to maximum_allowed_nodes_in_search_tree
    for i in 0..tree.maximum_allowed_nodes_in_search_tree as usize {
        if tree.arena[i].generation != gen {
            // Node is unreached, push to free list
            tree.node_free_list.push(i as u32);
        } else {
            let state_idx = tree.arena[i].hidden_state_index;
            if state_idx != u32::MAX {
                used_gpu_states[state_idx as usize] = true;
            }
        }
    }

    for (i, &used) in used_gpu_states.iter().enumerate() {
        if !used {
            tree.gpu_cache_free_list.push(i as u32);
        }
    }

    tree
}

pub fn initialize_search_tree(
    previous_tree: Option<MctsTree>,
    last_executed_action: Option<i32>,
    maximum_allowed_nodes_in_search_tree: u32,
    total_simulations: usize,
) -> MctsTree {
    if let Some(mut existing_tree) = previous_tree {
        if let Some(action) = last_executed_action {
            let child_index = existing_tree.arena[existing_tree.root_index]
                .get_child(&existing_tree.arena, action);
            if child_index != usize::MAX {
                let advanced_tree = advance_root(existing_tree, child_index);
                if advanced_tree.node_free_list.len() > total_simulations + 10
                    && advanced_tree.gpu_cache_free_list.len() > total_simulations + 10
                {
                    return advanced_tree;
                }
                existing_tree = advanced_tree;
            }
        } else {
            let root = existing_tree.root_index;
            let advanced_tree = advance_root(existing_tree, root);
            if advanced_tree.node_free_list.len() > total_simulations + 10
                && advanced_tree.gpu_cache_free_list.len() > total_simulations + 10
            {
                return advanced_tree;
            }
            existing_tree = advanced_tree;
        }

        // If we hit here, we need to reset the tree but WE CAN REUSE the memory!
        existing_tree.current_generation += 1;
        let gen = existing_tree.current_generation;
        existing_tree.root_index = 0;
        existing_tree.arena[0] = LatentNode::new(1.0, -1, gen);

        existing_tree.node_free_list.clear();
        for i in (1..maximum_allowed_nodes_in_search_tree).rev() {
            existing_tree.node_free_list.push(i);
        }

        existing_tree.gpu_cache_free_list.clear();
        for i in (0..maximum_allowed_nodes_in_search_tree).rev() {
            existing_tree.gpu_cache_free_list.push(i);
        }
        return existing_tree;
    }

    let capacity = maximum_allowed_nodes_in_search_tree as usize;
    let mut arena = vec![LatentNode::new(0.0, -1, 0); capacity];

    let mut node_free_list = Vec::with_capacity(capacity);
    for i in (1..maximum_allowed_nodes_in_search_tree).rev() {
        node_free_list.push(i);
    }

    let mut gpu_cache_free_list = Vec::with_capacity(capacity);
    for i in (0..maximum_allowed_nodes_in_search_tree).rev() {
        gpu_cache_free_list.push(i);
    }

    arena[0] = LatentNode::new(1.0, -1, 1);

    MctsTree {
        arena,
        node_free_list,
        gpu_cache_free_list,
        current_generation: 1,
        root_index: 0,
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

        for _step in 0..10_000 {
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

            tree = advance_root(tree, new_root as usize);

            assert!(
                tree.node_free_list.len() >= max_nodes as usize - 100,
                "Nodes leaked!"
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

        let new_tree = advance_root(tree, new_root);

        assert_eq!(new_tree.node_free_list.len(), 500);
        assert_eq!(new_tree.root_index, 0);
    }

    #[test]
    fn test_tree_garbage_collection() {
        let mut mock_arena = vec![LatentNode::new(0.0, 0, 0); 10];
        let mut root = LatentNode::new(1.0, -1, 0);
        root.is_topologically_expanded = true;
        root.first_child = 1;
        mock_arena[0] = root;

        let mut child1 = LatentNode::new(0.5, 0, 0);
        child1.is_topologically_expanded = true;
        child1.first_child = 3;
        child1.next_sibling = 2;
        child1.visits = 10;
        child1.hidden_state_index = 5;
        mock_arena[1] = child1;

        let mut child2 = LatentNode::new(0.5, 1, 0);
        child2.visits = 5;
        child2.hidden_state_index = 6;
        mock_arena[2] = child2;

        let mut grandchild = LatentNode::new(1.0, 2, 0);
        grandchild.visits = 2;
        grandchild.hidden_state_index = 7;
        mock_arena[3] = grandchild;

        let mut disconnected = LatentNode::new(1.0, 3, 0);
        disconnected.hidden_state_index = 8;
        mock_arena[4] = disconnected;

        let mut node_free_list = vec![9];

        let tree = MctsTree {
            arena: mock_arena,
            node_free_list,
            gpu_cache_free_list: vec![],
            current_generation: 0,
            root_index: 0,
            maximum_allowed_nodes_in_search_tree: 10,
        };

        let new_tree = advance_root(tree, 1);
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
            new_tree.gpu_cache_free_list.contains(&6),
            "Child 2 cache slot must be freed"
        );
        assert!(
            new_tree.gpu_cache_free_list.contains(&8),
            "Disconnected node cache slot must be freed"
        );
        assert!(
            !new_tree.gpu_cache_free_list.contains(&5),
            "Child 1 cache slot must survive"
        );
        assert!(
            !new_tree.gpu_cache_free_list.contains(&7),
            "Grandchild cache slot must survive"
        );
    }
}
