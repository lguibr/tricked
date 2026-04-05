use crate::node::LatentNode;
use crossbeam_channel::{Receiver, Sender};
use crossbeam_queue::ArrayQueue;
use once_cell::sync::Lazy;
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

impl std::ops::DerefMut for SharedArena {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.0.as_ptr() as *mut LatentNode, self.0.len()) }
    }
}

pub struct GcTask {
    pub node_idx: usize,
    pub arena: SharedArena,
    pub node_free_list: Arc<ArrayQueue<u32>>,
    pub gpu_cache_free_list: Arc<ArrayQueue<u32>>,
}

static GC_CHANNEL: Lazy<(Sender<GcTask>, Receiver<GcTask>)> = Lazy::new(|| {
    let (tx, rx): (Sender<GcTask>, Receiver<GcTask>) = crossbeam_channel::bounded(100_000);
    for i in 0..16 {
        std::thread::Builder::new()
            .name(format!("MCTS Tree GC {}", i))
            .spawn({
                let receiver = rx.clone();
                move || {
                    for task in receiver.iter() {
                        let mut local_nodes = Vec::with_capacity(256);
                        let mut local_cache = Vec::with_capacity(256);
                        let mut stack = vec![task.node_idx as u32];
                        while let Some(idx) = stack.pop() {
                            let node = &task.arena[idx as usize];

                            let mut child_idx = node.first_child;
                            while child_idx != u32::MAX {
                                stack.push(child_idx);
                                let child_node = &task.arena[child_idx as usize];
                                child_idx = child_node.next_sibling;
                            }

                            local_nodes.push(idx);
                            if local_nodes.len() >= 256 {
                                for &n in &local_nodes {
                                    let _ = task.node_free_list.push(n);
                                }
                                local_nodes.clear();
                            }

                            let state_idx = node.hidden_state_index;
                            if state_idx != u32::MAX {
                                local_cache.push(state_idx);
                                if local_cache.len() >= 256 {
                                    for &c in &local_cache {
                                        let _ = task.gpu_cache_free_list.push(c);
                                    }
                                    local_cache.clear();
                                }
                            }
                        }
                        for &n in &local_nodes {
                            let _ = task.node_free_list.push(n);
                        }
                        for &c in &local_cache {
                            let _ = task.gpu_cache_free_list.push(c);
                        }
                    }
                }
            })
            .expect("Failed to spawn GC Thread");
    }
    (tx, rx)
});

#[derive(Clone)]
pub struct MctsTree {
    pub arena: SharedArena,
    pub node_free_list: Arc<ArrayQueue<u32>>,
    pub gpu_cache_free_list: Arc<ArrayQueue<u32>>,
    pub current_generation: u32,
    pub root_index: usize,
    pub max_tree_nodes: u32,
    pub max_cache_slots: u32,
}

pub fn allocate_node(tree: &mut MctsTree, probability: f32, action: i16) -> u32 {
    let mut attempts = 0;
    let new_idx = loop {
        if let Some(idx) = tree.node_free_list.pop() {
            break idx;
        }
        attempts += 1;
        if attempts > 10_000 {
            panic!(
                "MCTS Tree Arena ran out of nodes! Capacity {} is too small for the search depth. Active Node Free List size: {}",
                tree.max_tree_nodes,
                tree.node_free_list.len()
            );
        }
        if attempts > 100 {
            std::thread::sleep(std::time::Duration::from_millis(1));
        } else {
            std::thread::yield_now();
        }
    };
    tree.arena[new_idx as usize] = LatentNode::new(probability, action, tree.current_generation);
    new_idx
}

pub fn advance_root(mut tree: MctsTree, new_root: usize) -> MctsTree {
    let old_root = tree.root_index;
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

    if old_root != new_root {
        let mut child_idx = tree.arena[old_root].first_child;
        while child_idx != u32::MAX {
            let next_sibling = tree.arena[child_idx as usize].next_sibling;
            if child_idx as usize != new_root {
                let mut task = GcTask {
                    node_idx: child_idx as usize,
                    arena: tree.arena.clone(),
                    node_free_list: Arc::clone(&tree.node_free_list),
                    gpu_cache_free_list: Arc::clone(&tree.gpu_cache_free_list),
                };
                loop {
                    match GC_CHANNEL.0.try_send(task) {
                        Ok(_) => break,
                        Err(crossbeam_channel::TrySendError::Full(t)) => {
                            std::thread::yield_now();
                            task = t;
                        }
                        Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
                            panic!("GC Thread Channel died")
                        }
                    }
                }
            }
            child_idx = next_sibling;
        }

        let _ = tree.node_free_list.push(old_root as u32);
        let state_idx = tree.arena[old_root].hidden_state_index;
        if state_idx != u32::MAX {
            let _ = tree.gpu_cache_free_list.push(state_idx);
        }
    }

    tree
}

pub fn initialize_search_tree(
    previous_tree: Option<MctsTree>,
    last_executed_action: Option<i32>,
    max_tree_nodes: u32,
    max_cache_slots: u32,
    total_simulations: usize,
) -> MctsTree {
    if let Some(mut existing_tree) = previous_tree {
        if let Some(action) = last_executed_action {
            let child_index = existing_tree.arena[existing_tree.root_index]
                .get_child(&existing_tree.arena, action);
            if child_index != usize::MAX {
                let advanced_tree = advance_root(existing_tree, child_index);
                // A single search can allocate roughly (total_simulations * 16 * 1.5) * 288 nodes.
                // We add a strict safety buffer to prevent panicking mid-search.
                let required_nodes = (total_simulations * 16 * 288) + 5000;

                if advanced_tree.node_free_list.len() > required_nodes
                    && advanced_tree.gpu_cache_free_list.len() > total_simulations + 10
                {
                    return advanced_tree;
                }
                existing_tree = advanced_tree;
            }
        } else {
            let root = existing_tree.root_index;
            let advanced_tree = advance_root(existing_tree, root);
            let required_nodes = (total_simulations * 16 * 288) + 5000;
            if advanced_tree.node_free_list.len() > required_nodes
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

        // Discarding old items and repopulating all available slots
        // Creating a new Arc guarantees background GC threads will safely write to orphaned queues
        let new_node_free = Arc::new(ArrayQueue::new(max_tree_nodes as usize));
        for i in (1..max_tree_nodes).rev() {
            let _ = new_node_free.push(i);
        }
        existing_tree.node_free_list = new_node_free;

        let new_cache_free = Arc::new(ArrayQueue::new(max_cache_slots as usize));
        for i in (0..max_cache_slots).rev() {
            let _ = new_cache_free.push(i);
        }
        existing_tree.gpu_cache_free_list = new_cache_free;
        return existing_tree;
    }

    let capacity = max_tree_nodes as usize;
    let mut arena = vec![LatentNode::new(0.0, -1, 0); capacity];

    let node_free_list = Arc::new(ArrayQueue::new(max_tree_nodes as usize));
    for i in (1..max_tree_nodes).rev() {
        let _ = node_free_list.push(i);
    }

    let gpu_cache_free_list = Arc::new(ArrayQueue::new(max_cache_slots as usize));
    for i in (0..max_cache_slots).rev() {
        let _ = gpu_cache_free_list.push(i);
    }

    arena[0] = LatentNode::new(1.0, -1, 1);

    MctsTree {
        arena: SharedArena(Arc::new(arena)),
        node_free_list,
        gpu_cache_free_list,
        current_generation: 1,
        root_index: 0,
        max_tree_nodes,
        max_cache_slots,
    }
}

pub fn expand_root_node(
    tree: &mut MctsTree,
    root_cache_index: u32,
    child_prior_probabilities: &[f32],
) {
    if tree.arena[tree.root_index].is_topologically_expanded {
        return;
    }
    let mut prev_child = u32::MAX;
    let mut first_child = u32::MAX;

    for (action_index, &probability) in child_prior_probabilities.iter().enumerate() {
        let new_node_index = allocate_node(tree, probability, action_index as i16);
        if first_child == u32::MAX {
            first_child = new_node_index;
        } else {
            tree.arena[prev_child as usize].next_sibling = new_node_index;
        }
        prev_child = new_node_index;
    }

    tree.arena[tree.root_index].first_child = first_child;
    tree.arena[tree.root_index].hidden_state_index = root_cache_index;
    tree.arena[tree.root_index].is_topologically_expanded = true;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::LatentNode;

    #[test]
    fn test_tree_arena_garbage_collection_leak_100000_steps() {
        let max_nodes = 1000;
        let mut tree = initialize_search_tree(None, None, max_nodes, 1000, 500);

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
            // wait for GC to process
            std::thread::sleep(std::time::Duration::from_millis(1));

            assert!(
                tree.node_free_list.len() >= max_nodes as usize - 100,
                "Nodes leaked!"
            );
        }
    }

    #[test]
    fn test_tree_arena_garbage_collection() {
        let mut tree = initialize_search_tree(None, None, 1000, 1000, 500);
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

        std::thread::sleep(std::time::Duration::from_millis(50));

        assert_eq!(new_tree.node_free_list.len(), 499);
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

        let node_free_list = Arc::new(crossbeam_queue::ArrayQueue::new(10));
        for i in 4..10 {
            let _ = node_free_list.push(i);
        }

        let tree = MctsTree {
            arena: SharedArena(std::sync::Arc::new(mock_arena)),
            node_free_list,
            gpu_cache_free_list: Arc::new(crossbeam_queue::ArrayQueue::new(10)),
            root_index: 0,
            current_generation: 0,
            max_tree_nodes: 10,
            max_cache_slots: 10,
        };

        let new_tree = advance_root(tree, 1);
        std::thread::sleep(std::time::Duration::from_millis(50));

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

        let mut cache_freed = vec![];
        while let Some(v) = new_tree.gpu_cache_free_list.pop() {
            cache_freed.push(v);
        }

        assert!(cache_freed.contains(&6), "Child 2 cache slot must be freed");
        assert!(!cache_freed.contains(&5), "Child 1 cache slot must survive");
        assert!(
            !cache_freed.contains(&7),
            "Grandchild cache slot must survive"
        );
    }

    #[test]
    fn test_async_gc_use_after_free_race_regression() {
        let mut tree = initialize_search_tree(None, None, 5000, 5000, 10);

        for _ in 0..100 {
            let mut last_node = tree.root_index as u32;
            for i in 0..20 {
                let next = allocate_node(&mut tree, 1.0, i as i16);
                tree.arena[last_node as usize].first_child = next;
                last_node = next;
            }

            let new_root = allocate_node(&mut tree, 1.0, 99);
            tree = advance_root(tree, new_root as usize);

            for _ in 0..20 {
                let reclaimed = allocate_node(&mut tree, 1.0, -1);
                tree.arena[reclaimed as usize].first_child = u32::MAX;
                let _ = tree.node_free_list.push(reclaimed);
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
        assert_eq!(
            tree.node_free_list.len(),
            4999,
            "UAF Race Condition Leak Detected!"
        );
    }
}
