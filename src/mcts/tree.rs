use crate::node::LatentNode;
use std::sync::atomic::Ordering;

use crossbeam_queue::ArrayQueue;

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

// Removed background GC pool to use synchronous bulk drop in advance_root.
// Restored background GC pool (GcTask) to prevent worker DFS stuttering.

pub struct GcTask {
    pub arena: SharedArena,
    pub node_free_list: Arc<ArrayQueue<u32>>,
    pub gpu_cache_free_list: Arc<ArrayQueue<u32>>,
    pub old_root: usize,
    pub new_root: usize,
}

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
                "MCTS Tree Arena ran out of nodes! Capacity {} is too small for the search depth.",
                tree.max_tree_nodes
            );
        }
        if attempts > 100 {
            std::thread::sleep(std::time::Duration::from_millis(1));
        } else {
            std::thread::yield_now();
        }
    };
    tree.arena[new_idx as usize].reset(probability, action, tree.current_generation);
    new_idx
}

pub fn gc_worker_loop(rx: crossbeam_channel::Receiver<GcTask>) {
    while let Ok(task) = rx.recv() {
        let mut child_idx = task.arena[task.old_root]
            .first_child
            .load(Ordering::Relaxed);
        while child_idx != u32::MAX {
            let next_sibling = task.arena[child_idx as usize]
                .next_sibling
                .load(Ordering::Relaxed);
            if child_idx as usize != task.new_root {
                let mut stack = vec![child_idx];
                while let Some(idx) = stack.pop() {
                    let node = &task.arena[idx as usize];
                    let mut c = node.first_child.load(Ordering::Relaxed);
                    while c != u32::MAX {
                        stack.push(c);
                        c = task.arena[c as usize].next_sibling.load(Ordering::Relaxed);
                    }
                    let _ = task.node_free_list.push(idx);
                    if node.hidden_state_index.load(Ordering::Relaxed) != u32::MAX {
                        let _ = task
                            .gpu_cache_free_list
                            .push(node.hidden_state_index.load(Ordering::Relaxed));
                    }
                }
            }
            child_idx = next_sibling;
        }

        let _ = task.node_free_list.push(task.old_root as u32);
        let state_idx = task.arena[task.old_root]
            .hidden_state_index
            .load(Ordering::Relaxed);
        if state_idx != u32::MAX {
            let _ = task.gpu_cache_free_list.push(state_idx);
        }
    }
}

pub fn advance_root(
    mut tree: MctsTree,
    new_root: usize,
    gc_tx: &crossbeam_channel::Sender<GcTask>,
) -> MctsTree {
    let old_root = tree.root_index;
    tree.current_generation += 1;
    let gen = tree.current_generation;
    tree.root_index = new_root;

    // BFS to update generations
    let mut queue = vec![new_root as u32];
    let mut head = 0;
    tree.arena[new_root].generation.store(gen, Ordering::SeqCst);

    while head < queue.len() {
        let node_idx = queue[head] as usize;
        head += 1;

        let mut child_idx = tree.arena[node_idx].first_child.load(Ordering::Relaxed);
        while child_idx != u32::MAX {
            tree.arena[child_idx as usize]
                .generation
                .store(gen, Ordering::SeqCst);
            queue.push(child_idx);
            child_idx = tree.arena[child_idx as usize]
                .next_sibling
                .load(Ordering::Relaxed);
        }
    }

    if old_root != new_root {
        let task = GcTask {
            arena: SharedArena(Arc::clone(&tree.arena.0)),
            node_free_list: Arc::clone(&tree.node_free_list),
            gpu_cache_free_list: Arc::clone(&tree.gpu_cache_free_list),
            old_root,
            new_root,
        };
        let _ = gc_tx.try_send(task);
    }

    tree
}

pub fn initialize_search_tree(
    previous_tree: Option<MctsTree>,
    last_executed_action: Option<i32>,
    mut max_tree_nodes: u32,
    mut max_cache_slots: u32,
    total_simulations: usize,
    gc_tx: &crossbeam_channel::Sender<GcTask>,
) -> MctsTree {
    // FIX: Guarantee the arena is mathematically large enough to survive a full turn.
    // 1 simulation expands at most 1 node (which has up to 288 children).
    let safe_min_nodes = (total_simulations as u32 * 288) + 5000;
    if max_tree_nodes < safe_min_nodes {
        max_tree_nodes = safe_min_nodes;
    }
    if max_cache_slots < safe_min_nodes {
        max_cache_slots = safe_min_nodes;
    }

    if let Some(existing_tree) = previous_tree {
        // Only reuse the tree if its capacity meets our new safe minimums
        if existing_tree.max_tree_nodes >= max_tree_nodes {
            if let Some(action) = last_executed_action {
                let child_index = existing_tree.arena[existing_tree.root_index]
                    .get_child(&existing_tree.arena, action);
                if child_index != usize::MAX {
                    let advanced_tree = advance_root(existing_tree, child_index, gc_tx);
                    let required_nodes = total_simulations * 288 + 1000;

                    if advanced_tree.node_free_list.len() > required_nodes
                        && advanced_tree.gpu_cache_free_list.len() > total_simulations + 10
                    {
                        return advanced_tree;
                    }
                }
            }
        }
    }

    // FIX: If we run out of nodes, allocate a completely fresh tree.
    // This prevents the GC Orphan Bug where the GC thread pushes to an abandoned free list.
    let capacity = max_tree_nodes as usize;
    let mut arena = Vec::with_capacity(capacity);
    for _ in 0..capacity {
        arena.push(LatentNode::new(0.0, -1, 0));
    }

    let node_free_list = Arc::new(ArrayQueue::new(max_tree_nodes as usize));
    for i in (1..max_tree_nodes).rev() {
        let _ = node_free_list.push(i);
    }

    let gpu_cache_free_list = Arc::new(ArrayQueue::new(max_cache_slots as usize));
    for i in (0..max_cache_slots).rev() {
        let _ = gpu_cache_free_list.push(i);
    }

    arena[0].reset(1.0, -1, 1);

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
    use crate::node::LatentNode;

    #[test]
    fn test_tree_arena_garbage_collection_leak_100000_steps() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        std::thread::spawn(move || {
            crate::mcts::tree::gc_worker_loop(_rx);
        });
        let max_nodes = 1000;
        let mut tree = initialize_search_tree(None, None, max_nodes, 1000, 500, &tx);

        for _step in 0..10_000 {
            let current_root = tree.root_index;
            let mut prev = u32::MAX;
            let mut first = u32::MAX;
            for i in 0..10 {
                let new_node = allocate_node(&mut tree, 0.1, i);
                tree.arena[new_node as usize]
                    .hidden_state_index
                    .store(i as u32, Ordering::SeqCst);
                if first == u32::MAX {
                    first = new_node;
                } else {
                    tree.arena[prev as usize]
                        .next_sibling
                        .store(new_node, Ordering::SeqCst);
                }
                prev = new_node;
            }
            tree.arena[current_root]
                .first_child
                .store(first, Ordering::SeqCst);

            let mut new_root = first;
            for _ in 0..4 {
                new_root = tree.arena[new_root as usize]
                    .next_sibling
                    .load(Ordering::Relaxed);
            }

            tree = advance_root(tree, new_root as usize, &tx);
            let mut attempts = 0;
            let expected_min = (tree.max_tree_nodes as usize).saturating_sub(100);
            while tree.node_free_list.len() < expected_min {
                std::thread::sleep(std::time::Duration::from_millis(1));
                attempts += 1;
                if attempts > 100 {
                    panic!(
                        "Nodes leaked! Expected >= {}, got {}",
                        expected_min,
                        tree.node_free_list.len()
                    );
                }
            }
        }
    }

    #[test]
    fn test_tree_arena_garbage_collection() {
        let (tx, _rx) = crossbeam_channel::unbounded();
        std::thread::spawn(move || {
            crate::mcts::tree::gc_worker_loop(_rx);
        });
        let mut tree = initialize_search_tree(None, None, 1000, 1000, 500, &tx);
        let new_root = 0;

        for i in 0..500 {
            let p_idx = allocate_node(&mut tree, 1.0, i as i16);
            if i > 0 {
                tree.arena[p_idx as usize].next_sibling.store(
                    tree.arena[new_root].first_child.load(Ordering::Relaxed),
                    Ordering::SeqCst,
                );
                tree.arena[new_root]
                    .first_child
                    .store(p_idx, Ordering::SeqCst);
                tree.arena[p_idx as usize]
                    .hidden_state_index
                    .store(i as u32, Ordering::SeqCst);
            }
        }

        let new_tree = advance_root(tree, new_root, &tx);

        std::thread::sleep(std::time::Duration::from_millis(50));

        let expected_free = new_tree.max_tree_nodes as usize - 501;
        assert_eq!(new_tree.node_free_list.len(), expected_free);
        assert_eq!(new_tree.root_index, 0);
    }

    #[test]
    fn test_tree_garbage_collection() {
        let mut mock_arena = Vec::with_capacity(10);
        for _ in 0..10 {
            mock_arena.push(LatentNode::new(0.0, 0, 0));
        }
        let root = LatentNode::new(1.0, -1, 0);
        root.is_topologically_expanded.store(true, Ordering::SeqCst);
        root.first_child.store(1, Ordering::SeqCst);
        mock_arena[0] = root;

        let child1 = LatentNode::new(0.5, 0, 0);
        child1
            .is_topologically_expanded
            .store(true, Ordering::SeqCst);
        child1.first_child.store(3, Ordering::SeqCst);
        child1.next_sibling.store(2, Ordering::SeqCst);
        child1.visits.store(10, Ordering::SeqCst);
        child1.hidden_state_index.store(5, Ordering::SeqCst);
        mock_arena[1] = child1;

        let child2 = LatentNode::new(0.5, 1, 0);
        child2.visits.store(5, Ordering::SeqCst);
        child2.hidden_state_index.store(6, Ordering::SeqCst);
        mock_arena[2] = child2;

        let grandchild = LatentNode::new(1.0, 2, 0);
        grandchild.visits.store(2, Ordering::SeqCst);
        grandchild.hidden_state_index.store(7, Ordering::SeqCst);
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

        let (tx, _rx) = crossbeam_channel::unbounded();
        std::thread::spawn(move || {
            crate::mcts::tree::gc_worker_loop(_rx);
        });
        let new_tree = advance_root(tree, 1, &tx);
        std::thread::sleep(std::time::Duration::from_millis(50));

        let actual_root = new_tree.root_index;
        let new_root_node = &new_tree.arena[actual_root];
        assert_eq!(
            new_root_node.visits.load(Ordering::Relaxed),
            10,
            "Visits must be retained"
        );

        let first_child_idx = new_root_node.first_child.load(Ordering::Relaxed) as usize;
        assert!(
            first_child_idx != u32::MAX as usize,
            "Children must be retained"
        );

        let new_grandchild = &new_tree.arena[first_child_idx];
        assert_eq!(
            new_grandchild.visits.load(Ordering::Relaxed),
            2,
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
        let (tx, _rx) = crossbeam_channel::unbounded();
        std::thread::spawn(move || {
            crate::mcts::tree::gc_worker_loop(_rx);
        });
        let mut tree = initialize_search_tree(None, None, 5000, 5000, 10, &tx);

        for _ in 0..100 {
            let mut last_node = tree.root_index as u32;
            for i in 0..20 {
                let next = allocate_node(&mut tree, 1.0, i as i16);
                tree.arena[last_node as usize]
                    .first_child
                    .store(next, Ordering::SeqCst);
                last_node = next;
            }

            let new_root = allocate_node(&mut tree, 1.0, 99);
            tree = advance_root(tree, new_root as usize, &tx);

            for _ in 0..20 {
                let reclaimed = allocate_node(&mut tree, 1.0, -1);
                tree.arena[reclaimed as usize]
                    .first_child
                    .store(u32::MAX, Ordering::SeqCst);
                let _ = tree.node_free_list.push(reclaimed);
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
        assert_eq!(
            tree.node_free_list.len(),
            tree.max_tree_nodes as usize - 1,
            "UAF Race Condition Leak Detected!"
        );
    }
}
