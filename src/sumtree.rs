pub type SumTreeSample = (Vec<(usize, f64)>, Vec<f32>);
#[cfg(loom)]
use loom::sync::atomic::{AtomicU64, Ordering};
use rand::{thread_rng, Rng};
#[cfg(not(loom))]
use std::sync::atomic::{AtomicU64, Ordering};

fn update_max_priority(atom: &AtomicU64, new_val: f64) {
    if new_val >= 0.0 {
        atom.fetch_max(new_val.to_bits(), Ordering::Relaxed);
    }
}

pub struct SegmentTree {
    _buffer_capacity_limit: usize,
    tree_buffer_capacity_limit: usize,
    pub tree_array: Vec<AtomicU64>,
    pub lock_contention_nanos: AtomicU64,
    batch_deltas: std::sync::Mutex<Vec<f64>>,
}

impl SegmentTree {
    pub fn new(buffer_capacity_limit: usize) -> Self {
        let mut tree_buffer_capacity_limit = 1;
        while tree_buffer_capacity_limit < buffer_capacity_limit {
            tree_buffer_capacity_limit *= 2;
        }

        let mut tree_array = Vec::with_capacity(2 * tree_buffer_capacity_limit);
        for _ in 0..(2 * tree_buffer_capacity_limit) {
            tree_array.push(AtomicU64::new(0.0f64.to_bits()));
        }

        let batch_deltas = std::sync::Mutex::new(vec![0.0; tree_buffer_capacity_limit]);

        Self {
            _buffer_capacity_limit: buffer_capacity_limit,
            tree_buffer_capacity_limit,
            tree_array,
            lock_contention_nanos: AtomicU64::new(0),
            batch_deltas,
        }
    }

    pub fn update(&self, data_index: usize, priority_value: f64) {
        assert!(
            priority_value.is_finite(),
            "Segment tree update received non-finite priority"
        );
        assert!(
            priority_value >= 0.0,
            "Segment tree priority cannot be negative"
        );
        assert!(
            data_index < self.tree_buffer_capacity_limit,
            "Segment tree update index {} violates sumtree array bounds {}!",
            data_index,
            self.tree_buffer_capacity_limit
        );

        let mut tree_index = data_index + self.tree_buffer_capacity_limit;
        let old_bits = self.tree_array[tree_index].swap(priority_value.to_bits(), Ordering::SeqCst);
        let old_val = f64::from_bits(old_bits);
        let delta_change = priority_value - old_val;

        tree_index /= 2;
        while tree_index > 0 {
            let atom = &self.tree_array[tree_index];
            let mut current_bits = atom.load(Ordering::Relaxed);
            loop {
                let current_val = f64::from_bits(current_bits);
                let new_val = current_val + delta_change;
                match atom.compare_exchange_weak(
                    current_bits,
                    new_val.to_bits(),
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(actual) => current_bits = actual,
                }
            }
            tree_index /= 2;
        }
    }

    pub fn update_batch(&self, updates: &[(usize, f64)]) {
        let start = std::time::Instant::now();

        let mut deltas = self.batch_deltas.lock().unwrap();

        for &(data_index, priority_value) in updates {
            assert!(
                priority_value.is_finite(),
                "Segment tree update received non-finite priority"
            );
            assert!(
                priority_value >= 0.0,
                "Segment tree priority cannot be negative"
            );
            assert!(
                data_index < self.tree_buffer_capacity_limit,
                "Segment tree update index {} violates sumtree array bounds {}!",
                data_index,
                self.tree_buffer_capacity_limit
            );

            let tree_index = data_index + self.tree_buffer_capacity_limit;
            let old_bits =
                self.tree_array[tree_index].swap(priority_value.to_bits(), Ordering::SeqCst);
            let old_val = f64::from_bits(old_bits);
            let delta = priority_value - old_val;

            if delta != 0.0 {
                let parent_idx = tree_index / 2;
                deltas[parent_idx] += delta;
            }
        }

        for node_idx in (1..self.tree_buffer_capacity_limit).rev() {
            let delta = deltas[node_idx];
            if delta != 0.0 {
                let atom = &self.tree_array[node_idx];
                let mut current_bits = atom.load(Ordering::Relaxed);
                loop {
                    let current_val = f64::from_bits(current_bits);
                    let new_val = current_val + delta;
                    match atom.compare_exchange_weak(
                        current_bits,
                        new_val.to_bits(),
                        Ordering::SeqCst,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(actual) => current_bits = actual,
                    }
                }

                if node_idx > 1 {
                    let parent_idx = node_idx / 2;
                    deltas[parent_idx] += delta;
                }

                deltas[node_idx] = 0.0;
            }
        }

        let contention = start.elapsed().as_nanos() as u64;
        self.lock_contention_nanos
            .fetch_add(contention, Ordering::Relaxed);
    }

    pub fn get_total_priority(&self) -> f64 {
        f64::from_bits(self.tree_array[1].load(Ordering::Relaxed))
    }

    pub fn get_leaf(&self, mut target_value: f64) -> (usize, f64) {
        let mut tree_index = 1;

        while tree_index < self.tree_buffer_capacity_limit {
            let left_child_index = 2 * tree_index;
            let left_val =
                f64::from_bits(self.tree_array[left_child_index].load(Ordering::Relaxed));

            if target_value <= left_val {
                tree_index = left_child_index;
            } else {
                target_value -= left_val;
                tree_index = left_child_index + 1;
            }
        }
        let data_index = tree_index - self.tree_buffer_capacity_limit;
        let val = f64::from_bits(self.tree_array[tree_index].load(Ordering::Relaxed));
        (data_index, val)
    }

    pub fn sample_proportional(&self, batch_size: usize) -> Vec<(usize, f64)> {
        let mut random_generator = thread_rng();
        let total_priority = self.get_total_priority();

        if total_priority <= 0.0 || !total_priority.is_finite() {
            return vec![(0, 0.0); batch_size];
        }

        (0..batch_size)
            .map(|_| {
                let target_value = random_generator.gen_range(0.0..=total_priority);
                self.get_leaf(target_value)
            })
            .collect()
    }
}

pub struct PrioritizedReplay {
    pub segment_tree: SegmentTree,
    pub maximum_priority: AtomicU64,
    pub alpha_factor: f64,
    #[allow(dead_code)]
    pub beta_factor: f64,
}

impl PrioritizedReplay {
    pub fn new(buffer_capacity_limit: usize, alpha_factor: f64, beta_factor: f64) -> Self {
        Self {
            segment_tree: SegmentTree::new(buffer_capacity_limit),
            maximum_priority: AtomicU64::new(10.0f64.to_bits()),
            alpha_factor,
            beta_factor,
        }
    }

    pub fn add_experience(&self, data_index: usize, difficulty_penalty: f64) {
        let max_p = f64::from_bits(self.maximum_priority.load(Ordering::Relaxed));
        let priority_value = max_p.powf(self.alpha_factor) * difficulty_penalty;
        self.segment_tree.update(data_index, priority_value);
    }
}

pub struct ShardedPrioritizedReplay {
    shards: Vec<PrioritizedReplay>,
    shard_count: usize,
}

impl ShardedPrioritizedReplay {
    pub fn new(
        buffer_capacity_limit: usize,
        alpha_factor: f64,
        beta_factor: f64,
        shard_count: usize,
    ) -> Self {
        let mut shards = Vec::with_capacity(shard_count);
        let shard_buffer_capacity_limit = buffer_capacity_limit / shard_count + 1;
        for _ in 0..shard_count {
            shards.push(PrioritizedReplay::new(
                shard_buffer_capacity_limit,
                alpha_factor,
                beta_factor,
            ));
        }
        Self {
            shards,
            shard_count,
        }
    }

    #[allow(dead_code)]
    pub fn add(&self, circular_index: usize, difficulty_penalty: f64) {
        let shard_index = circular_index % self.shard_count;
        let internal_index = circular_index / self.shard_count;
        self.shards[shard_index].add_experience(internal_index, difficulty_penalty);
    }

    pub fn add_batch(&self, circular_indices: &[usize], difficulty_penalties: &[f64]) {
        let mut shard_operations = vec![(Vec::new(), Vec::new()); self.shard_count];
        for iterator_index in 0..circular_indices.len() {
            let circular_index = circular_indices[iterator_index];
            let shard_index = circular_index % self.shard_count;
            let internal_index = circular_index / self.shard_count;
            shard_operations[shard_index].0.push(internal_index);
            shard_operations[shard_index]
                .1
                .push(difficulty_penalties[iterator_index]);
        }

        for (shard_index, operations) in shard_operations.iter().enumerate() {
            if !operations.0.is_empty() {
                for iterator_index in 0..operations.0.len() {
                    self.shards[shard_index]
                        .add_experience(operations.0[iterator_index], operations.1[iterator_index]);
                }
            }
        }
    }

    pub fn sample(
        &self,
        batch_size: usize,
        global_buffer_capacity_limit: usize,
        beta: f64,
    ) -> Option<SumTreeSample> {
        let shard_index = thread_rng().gen_range(0..self.shard_count);
        let shard = &self.shards[shard_index];

        let total_priority = shard.segment_tree.get_total_priority();
        if total_priority <= 0.0 || !total_priority.is_finite() {
            return None;
        }

        let shard_samples = shard.segment_tree.sample_proportional(batch_size);
        let mut importance_weights = Vec::with_capacity(batch_size);
        let mut output_samples = Vec::with_capacity(batch_size);

        let theoretical_min_priority = 1e-4;
        let p_min_global = (theoretical_min_priority / total_priority) / (self.shard_count as f64);
        let max_theoretical_weight =
            ((global_buffer_capacity_limit as f64 * (p_min_global + 1e-8)).powf(-beta)) as f32;

        for &(data_index, priority_value) in &shard_samples {
            let sample_probability = priority_value / total_priority;
            let global_probability = sample_probability / (self.shard_count as f64);
            let importance_weight = ((global_buffer_capacity_limit as f64
                * (global_probability + 1e-8))
                .powf(-beta)) as f32;

            importance_weights.push(importance_weight / max_theoretical_weight);
            output_samples.push((data_index * self.shard_count + shard_index, priority_value));
        }

        Some((output_samples, importance_weights))
    }

    pub fn update_priorities(
        &self,
        circular_indices: &[usize],
        difficulty_penalties: &[f64],
        new_priorities: &[f64],
    ) {
        let mut shard_updates = vec![(Vec::new(), Vec::new(), Vec::new()); self.shard_count];

        for iterator_index in 0..circular_indices.len() {
            let circular_index = circular_indices[iterator_index];
            let shard_index = circular_index % self.shard_count;
            let internal_index = circular_index / self.shard_count;

            shard_updates[shard_index].0.push(internal_index);
            shard_updates[shard_index]
                .1
                .push(difficulty_penalties[iterator_index]);
            shard_updates[shard_index]
                .2
                .push(new_priorities[iterator_index]);
        }

        for (shard_index, updates) in shard_updates.iter().enumerate() {
            if !updates.0.is_empty() {
                let shard = &self.shards[shard_index];
                let mut batch_updates = Vec::with_capacity(updates.0.len());
                for iterator_index in 0..updates.0.len() {
                    let mut priority_value = updates.2[iterator_index];

                    if !priority_value.is_finite() {
                        priority_value = 1e-4;
                    }

                    update_max_priority(&shard.maximum_priority, priority_value);

                    if priority_value < 1e-4 {
                        priority_value = 1e-4;
                    }

                    let final_priority_value =
                        priority_value.powf(shard.alpha_factor) * updates.1[iterator_index];
                    batch_updates.push((updates.0[iterator_index], final_priority_value));
                }
                shard.segment_tree.update_batch(&batch_updates);
            }
        }
    }

    pub fn get_and_reset_contention(&self) -> u64 {
        self.shards
            .iter()
            .map(|shard| {
                shard
                    .segment_tree
                    .lock_contention_nanos
                    .swap(0, std::sync::atomic::Ordering::Relaxed)
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn fuzz_segment_tree_math(priorities in prop::collection::vec(0.1f64..100.0, 1..500)) {
            let config_len = priorities.len();
            let tree = SegmentTree::new(config_len);

            let mut expected_sum = 0.0;
            for (i, &p) in priorities.iter().enumerate() {
                tree.update(i, p);
                expected_sum += p;
            }

            let float_error_margin = 0.05;
            let total = tree.get_total_priority();
            assert!((total - expected_sum).abs() < float_error_margin, "SumTree didn't accumulate exactly. {} != {}", total, expected_sum);

            let mut running_sum = 0.0;
            for (i, &p) in priorities.iter().enumerate() {
                running_sum += p;
                let target = running_sum - (p / 2.0); // middle of the slice
                let (idx, val) = tree.get_leaf(target);
                assert_eq!(idx, i, "SumTree search fell into wrong bucket. Target {} landed in {}, expected {}", target, idx, i);
                assert!((val - p).abs() < float_error_margin);
            }
        }
    }

    #[test]
    fn test_segment_tree_updates_and_zero_sum() {
        let tree = SegmentTree::new(4);

        let samples = tree.sample_proportional(2);
        assert_eq!(
            samples.len(),
            2,
            "Fallback for zero-sum tree should return empty placeholder batch"
        );
        assert_eq!(samples[0].0, 0, "Fallback index should be 0");
        assert_eq!(samples[0].1, 0.0, "Fallback weight should be zero");

        tree.update(0, 1.0);
        tree.update(1, 2.0);
        tree.update(2, 3.0);

        assert_eq!(
            tree.get_total_priority(),
            6.0,
            "Sumtree total propagation failed"
        );

        tree.update(1, 0.5);
        assert_eq!(
            tree.get_total_priority(),
            4.5,
            "Sumtree modification update tracking mathematical bug"
        );
    }

    #[test]
    fn test_batch_zero_allocation_dense_update() {
        let tree = SegmentTree::new(16);
        let batch_updates = vec![(0, 2.0), (1, 3.0), (5, 5.0), (10, 10.0)];
        tree.update_batch(&batch_updates);

        assert_eq!(tree.get_total_priority(), 20.0);

        // Assert the internal delta buffer has been perfectly zeroed out back to memory-safe state
        let lock = tree.batch_deltas.lock().unwrap();
        for &val in lock.iter() {
            assert_eq!(val, 0.0, "Delta buffer was not perfectly zeroed out!");
        }
    }

    #[test]
    fn test_sharded_per_weighting() {
        let per = ShardedPrioritizedReplay::new(10, 1.0, 1.0, 2);
        per.add_batch(&[0, 1, 2], &[1.0, 2.0, 3.0]);
        let mut successful_sample = false;

        for _ in 0..100 {
            if let Some((_, weights)) = per.sample(2, 10, 1.0) {
                assert_eq!(weights.len(), 2, "Sampled weights mismatch");
                successful_sample = true;
                break;
            }
        }

        assert!(
            successful_sample,
            "Should have sampled from populated PER shard"
        );
    }
}
