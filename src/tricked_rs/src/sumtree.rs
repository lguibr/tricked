use pyo3::prelude::*;
use rand::{thread_rng, Rng};
use std::sync::Mutex;

#[pyclass]
pub struct SegmentTree {
    capacity: usize,
    tree_cap: usize,
    pub tree: Vec<f64>,
}

#[pymethods]
impl SegmentTree {
    #[new]
    pub fn new(capacity: usize) -> Self {
        let mut tree_cap = 1;
        while tree_cap < capacity {
            tree_cap *= 2;
        }
        Self {
            capacity,
            tree_cap,
            tree: vec![0.0; 2 * tree_cap],
        }
    }

    pub fn update(&mut self, idx: usize, p: f64) {
        let mut tree_idx = idx + self.tree_cap;
        let change = p - self.tree[tree_idx];
        while tree_idx > 0 {
            self.tree[tree_idx] += change;
            tree_idx /= 2;
        }
    }

    pub fn total(&self) -> f64 {
        self.tree[1]
    }

    pub fn get_leaf(&self, mut v: f64) -> (usize, f64) {
        let mut idx = 1;
        while idx < self.tree_cap {
            let left = 2 * idx;
            if v <= self.tree[left] {
                idx = left;
            } else {
                v -= self.tree[left];
                idx = left + 1;
            }
        }
        let data_idx = idx - self.tree_cap;
        (data_idx, self.tree[idx])
    }

    pub fn sample_proportional(&self, batch_size: usize) -> Vec<(usize, f64)> {
        let mut rng = thread_rng();
        let total = self.total();
        (0..batch_size)
            .map(|_| {
                let v = rng.gen_range(0.0..=total);
                self.get_leaf(v)
            })
            .collect()
    }
}

pub struct PrioritizedReplay {
    pub tree: SegmentTree,
    pub max_priority: f64,
    pub alpha: f64,
    pub beta: f64,
}

impl PrioritizedReplay {
    pub fn new(capacity: usize, alpha: f64, beta: f64) -> Self {
        Self {
            tree: SegmentTree::new(capacity),
            max_priority: 10.0,
            alpha,
            beta,
        }
    }

    pub fn add(&mut self, idx: usize, diff_penalty: f64) {
        let p = self.max_priority.powf(self.alpha) * diff_penalty;
        self.tree.update(idx, p);
    }
}

pub struct ShardedPrioritizedReplay {
    shards: Vec<Mutex<PrioritizedReplay>>,
    num_shards: usize,
}

impl ShardedPrioritizedReplay {
    pub fn new(capacity: usize, alpha: f64, beta: f64, num_shards: usize) -> Self {
        let mut shards = Vec::with_capacity(num_shards);
        let shard_capacity = capacity / num_shards + 1;
        for _ in 0..num_shards {
            shards.push(Mutex::new(PrioritizedReplay::new(
                shard_capacity,
                alpha,
                beta,
            )));
        }
        Self { shards, num_shards }
    }

    pub fn add(&self, circ_idx: usize, diff_penalty: f64) {
        let shard_idx = circ_idx % self.num_shards;
        let internal_idx = circ_idx / self.num_shards;
        let mut shard = self.shards[shard_idx].lock().unwrap();
        shard.add(internal_idx, diff_penalty);
    }

    pub fn add_batch(&self, circ_indices: &[usize], diff_penalties: &[f64]) {
        let mut shard_adds = vec![(Vec::new(), Vec::new()); self.num_shards];
        for i in 0..circ_indices.len() {
            let circ_idx = circ_indices[i];
            let shard_idx = circ_idx % self.num_shards;
            let internal_idx = circ_idx / self.num_shards;
            shard_adds[shard_idx].0.push(internal_idx);
            shard_adds[shard_idx].1.push(diff_penalties[i]);
        }
        for shard_idx in 0..self.num_shards {
            let adds = &shard_adds[shard_idx];
            if !adds.0.is_empty() {
                let mut shard = self.shards[shard_idx].lock().unwrap();
                for i in 0..adds.0.len() {
                    shard.add(adds.0[i], adds.1[i]);
                }
            }
        }
    }

    pub fn sample(
        &self,
        batch_size: usize,
        global_capacity: usize,
    ) -> Option<(Vec<(usize, f64)>, Vec<f32>)> {
        let shard_idx = thread_rng().gen_range(0..self.num_shards);
        let shard = self.shards[shard_idx].lock().unwrap();

        let total_p = shard.tree.total();
        if total_p == 0.0 {
            return None;
        }

        let samples = shard.tree.sample_proportional(batch_size);
        let mut weights = Vec::with_capacity(batch_size);
        let mut out_samples = Vec::with_capacity(batch_size);

        for &(idx, p) in &samples {
            let p_i = p / total_p;
            // The global distribution is roughly num_shards times the shard distribution
            let p_global = p_i / (self.num_shards as f64);
            let weight = ((global_capacity as f64 * (p_global + 1e-8)).powf(-shard.beta)) as f32;
            weights.push(weight);

            // Map internal index back to global circ_idx
            out_samples.push((idx * self.num_shards + shard_idx, p));
        }

        Some((out_samples, weights))
    }

    pub fn update_priorities(
        &self,
        circ_indices: &[usize],
        diff_penalties: &[f64],
        priorities: &[f64],
    ) {
        let mut shard_updates = vec![(Vec::new(), Vec::new(), Vec::new()); self.num_shards];

        for i in 0..circ_indices.len() {
            let circ_idx = circ_indices[i];
            let shard_idx = circ_idx % self.num_shards;
            let internal_idx = circ_idx / self.num_shards;

            shard_updates[shard_idx].0.push(internal_idx);
            shard_updates[shard_idx].1.push(diff_penalties[i]);
            shard_updates[shard_idx].2.push(priorities[i]);
        }

        for shard_idx in 0..self.num_shards {
            let updates = &shard_updates[shard_idx];
            if !updates.0.is_empty() {
                let mut shard = self.shards[shard_idx].lock().unwrap();
                for i in 0..updates.0.len() {
                    let mut p = updates.2[i];
                    if p > shard.max_priority {
                        shard.max_priority = p;
                    }
                    if p < 1e-4 {
                        p = 1e-4;
                    }

                    let final_p = p.powf(shard.alpha) * updates.1[i];
                    shard.tree.update(updates.0[i], final_p);
                }
            }
        }
    }
}
