use pyo3::prelude::*;
use rand::{Rng, thread_rng};

#[pyclass]
pub struct SegmentTree {
    capacity: usize,
    tree_cap: usize,
    tree: Vec<f64>,
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
        (0..batch_size).map(|_| {
            let v = rng.gen_range(0.0..=total);
            self.get_leaf(v)
        }).collect()
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

    pub fn sample(&self, batch_size: usize, capacity: usize) -> Option<(Vec<(usize, f64)>, Vec<f32>)> {
        let total_p = self.tree.total();
        if total_p == 0.0 {
            return None;
        }

        let samples = self.tree.sample_proportional(batch_size);
        let mut weights = Vec::with_capacity(batch_size);

        for &(_idx, p) in &samples {
            let p_i = p / total_p;
            let weight = ((capacity as f64 * (p_i + 1e-8)).powf(-self.beta)) as f32;
            weights.push(weight);
        }

        Some((samples, weights))
    }

    pub fn update_priorities(&mut self, circ_indices: &[usize], diff_penalties: &[f64], priorities: &[f64]) {
        for i in 0..circ_indices.len() {
            let mut p = priorities[i];
            if p > self.max_priority {
                self.max_priority = p;
            }
            if p < 1e-4 { p = 1e-4; }
            
            let final_p = p.powf(self.alpha) * diff_penalties[i];
            self.tree.update(circ_indices[i], final_p);
        }
    }
}
