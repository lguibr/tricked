use proptest::prelude::*;
use rand::{thread_rng, Rng};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub struct SegmentTreeMutex {
    pub tree_buffer_capacity_limit: usize,
    pub tree_array: Vec<AtomicU64>,
    batch_deltas: Mutex<Vec<f64>>,
}

impl SegmentTreeMutex {
    pub fn new(buffer_capacity_limit: usize) -> Self {
        let mut capacity = 1;
        while capacity < buffer_capacity_limit {
            capacity *= 2;
        }
        let mut tree_array = Vec::with_capacity(2 * capacity);
        for _ in 0..(2 * capacity) {
            tree_array.push(AtomicU64::new(0.0f64.to_bits()));
        }
        let batch_deltas = Mutex::new(vec![0.0; capacity]);
        Self {
            tree_buffer_capacity_limit: capacity,
            tree_array,
            batch_deltas,
        }
    }

    pub fn update_batch(&self, updates: &[(usize, f64)]) {
        let mut deltas = self.batch_deltas.lock().unwrap();
        for &(data_index, priority_value) in updates {
            let tree_index = data_index + self.tree_buffer_capacity_limit;
            let old_bits = self.tree_array[tree_index].swap(priority_value.to_bits(), Ordering::SeqCst);
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
    }
}

thread_local! {
    static DELTA_BUFFER: std::cell::RefCell<Vec<f64>> = std::cell::RefCell::new(Vec::new());
}

pub struct SegmentTreeThreadLocal {
    pub tree_buffer_capacity_limit: usize,
    pub tree_array: Vec<AtomicU64>,
}

impl SegmentTreeThreadLocal {
    pub fn new(buffer_capacity_limit: usize) -> Self {
        let mut capacity = 1;
        while capacity < buffer_capacity_limit {
            capacity *= 2;
        }
        let mut tree_array = Vec::with_capacity(2 * capacity);
        for _ in 0..(2 * capacity) {
            tree_array.push(AtomicU64::new(0.0f64.to_bits()));
        }
        Self {
            tree_buffer_capacity_limit: capacity,
            tree_array,
        }
    }

    pub fn update_batch(&self, updates: &[(usize, f64)]) {
        DELTA_BUFFER.with(|tls_buffer| {
            let mut deltas = tls_buffer.borrow_mut();
            if deltas.len() < self.tree_buffer_capacity_limit {
                deltas.resize(self.tree_buffer_capacity_limit, 0.0);
            }
            
            for &(data_index, priority_value) in updates {
                let tree_index = data_index + self.tree_buffer_capacity_limit;
                let old_bits = self.tree_array[tree_index].swap(priority_value.to_bits(), Ordering::SeqCst);
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
        });
    }
}

fn main() {
    println!("Benchmarking Mutex vs ThreadLocal SegmentTree");
    let size = 15000;
    
    let mut batch = Vec::new();
    let mut rng = thread_rng();
    for _ in 0..128 {
        batch.push((rng.gen_range(0..size), rng.gen_range(0.0..10.0)));
    }
    
    let dt_start = Instant::now();
    let tree1 = Arc::new(SegmentTreeMutex::new(size));
    let mut threads = vec![];
    for _ in 0..8 {
        let t = tree1.clone();
        let b = batch.clone();
        threads.push(std::thread::spawn(move || {
            for _ in 0..1000 {
                t.update_batch(&b);
            }
        }));
    }
    for t in threads { t.join().unwrap(); }
    println!("Mutex time: {:?}", dt_start.elapsed());
    
    let dt_start = Instant::now();
    let tree2 = Arc::new(SegmentTreeThreadLocal::new(size));
    let mut threads = vec![];
    for _ in 0..8 {
        let t = tree2.clone();
        let b = batch.clone();
        threads.push(std::thread::spawn(move || {
            for _ in 0..1000 {
                t.update_batch(&b);
            }
        }));
    }
    for t in threads { t.join().unwrap(); }
    println!("ThreadLocal time: {:?}", dt_start.elapsed());
}
