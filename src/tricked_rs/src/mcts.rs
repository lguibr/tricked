use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use std::collections::HashMap;

use rand::Rng;

use std::sync::Mutex;
use once_cell::sync::Lazy;

pub static MODEL_CACHE: Lazy<Mutex<Option<tch::CModule>>> = Lazy::new(|| Mutex::new(None));

#[pyfunction]
pub fn init_model(path: String) -> PyResult<()> {
    let m = tch::CModule::load(&path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    *MODEL_CACHE.lock().unwrap() = Some(m);
    Ok(())
}

use crate::node::{LatentNode, get_valid_action_mask, select_child};

#[pyfunction]
#[pyo3(signature = (h0_bytes, policy_bytes, state, simulations, max_gumbel_k, gumbel_scale))]
pub fn mcts_search(
    py: Python,
    h0_bytes: &[u8],
    policy_bytes: &[u8],
    state: &crate::GameStateExt,
    simulations: usize,
    max_gumbel_k: usize,
    gumbel_scale: f32,
) -> PyResult<(i32, HashMap<i32, i32>, f32)> {
    let h0: Vec<f32> = h0_bytes.chunks_exact(4).map(|b| f32::from_ne_bytes(b.try_into().unwrap())).collect();
    let policy_probs: Vec<f32> = policy_bytes.chunks_exact(4).map(|b| f32::from_ne_bytes(b.try_into().unwrap())).collect();

    let valid_action_mask = get_valid_action_mask(state);
    let mut masked_probs = Vec::with_capacity(288);
    let mut num_valid = 0;
    
    for (i, &p) in policy_probs.iter().enumerate() {
        if valid_action_mask[i] {
            masked_probs.push(p.max(1e-8));
            num_valid += 1;
        } else {
            masked_probs.push(0.0);
        }
    }
    
    let sum_probs: f32 = masked_probs.iter().sum();
    if sum_probs > 0.0 {
        for p in masked_probs.iter_mut() { *p /= sum_probs; }
    } else if num_valid > 0 {
        for (i, p) in masked_probs.iter_mut().enumerate() {
            if valid_action_mask[i] { *p = 1.0 / (num_valid as f32); }
        }
    }

    let mut arena = vec![LatentNode::new(1.0)];
    let root_idx = 0;
    
    arena[root_idx].hidden_state = Some(h0);
    arena[root_idx].reward = 0.0;
    arena[root_idx].is_expanded = true;
    for (act_idx, &prob) in masked_probs.iter().enumerate() {
        if prob > 0.0 {
            let child_idx = arena.len();
            arena.push(LatentNode::new(prob));
            arena[root_idx].children.insert(act_idx as i32, child_idx);
        }
    }

    let valid_actions: Vec<i32> = valid_action_mask.iter().enumerate().filter(|&(_, &m)| m).map(|(i, _)| i as i32).collect();
    if valid_actions.is_empty() {
        return Ok((-1, HashMap::new(), 0.0));
    }

    let density = 1.0 - (num_valid as f32 / 288.0);
    let k_dynamic = 4 + ((max_gumbel_k as f32 - 4.0) * density) as usize;
    let k = k_dynamic.min(num_valid);

    if k == 1 {
        let mut map = HashMap::new();
        map.insert(valid_actions[0], 1);
        return Ok((valid_actions[0], map, arena[root_idx].value()));
    }

    let mut rng = rand::thread_rng();
    for &act in &valid_actions {
        let u: f32 = rng.gen_range(1e-6..=(1.0 - 1e-6));
        let gumbel = -( -( u.ln() ) ).ln();
        if let Some(&child_idx) = arena[root_idx].children.get(&act) {
            arena[child_idx].gumbel_noise = gumbel;
        }
    }

    let mut gumbel_pi = vec![std::f32::NEG_INFINITY; 288];
    for &act in &valid_actions {
        let act_usize = act as usize;
        let p = masked_probs[act_usize];
        let p_log = (p + 1e-8).ln();
        if let Some(&child_idx) = arena[root_idx].children.get(&act) {
            gumbel_pi[act_usize] = p_log + (arena[child_idx].gumbel_noise * gumbel_scale);
        }
    }

    let mut candidates = valid_actions.clone();
    candidates.sort_by(|&a, &b| gumbel_pi[b as usize].partial_cmp(&gumbel_pi[a as usize]).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(k);

    let phases = if k > 0 { (k as f32).log2().ceil() as usize } else { 0 };
    let sims_per_phase = if phases > 0 { simulations / phases } else { simulations };

    let state_available = state.available.clone();
    
    let res = py.detach(move || -> Result<(i32, HashMap<i32, i32>, f32), String> {
        let lock = MODEL_CACHE.lock().unwrap();
        let model = lock.as_ref().ok_or_else(|| "Model not initialized. Call init_model() first.".to_string())?;

        for _phase in 0..phases {
            let num_candidates = candidates.len();
            if num_candidates <= 1 { break; }
            let mut visits_per_candidate = sims_per_phase / num_candidates;
            if visits_per_candidate == 0 { visits_per_candidate = 1; }

            for &cand_action in &candidates {
                for _ in 0..visits_per_candidate {
                    let mut search_path = vec![root_idx];
                    let mut actions = vec![];
                    
                    let mut curr_node = root_idx;
                    if let Some(&child_idx) = arena[curr_node].children.get(&cand_action) {
                        actions.push(cand_action);
                        search_path.push(child_idx);
                        curr_node = child_idx;
                    } else {
                        break;
                    }

                    while arena[curr_node].is_expanded {
                        let (act, next_node) = select_child(&arena, curr_node, false);
                        if next_node == usize::MAX { break; }
                        actions.push(act);
                        search_path.push(next_node);
                        curr_node = next_node;
                    }

                    let parent_idx = search_path[search_path.len() - 2];
                    let last_action = actions.last().unwrap();
                    let slot = last_action / 96;
                    let pos = last_action % 96;
                    let mut piece_id = state_available[slot as usize];
                    if piece_id == -1 { piece_id = 0; }
                    let piece_action = piece_id * 96 + pos;

                    let h_last = arena[parent_idx].hidden_state.as_ref().unwrap();
                    
                    // LibTorch Inference
                    let h_tensor = tch::Tensor::from_slice(h_last).view([1, 128, 96]);
                    let a_tensor = tch::Tensor::from_slice(&[piece_action as i64]).view([1]);
                    let p_tensor = tch::Tensor::from_slice(&[piece_id as i64]).view([1]);

                    let out = model.forward_is(&[
                        tch::IValue::Tensor(h_tensor),
                        tch::IValue::Tensor(a_tensor),
                        tch::IValue::Tensor(p_tensor)
                    ]).map_err(|e| e.to_string())?;
                    
                    let out_tuple = match out {
                        tch::IValue::Tuple(t) => t,
                        _ => return Err("Model output is not a tuple".to_string()),
                    };

                    // Extract output tensors
                    let h_next_tensor = match &out_tuple[0] { tch::IValue::Tensor(t) => t, _ => return Err("Err".to_string()) };
                    let r_tensor = match &out_tuple[1] { tch::IValue::Tensor(t) => t, _ => return Err("Err".to_string()) };
                    let v_tensor = match &out_tuple[2] { tch::IValue::Tensor(t) => t, _ => return Err("Err".to_string()) };
                    let p_tensor = match &out_tuple[3] { tch::IValue::Tensor(t) => t, _ => return Err("Err".to_string()) };
                    
                    let h_next: Vec<f32> = h_next_tensor.flatten(0, -1).try_into().unwrap();
                    let reward: f32 = r_tensor.double_value(&[0]) as f32;
                    let value: f32 = v_tensor.double_value(&[0]) as f32;
                    let p_next: Vec<f32> = p_tensor.flatten(0, -1).try_into().unwrap();

                    let leaf_idx = curr_node;
                    arena[leaf_idx].hidden_state = Some(h_next);
                    arena[leaf_idx].reward = reward;
                    arena[leaf_idx].is_expanded = true;
                    for (act_idx, &prob) in p_next.iter().enumerate() {
                        if prob > 0.0 {
                            let new_child = arena.len();
                            arena.push(LatentNode::new(prob));
                            arena[leaf_idx].children.insert(act_idx as i32, new_child);
                        }
                    }

                    let mut v = value;
                    for &node_idx in search_path.iter().rev() {
                        arena[node_idx].visits += 1;
                        arena[node_idx].value_sum += v;
                        v = arena[node_idx].reward + 0.99 * v;
                    }
                }
            }

            candidates.sort_by(|&a, &b| {
                let ca = &arena[arena[root_idx].children[&a]];
                let cb = &arena[arena[root_idx].children[&b]];
                let qa = ca.reward + 0.99 * ca.value();
                let qb = cb.reward + 0.99 * cb.value();
                
                let c_scale_a = 50.0 / ((ca.visits + 1) as f32);
                let score_a = gumbel_pi[a as usize] + (c_scale_a * qa);
                
                let c_scale_b = 50.0 / ((cb.visits + 1) as f32);
                let score_b = gumbel_pi[b as usize] + (c_scale_b * qb);
                
                score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
            });
            let drop_count = candidates.len() / 2;
            candidates.truncate(candidates.len() - drop_count);
        }

        let mut evaluated_k = Vec::new();
        for (&act, &child_idx) in &arena[root_idx].children {
            if arena[child_idx].visits > 0 {
                evaluated_k.push(act);
            }
        }
        
        if evaluated_k.is_empty() {
            let mut map = HashMap::new();
            map.insert(candidates[0], 1);
            return Ok((candidates[0], map, arena[root_idx].value()));
        }

        let mut q_values = Vec::new();
        let mut max_q = std::f32::NEG_INFINITY;
        let mut min_q = std::f32::INFINITY;

        for &act in &evaluated_k {
            let child_idx = arena[root_idx].children[&act];
            let q = arena[child_idx].reward + 0.99 * arena[child_idx].value();
            q_values.push(q);
            if q > max_q { max_q = q; }
            if q < min_q { min_q = q; }
        }

        let q_range = if max_q > min_q { max_q - min_q } else { 1.0 };
        let mut exp_sum = 0.0;
        let mut q_probs = Vec::new();
        for &q in &q_values {
            let exp_q = ((q - max_q) / q_range).exp();
            q_probs.push(exp_q);
            exp_sum += exp_q;
        }

        let mut visits = HashMap::new();
        for (i, &act) in evaluated_k.iter().enumerate() {
            let p = q_probs[i] / exp_sum;
            let v = (p * (simulations as f32)) as i32;
            visits.insert(act, v.max(1));
        }

        // Fixed Gumbel Top-K completed selection
        let mut best_action = candidates[0];
        let mut best_score = std::f32::NEG_INFINITY;
        for &act in &evaluated_k {
            let child_idx = arena[root_idx].children[&act];
            let q = arena[child_idx].reward + 0.99 * arena[child_idx].value();
            let c_scale = 50.0 / ((arena[child_idx].visits + 1) as f32);
            let score = gumbel_pi[act as usize] + c_scale * q;
            if score > best_score {
                best_score = score;
                best_action = act;
            }
        }

        Ok((best_action, visits, arena[root_idx].value()))
    });

    match res {
        Ok(v) => Ok(v),
        Err(e) => Err(PyRuntimeError::new_err(e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GameStateExt;

    #[test]
    fn test_latent_node() {
        let node = LatentNode::new(0.5);
        assert_eq!(node.value(), 0.0);
        
        let mut node2 = LatentNode::new(0.5);
        node2.visits = 2;
        node2.value_sum = 1.0;
        assert_eq!(node2.value(), 0.5);
    }

    #[test]
    fn test_valid_action_mask() {
        let mut state = GameStateExt::new(Some(vec![0, 0, 0]), 0, 0, 6, 0);
        let mask = get_valid_action_mask(&state);
        assert!(mask.contains(&true));
        
        state.terminal = true;
        let terminal_mask = get_valid_action_mask(&state);
        assert!(!terminal_mask.contains(&true));
    }
    
    #[test]
    fn test_select_child() {
        let mut arena = vec![LatentNode::new(1.0)];
        arena[0].children.insert(10, 1);
        arena[0].children.insert(20, 2);
        
        arena.push(LatentNode::new(0.3)); 
        arena.push(LatentNode::new(0.7)); 
        
        let (best_action, best_child) = select_child(&arena, 0, false);
        assert_eq!(best_action, 20);
        assert_eq!(best_child, 2);
        
        let (root_action, root_child) = select_child(&arena, 0, true);
        assert_eq!(root_action, 20);
        assert_eq!(root_child, 2);
    }
}
