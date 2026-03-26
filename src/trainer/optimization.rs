use tch::{nn, Device, Kind, Tensor};
use crate::network::MuZeroNet;
use crate::buffer::ReplayBuffer;
use crate::config::Config;
use crate::trainer::loss::{
    binary_cross_entropy, negative_cosine_similarity, scale_gradient, soft_cross_entropy,
};

pub fn train_step(
    model: &MuZeroNet,
    ema_model: &MuZeroNet,
    opt: &mut nn::Optimizer,
    buffer: &ReplayBuffer,
    cfg: &Config,
    device: Device,
) {
    let batch_size = cfg.train_batch_size;
    let steps = cfg.unroll_steps as i64;

    let batch = match buffer.sample_batch(batch_size, device) {
        Some(b) => b,
        None => return,
    };

    let s_states = batch.b_states;
    let s_acts = batch.b_acts;
    let s_pids = batch.b_pids;
    let s_rews = batch.b_rews;
    let s_t_pols = batch.b_t_pols;
    let s_t_vals = batch.b_t_vals;
    let s_t_states = batch.b_t_states;
    let s_masks = batch.b_masks;
    let s_weights = batch.b_weights;
    let indices = batch.indices;

    opt.zero_grad();
    let max_w = s_weights.max();
    let scaled_weights = &s_weights / (max_w + 1e-8);

    let mut h = model.representation.forward(&s_states);
    h = scale_gradient(&h, 0.5);

    let (v_logits, p_logits, h_logits) = model.prediction.forward(&h);

    let v_targets_0 = model.scalar_to_support(&s_t_vals.select(1, 0));
    let v_loss_0 = soft_cross_entropy(&v_logits, &v_targets_0);
    
    let p_probs_0 = s_t_pols.select(1, 0) + 1e-8;
    let p_loss_0 = soft_cross_entropy(&p_logits, &p_probs_0);

    let mut bce_0 = binary_cross_entropy(&h_logits, &s_states.select(1, 19));
    if bce_0.dim() > 1 {
        bce_0 = bce_0.flatten(1, -1).mean_dim(&[1], false, Kind::Float);
    }

    let mut loss = &v_loss_0 + &p_loss_0 + (&bce_0 * 0.5);

    let mut tracker_v = v_loss_0.mean(Kind::Float);
    let mut tracker_p = p_loss_0.mean(Kind::Float);
    let mut tracker_r = Tensor::zeros_like(&tracker_v);

    for k in 0..steps {
        let act_k = s_acts.select(1, k);
        let pid_k = s_pids.select(1, k);
        
        let (h_next, r_logits) = model.dynamics.forward(&h, &act_k, &pid_k);
        h = scale_gradient(&h_next, 0.5);

        let target_h = tch::no_grad(|| ema_model.representation.forward(&s_t_states.select(1, k)));
        let target_proj = tch::no_grad(|| ema_model.projector.forward(&target_h));
        
        let proj_h = model.projector.forward(&h);
        let (v_l, p_l, h_l) = model.prediction.forward(&h);

        let r_targets = model.scalar_to_support(&s_rews.select(1, k));
        let mask = s_masks.select(1, k + 1);

        let rl = soft_cross_entropy(&r_logits, &r_targets) * &mask;
        
        let v_targets = model.scalar_to_support(&s_t_vals.select(1, k + 1));
        let vl = soft_cross_entropy(&v_l, &v_targets) * &mask;

        let p_probs = s_t_pols.select(1, k + 1) + 1e-8;
        let pl = soft_cross_entropy(&p_l, &p_probs) * &mask;

        tracker_r += rl.mean(Kind::Float);
        tracker_v += vl.mean(Kind::Float);
        tracker_p += pl.mean(Kind::Float);

        loss += &rl + &vl + &pl;
        loss += negative_cosine_similarity(&proj_h, &target_proj) * &mask;

        let mut bce_k = binary_cross_entropy(&h_l, &s_t_states.select(1, k).select(1, 19));
        if bce_k.dim() > 1 {
            bce_k = bce_k.flatten(1, -1).mean_dim(&[1], false, Kind::Float);
        }
        loss += bce_k * 0.5 * &mask;
    }

    let final_loss = (loss * scaled_weights).mean(Kind::Float) / (steps as f64);
    final_loss.backward();

    opt.clip_grad_norm(5.0);
    opt.step();

    // Priority updates
    tch::no_grad(|| {
        let td_errors = (model.scalar_to_support(&s_t_vals.select(1, 0)) - v_logits.softmax(-1, Kind::Float))
            .abs()
            .sum_dim_intlist(&[-1], false, Kind::Float);
        
        let td_vec: Vec<f32> = td_errors.try_into().unwrap_or_default();
        let td_f64: Vec<f64> = td_vec.into_iter().map(|x| x as f64).collect();
        buffer.update_priorities(&indices, &td_f64);
    });
}
