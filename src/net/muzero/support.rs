use crate::net::muzero::ffi::load_tricked_ops;
use tch::Tensor;

pub fn support_to_scalar_fused(logits: &Tensor, support_size: i64, epsilon: f64) -> Tensor {
    let batch_size = logits.size()[0] as i32;
    let out = Tensor::zeros([batch_size as i64], (tch::Kind::Float, logits.device()));
    if !logits.device().is_cuda() {
        let probs = logits.softmax(-1, tch::Kind::Float);
        let support = Tensor::arange(2 * support_size + 1, (tch::Kind::Float, logits.device()))
            - support_size as f64;
        let expected_value =
            (probs * support).sum_dim_intlist(Some(&[-1][..]), false, tch::Kind::Float);
        let sgn = expected_value.sign();
        let abs_x = expected_value.abs();

        let eps = epsilon;
        let term1 = ((&abs_x + (1.0 + eps)) * (4.0 * eps) + 1.0).sqrt() - 1.0;
        let term2 = term1 / (2.0 * eps);
        let inv = sgn * (term2.pow_tensor_scalar(2.0) - 1.0);
        return inv;
    } else {
        unsafe {
            if let Some(lib) = load_tricked_ops() {
                if let Ok(func) = lib.get::<unsafe extern "C" fn(*const f32, *mut f32, i32, i32, f32)>(
                    b"launch_support_to_scalar\0",
                ) {
                    func(
                        logits.data_ptr() as *const f32,
                        out.data_ptr() as *mut f32,
                        batch_size,
                        support_size as i32,
                        epsilon as f32,
                    );
                } else {
                    eprintln!("WARNING: Could not find launch_support_to_scalar in tricked_ops.so");
                }
                std::mem::forget(lib);
            }
        }
    }
    out
}

pub fn scalar_to_support_fused(scalar: &Tensor, support_size: i64, epsilon: f64) -> Tensor {
    let batch_size = scalar.size()[0] as i32;
    let mut out = Tensor::zeros(
        [batch_size as i64, 2 * support_size + 1],
        (tch::Kind::Float, scalar.device()),
    );
    if !scalar.device().is_cuda() {
        let safe_scalar = scalar.nan_to_num(0.0, 0.0, 0.0);
        let transformed = safe_scalar.sign() * ((safe_scalar.abs() + 1.0).sqrt() - 1.0)
            + safe_scalar.copy() * epsilon;
        let clamped = transformed
            .reshape([-1])
            .clamp(-support_size as f64, support_size as f64);
        let shifted = clamped + support_size as f64;
        let floor_val = shifted.floor();
        let ceil_val = shifted.ceil();

        let upper_prob = shifted.copy() - floor_val.copy();
        let lower_prob = 1.0 - upper_prob.copy();

        let lower_idx = floor_val.to_kind(tch::Kind::Int64);
        let upper_idx = ceil_val.to_kind(tch::Kind::Int64);

        let batch_indices = Tensor::arange(batch_size as i64, (tch::Kind::Int64, scalar.device()));

        let _ = out.index_put_(
            &[Some(batch_indices.copy()), Some(lower_idx)],
            &lower_prob,
            true,
        );
        let _ = out.index_put_(
            &[Some(batch_indices.copy()), Some(upper_idx)],
            &upper_prob,
            true,
        );
        return out;
    } else {
        unsafe {
            if let Some(lib) = load_tricked_ops() {
                if let Ok(func) = lib.get::<unsafe extern "C" fn(*const f32, *mut f32, i32, i32, f32)>(
                    b"launch_scalar_to_support\0",
                ) {
                    func(
                        scalar.data_ptr() as *const f32,
                        out.data_ptr() as *mut f32,
                        batch_size,
                        support_size as i32,
                        epsilon as f32,
                    );
                } else {
                    eprintln!("WARNING: Could not find launch_scalar_to_support in tricked_ops.so");
                }
                std::mem::forget(lib);
            }
        }
    }
    out
}
