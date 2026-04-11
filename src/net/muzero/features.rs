use crate::net::muzero::ffi::load_tricked_ops;
use tch::Tensor;

#[allow(clippy::too_many_arguments)]
pub fn extract_initial_features(
    spatial_channel_count: i64,
    canonical_tensor: &Tensor,
    compact_tensor: &Tensor,
    standard_tensor: &Tensor,
    num_standard_pieces: i32,
    boards: &Tensor,
    avail: &Tensor,
    hist: &Tensor,
    acts: &Tensor,
    diff: &Tensor,
) -> Tensor {
    let batch_size = boards.size()[0] as i32;
    let mut out = Tensor::zeros(
        [
            batch_size as i64,
            crate::core::features::NATIVE_FEATURE_CHANNELS as i64,
            8,
            16,
        ],
        (tch::Kind::Float, boards.device()),
    );

    if !boards.device().is_cuda() {
        let mut out_data = vec![
            0.0f32;
            (batch_size * crate::core::features::NATIVE_FEATURE_CHANNELS as i32 * 128)
                as usize
        ];
        let boards_data = Vec::<i64>::try_from(boards).unwrap_or_default();
        let avail_data = Vec::<i32>::try_from(avail).unwrap_or_default();
        let hist_data = Vec::<i64>::try_from(hist).unwrap_or_default();
        let acts_data = Vec::<i32>::try_from(acts).unwrap_or_default();
        let diff_data = Vec::<i32>::try_from(diff).unwrap_or_default();

        for b in 0..batch_size as usize {
            let board_low = boards_data.get(b * 2).copied().unwrap_or(0);
            let board_high = boards_data.get(b * 2 + 1).copied().unwrap_or(0);
            let board = (board_low as u64 as u128) | ((board_high as u64 as u128) << 64);

            let avail_arr = [
                avail_data.get(b * 3).copied().unwrap_or(-1),
                avail_data.get(b * 3 + 1).copied().unwrap_or(-1),
                avail_data.get(b * 3 + 2).copied().unwrap_or(-1),
            ];

            let mut history = Vec::with_capacity(7);
            for i in (0..7).rev() {
                let hl = hist_data.get(b * 14 + i * 2).copied().unwrap_or(0);
                let hh = hist_data.get(b * 14 + i * 2 + 1).copied().unwrap_or(0);
                history.push((hl as u64 as u128) | ((hh as u64 as u128) << 64));
            }

            let mut actions = Vec::with_capacity(3);
            for i in (0..3).rev() {
                actions.push(acts_data.get(b * 3 + i).copied().unwrap_or(-1));
            }

            let diff_val = diff_data.get(b).copied().unwrap_or(0);

            let start_idx = b * crate::core::features::NATIVE_FEATURE_CHANNELS * 128;
            let end_idx = start_idx + crate::core::features::NATIVE_FEATURE_CHANNELS * 128;
            crate::core::features::extract_feature_native(
                &mut out_data[start_idx..end_idx],
                board,
                &avail_arr,
                &history,
                &actions,
                diff_val,
            );
        }
        out = Tensor::from_slice(&out_data)
            .view([
                batch_size as i64,
                crate::core::features::NATIVE_FEATURE_CHANNELS as i64,
                8,
                16,
            ])
            .to_device(boards.device());
    } else {
        unsafe {
            if let Some(lib) = load_tricked_ops() {
                if let Ok(func) = lib.get::<unsafe extern "C" fn(
                    *const i64,
                    *const i32,
                    *const i64,
                    *const i32,
                    *const i32,
                    *mut f32,
                    *const i32,
                    *const i64,
                    *const i64,
                    i32,
                    i32,
                )>(b"launch_extract_features\0")
                {
                    func(
                        boards.data_ptr() as *const i64,
                        avail.data_ptr() as *const i32,
                        hist.data_ptr() as *const i64,
                        acts.data_ptr() as *const i32,
                        diff.data_ptr() as *const i32,
                        out.data_ptr() as *mut f32,
                        canonical_tensor.data_ptr() as *const i32,
                        compact_tensor.data_ptr() as *const i64,
                        standard_tensor.data_ptr() as *const i64,
                        batch_size,
                        num_standard_pieces,
                    );
                } else {
                    eprintln!("WARNING: Could not find launch_extract_features in tricked_ops.so");
                }
                std::mem::forget(lib);
            } else {
                eprintln!("WARNING: Could not load tricked_ops.so for extract_initial_features");
            }
        }
    }

    if spatial_channel_count > crate::core::features::NATIVE_FEATURE_CHANNELS as i64 {
        let padding = Tensor::zeros(
            [
                batch_size as i64,
                spatial_channel_count - crate::core::features::NATIVE_FEATURE_CHANNELS as i64,
                8,
                16,
            ],
            (tch::Kind::Float, boards.device()),
        );
        out = Tensor::cat(&[&out, &padding], 1);
    }
    out
}

pub fn extract_unrolled_features(
    spatial_channel_count: i64,
    boards: &Tensor,
    hist: &Tensor,
) -> Tensor {
    let batch_size = boards.size()[0] as i32;
    let unroll_steps = boards.size()[1] as i32;
    let mut out = Tensor::zeros(
        [
            batch_size as i64,
            unroll_steps as i64,
            crate::core::features::NATIVE_FEATURE_CHANNELS as i64,
            8,
            16,
        ],
        (tch::Kind::Float, boards.device()),
    );

    if !boards.device().is_cuda() {
        let mut out_data = vec![
            0.0f32;
            (batch_size
                * unroll_steps
                * crate::core::features::NATIVE_FEATURE_CHANNELS as i32
                * 128) as usize
        ];
        let boards_data = Vec::<i64>::try_from(boards).unwrap_or_default();
        let hist_data = Vec::<i64>::try_from(hist).unwrap_or_default();

        for b in 0..batch_size as usize {
            for u in 0..unroll_steps as usize {
                let board_low = boards_data
                    .get((b * (unroll_steps as usize) + u) * 2)
                    .copied()
                    .unwrap_or(0);
                let board_high = boards_data
                    .get((b * (unroll_steps as usize) + u) * 2 + 1)
                    .copied()
                    .unwrap_or(0);
                let board = (board_low as u64 as u128) | ((board_high as u64 as u128) << 64);

                let mut history = Vec::with_capacity(7);
                for i in (0..7).rev() {
                    let hl = hist_data
                        .get((b * (unroll_steps as usize) + u) * 14 + i * 2)
                        .copied()
                        .unwrap_or(0);
                    let hh = hist_data
                        .get((b * (unroll_steps as usize) + u) * 14 + i * 2 + 1)
                        .copied()
                        .unwrap_or(0);
                    history.push((hl as u64 as u128) | ((hh as u64 as u128) << 64));
                }

                let start_idx = (b * (unroll_steps as usize) + u)
                    * crate::core::features::NATIVE_FEATURE_CHANNELS
                    * 128;
                let out_slice = &mut out_data[start_idx
                    ..start_idx + crate::core::features::NATIVE_FEATURE_CHANNELS * 128];

                let fill_channel = |out: &mut [f32], c_idx: usize, mut bits: u128| {
                    let offset = c_idx * 128;
                    bits &= (1_u128 << 96) - 1;
                    while bits != 0 {
                        let bit_index = bits.trailing_zeros() as usize;
                        out[offset + crate::core::features::get_spatial_idx(bit_index)] = 1.0;
                        bits &= bits - 1;
                    }
                };

                fill_channel(out_slice, 0, board);
                for (i, &h) in history.iter().rev().enumerate() {
                    fill_channel(out_slice, i + 1, h);
                }
            }
        }
        out = Tensor::from_slice(&out_data)
            .view([
                batch_size as i64,
                unroll_steps as i64,
                crate::core::features::NATIVE_FEATURE_CHANNELS as i64,
                8,
                16,
            ])
            .to_device(boards.device());
    } else {
        unsafe {
            if let Some(lib) = load_tricked_ops() {
                if let Ok(func) =
                    lib.get::<unsafe extern "C" fn(*const i64, *const i64, *mut f32, i32, i32)>(
                        b"launch_extract_unrolled_features\0",
                    )
                {
                    func(
                        boards.data_ptr() as *const i64,
                        hist.data_ptr() as *const i64,
                        out.data_ptr() as *mut f32,
                        batch_size,
                        unroll_steps,
                    );
                } else {
                    eprintln!("WARNING: Could not find function symbol in tricked_ops.so");
                }
                std::mem::forget(lib);
            } else {
                eprintln!("WARNING: Could not load tricked_ops.so, extract_unrolled_features will return zeros");
            }
        }
    }

    if spatial_channel_count > crate::core::features::NATIVE_FEATURE_CHANNELS as i64 {
        let padding = Tensor::zeros(
            [
                batch_size as i64,
                unroll_steps as i64,
                spatial_channel_count - crate::core::features::NATIVE_FEATURE_CHANNELS as i64,
                8,
                16,
            ],
            (tch::Kind::Float, boards.device()),
        );
        out = Tensor::cat(&[&out, &padding], 2);
    }
    out
}
