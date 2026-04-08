#[cfg(test)]
mod tests {
    use crate::net::MuZeroNet;
    use tch::Tensor;
    use tch::{nn, Device, Kind};

    #[test]
    fn test_process_initial_inference_cpu_fallback_shape() {
        let vs = nn::VarStore::new(Device::Cpu);
        // Using config overrides where spatial_channel_count > native
        let net = MuZeroNet::new(&vs.root(), 64, 2, 300, 300, 64, 64);

        let batch_size = 2;
        let boards = Tensor::zeros([batch_size as i64], (Kind::Int64, Device::Cpu));
        let avail = Tensor::zeros([batch_size as i64, 3], (Kind::Int, Device::Cpu));
        let hist = Tensor::zeros([batch_size as i64, 10], (Kind::Int64, Device::Cpu));
        let acts = Tensor::zeros([batch_size as i64, 10], (Kind::Int, Device::Cpu));
        let diff = Tensor::zeros([batch_size as i64], (Kind::Int, Device::Cpu));

        // Simulating the behavior of CPU fallback initialization in inference_loop
        // that skips the CUDA FFI bindings
        let features = net.extract_initial_features(&boards, &avail, &hist, &acts, &diff);

        assert_eq!(
            features.size(),
            [batch_size as i64, 64, 8, 16],
            "CPU feature fallback MUST dynamically pad to spatial_channel_count, not natively 20"
        );
    }
}
