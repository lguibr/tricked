use tch::{nn, nn::Module, Kind, Tensor};

#[derive(Debug)]
pub struct GraphConv1d {
    linear: nn::Linear,
    a_dense: Tensor,
    grid_size: i64,
    in_channels: i64,
}

impl GraphConv1d {
    pub fn new(vs: &nn::Path, in_channels: i64, out_channels: i64, grid_size: i64) -> Self {
        let linear = nn::linear(vs, in_channels, out_channels, Default::default());

        let mut a_data = vec![0.0f32; (grid_size * grid_size) as usize];
        for i in 0..grid_size as usize {
            a_data[i * grid_size as usize + i] = 1.0;
            let mask = crate::neighbors::NEIGHBOR_MASKS[i];
            for j in 0..grid_size as usize {
                if (mask >> j) & 1 == 1 {
                    a_data[i * grid_size as usize + j] = 1.0;
                }
            }
        }

        let a = Tensor::from_slice(&a_data)
            .view((grid_size, grid_size))
            .to(vs.device());
        let d = a.sum_dim_intlist(&[-1], false, Kind::Float);
        let d_inv_sqrt = d.pow_tensor_scalar(-0.5);
        let d_inv_sqrt = d_inv_sqrt.where_scalarother(&d_inv_sqrt.isfinite(), 0.0);
        let d_mat = d_inv_sqrt.diag(0);
        let a_norm = d_mat.matmul(&a).matmul(&d_mat);

        Self {
            linear,
            a_dense: a_norm,
            grid_size,
            in_channels,
        }
    }
}

impl Module for GraphConv1d {
    fn forward(&self, x: &Tensor) -> Tensor {
        let b = x.size()[0];
        let x_reshaped = x.transpose(1, 2).reshape(&[self.grid_size, -1]);
        let msg_fp32 = self.a_dense.matmul(&x_reshaped);
        let msg = msg_fp32
            .view((self.grid_size, b, self.in_channels))
            .permute(&[1, 2, 0]);
        self.linear.forward(&msg.transpose(1, 2)).transpose(1, 2)
    }
}
