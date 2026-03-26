use tch::{nn, nn::Module, Kind, Tensor};

#[derive(Debug)]
pub struct GraphConv1d {
    linear_transformation: nn::Linear,
    adjacency_normalized: Tensor,
    spatial_grid_size: i64,
    input_channel_count: i64,
}

impl GraphConv1d {
    pub fn new(
        variable_store: &nn::Path,
        input_channel_count: i64,
        output_channel_count: i64,
        spatial_grid_size: i64,
    ) -> Self {
        let linear_transformation = nn::linear(
            variable_store,
            input_channel_count,
            output_channel_count,
            Default::default(),
        );

        let mut adjacency_raw_data = vec![0.0f32; (spatial_grid_size * spatial_grid_size) as usize];
        for source_index in 0..spatial_grid_size as usize {
            adjacency_raw_data[source_index * spatial_grid_size as usize + source_index] = 1.0;
            let neighbor_mask = crate::neighbors::NEIGHBOR_MASKS[source_index];
            for duplicate_target_index in 0..spatial_grid_size as usize {
                if (neighbor_mask >> duplicate_target_index) & 1 == 1 {
                    adjacency_raw_data
                        [source_index * spatial_grid_size as usize + duplicate_target_index] = 1.0;
                }
            }
        }

        let adjacency_dense_tensor = Tensor::from_slice(&adjacency_raw_data)
            .reshape([spatial_grid_size, spatial_grid_size])
            .to(variable_store.device());

        let degree_matrix =
            adjacency_dense_tensor.sum_dim_intlist(&[-1i64][..], false, Kind::Float);
        let degree_inverse_sqrt = degree_matrix.pow_tensor_scalar(-0.5);
        let valid_degree_inverse_sqrt =
            degree_inverse_sqrt.where_scalarother(&degree_inverse_sqrt.isfinite(), 0.0);

        let degree_diagonal_matrix = valid_degree_inverse_sqrt.diag(0);
        let adjacency_normalized = degree_diagonal_matrix
            .matmul(&adjacency_dense_tensor)
            .matmul(&degree_diagonal_matrix);

        Self {
            linear_transformation,
            adjacency_normalized,
            spatial_grid_size,
            input_channel_count,
        }
    }
}

impl Module for GraphConv1d {
    fn forward(&self, node_features: &Tensor) -> Tensor {
        let message_passing = self
            .adjacency_normalized
            .matmul(&node_features.transpose(1, 2));

        self.linear_transformation
            .forward(&message_passing)
            .transpose(1, 2)
    }
}
