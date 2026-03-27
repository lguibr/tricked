use tch::{nn, nn::Module, Tensor};

#[derive(Debug)]
pub struct GraphConv1d {
    linear_transformation: nn::Linear,
    adjacency_matrix: Tensor,
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

        let mut edges = Vec::new();
        for source_index in 0..spatial_grid_size as usize {
            edges.push((source_index, source_index));
            let neighbor_mask = crate::neighbors::NEIGHBOR_MASKS[source_index];
            for duplicate_target_index in 0..spatial_grid_size as usize {
                if (neighbor_mask >> duplicate_target_index) & 1 == 1 {
                    edges.push((source_index, duplicate_target_index));
                }
            }
        }

        let mut degrees = vec![0.0f32; spatial_grid_size as usize];
        for &(src, _) in &edges {
            degrees[src] += 1.0;
        }

        let mut adj = vec![0.0f32; (spatial_grid_size * spatial_grid_size) as usize];
        for &(src, tgt) in &edges {
            let normalized_val = 1.0 / (degrees[src].sqrt() * degrees[tgt].sqrt());
            adj[tgt * (spatial_grid_size as usize) + src] = normalized_val;
        }

        let adjacency_matrix = Tensor::from_slice(&adj)
            .view((spatial_grid_size, spatial_grid_size))
            .to_device(variable_store.device());

        Self {
            linear_transformation,
            adjacency_matrix,
        }
    }
}

impl Module for GraphConv1d {
    fn forward(&self, node_features: &Tensor) -> Tensor {
        let x_transposed = node_features.transpose(1, 2);
        let adj_aligned = self.adjacency_matrix.to_kind(node_features.kind());

        let out = adj_aligned.matmul(&x_transposed);
        let out_fp32 = out.to_kind(tch::Kind::Float);

        self.linear_transformation
            .forward(&out_fp32)
            .transpose(1, 2)
    }
}
