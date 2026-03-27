use tch::{nn, nn::Module, Tensor};

#[derive(Debug)]
pub struct GraphConv1d {
    linear_transformation: nn::Linear,
    src_indices: Tensor,
    dst_indices: Tensor,
    edge_weights: Tensor,
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

        let mut src_indices_vec = Vec::with_capacity(edges.len());
        let mut dst_indices_vec = Vec::with_capacity(edges.len());
        let mut edge_weights_vec = Vec::with_capacity(edges.len());

        for &(src, tgt) in &edges {
            src_indices_vec.push(src as i64);
            dst_indices_vec.push(tgt as i64);
            let normalized_val = 1.0 / (degrees[src].sqrt() * degrees[tgt].sqrt());
            edge_weights_vec.push(normalized_val);
        }

        let src_indices = Tensor::from_slice(&src_indices_vec).to_device(variable_store.device());
        let dst_indices = Tensor::from_slice(&dst_indices_vec).to_device(variable_store.device());
        let edge_weights = Tensor::from_slice(&edge_weights_vec)
            .view((-1, 1))
            .to_device(variable_store.device());

        Self {
            linear_transformation,
            src_indices,
            dst_indices,
            edge_weights,
        }
    }
}

impl Module for GraphConv1d {
    fn forward(&self, node_features: &Tensor) -> Tensor {
        let x_transposed = node_features.transpose(1, 2);
        let messages = x_transposed.index_select(1, &self.src_indices);
        let edge_weights_aligned = self.edge_weights.to_kind(node_features.kind());
        let weighted_messages = messages * edge_weights_aligned;

        let out =
            Tensor::zeros_like(&x_transposed).index_add(1, &self.dst_indices, &weighted_messages);

        let out_fp32 = out.to_kind(tch::Kind::Float);

        self.linear_transformation
            .forward(&out_fp32)
            .transpose(1, 2)
    }
}
