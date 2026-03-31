use crate::core::features::get_valid_spatial_mask_8x8;
use tch::{nn, nn::Module, Tensor};

#[derive(Debug)]
pub struct FlattenedResNetBlock {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
    spatial_mask: Tensor,
}

impl FlattenedResNetBlock {
    pub fn new(variable_store: &nn::Path, hidden_dimension_size: i64, _grid_size: i64) -> Self {
        let config = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
        let conv1 = nn::conv2d(
            &(variable_store / "conv1"),
            hidden_dimension_size,
            hidden_dimension_size,
            3,
            config,
        );
        let conv2 = nn::conv2d(
            &(variable_store / "conv2"),
            hidden_dimension_size,
            hidden_dimension_size,
            3,
            config,
        );

        let norm1 = nn::layer_norm(
            &(variable_store / "norm1"),
            vec![hidden_dimension_size],
            Default::default(),
        );
        let norm2 = nn::layer_norm(
            &(variable_store / "norm2"),
            vec![hidden_dimension_size],
            Default::default(),
        );

        let spatial_mask = get_valid_spatial_mask_8x8(variable_store.device());

        Self {
            conv1,
            conv2,
            norm1,
            norm2,
            spatial_mask,
        }
    }
}

impl Module for FlattenedResNetBlock {
    fn forward(&self, input_tensor_batch_channel_height_width: &Tensor) -> Tensor {
        let input_shape = input_tensor_batch_channel_height_width.size();
        assert_eq!(
            input_shape.len(),
            4,
            "FlattenedResNetBlock requires [Batch, Channels, Height, Width] input"
        );
        let residual = input_tensor_batch_channel_height_width;
        let mut output_tensor = self.conv1.forward(input_tensor_batch_channel_height_width);
        output_tensor = &output_tensor * &self.spatial_mask;

        output_tensor = self
            .norm1
            .forward(&output_tensor.permute([0, 2, 3, 1]).contiguous())
            .permute([0, 3, 1, 2])
            .contiguous()
            .mish();
        output_tensor = &output_tensor * &self.spatial_mask;

        output_tensor = self.conv2.forward(&output_tensor);
        output_tensor = &output_tensor * &self.spatial_mask;

        output_tensor = self
            .norm2
            .forward(&output_tensor.permute([0, 2, 3, 1]).contiguous())
            .permute([0, 3, 1, 2])
            .contiguous();
        output_tensor = &output_tensor * &self.spatial_mask;

        ((residual + output_tensor).mish()) * &self.spatial_mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::features::HEXAGONAL_TO_CARTESIAN_MAP_ARRAY;
    use tch::{Device, Kind};

    #[test]
    fn test_topology_wormhole_masking() {
        let variable_store = nn::VarStore::new(Device::Cpu);
        let hidden_dimension_size = 16;
        let block = FlattenedResNetBlock::new(&variable_store.root(), hidden_dimension_size, 0);

        let input = Tensor::zeros([1, hidden_dimension_size, 8, 8], (Kind::Float, Device::Cpu));
        // Place a 1.0 at a valid hex position mapping (row, col/2)
        let (r, c) = HEXAGONAL_TO_CARTESIAN_MAP_ARRAY[0];
        let _ = input
            .narrow(2, r as i64, 1)
            .narrow(3, (c / 2) as i64, 1)
            .fill_(1.0);

        let mut output_tensor = input;
        for _ in 0..10 {
            output_tensor = block.forward(&output_tensor);
        }

        // Verify that dead cells are absolutely zero
        let out_slice: Vec<f32> = output_tensor.reshape([-1]).try_into().unwrap();

        let mut dead_cells_count = 0;
        for r in 0..8 {
            for c in 0..8 {
                let mut is_valid = false;
                for &(vr, vc) in HEXAGONAL_TO_CARTESIAN_MAP_ARRAY.iter() {
                    if vr == r && (vc / 2) == c {
                        is_valid = true;
                        break;
                    }
                }
                if !is_valid {
                    dead_cells_count += 1;
                    for m in 0..hidden_dimension_size as usize {
                        let idx = m * 64 + r * 8 + c; // N=1
                        assert_eq!(
                            out_slice[idx], 0.0,
                            "Dead cell {},{} leaked activation!",
                            r, c
                        );
                    }
                }
            }
        }
        assert!(
            dead_cells_count > 0,
            "There should be multiple dead cells in the 8x8 folding"
        );
    }
}
