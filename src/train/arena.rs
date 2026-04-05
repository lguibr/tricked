use tch::{Device, Kind, Tensor};

pub struct PinnedBatchTensors {
    pub state_features: Tensor,
    pub actions: Tensor,
    pub piece_identifiers: Tensor,
    pub value_prefixs: Tensor,
    pub target_policies: Tensor,
    pub target_values: Tensor,
    pub model_values: Tensor,
    pub transition_boards: Tensor,
    pub transition_actions: Tensor,
    pub transition_metadata: Tensor,
    pub loss_masks: Tensor,
    pub importance_weights: Tensor,
}

impl PinnedBatchTensors {
    pub fn new(batch_size: usize, unroll: usize, device: Device) -> Self {
        let pin = |size: &[i64], kind: Kind| {
            let t = Tensor::zeros(size, (kind, Device::Cpu));
            if device.is_cuda() {
                t.pin_memory(device)
            } else {
                t
            }
        };
        Self {
            state_features: pin(&[batch_size as i64, 20, 8, 16], Kind::Float),
            actions: pin(&[batch_size as i64, unroll as i64], Kind::Int64),
            piece_identifiers: pin(&[batch_size as i64, unroll as i64], Kind::Int64),
            value_prefixs: pin(&[batch_size as i64, unroll as i64], Kind::Float),
            target_policies: pin(&[batch_size as i64, (unroll + 1) as i64, 288], Kind::Float),
            target_values: pin(&[batch_size as i64, (unroll + 1) as i64], Kind::Float),
            model_values: pin(&[batch_size as i64, (unroll + 1) as i64], Kind::Float),
            transition_boards: pin(&[batch_size as i64, unroll as i64, 8, 2], Kind::Int64),
            transition_actions: pin(&[batch_size as i64, unroll as i64, 3], Kind::Int),
            transition_metadata: pin(&[batch_size as i64, unroll as i64, 4], Kind::Int),
            loss_masks: pin(&[batch_size as i64, (unroll + 1) as i64], Kind::Float),
            importance_weights: pin(&[batch_size as i64], Kind::Float),
        }
    }

    pub fn copy_from_unpinned(&mut self, batch: &crate::train::buffer::BatchTensors) {
        self.state_features.copy_(&batch.state_features_batch);
        self.actions.copy_(&batch.actions_batch);
        self.piece_identifiers.copy_(&batch.piece_identifiers_batch);
        self.value_prefixs.copy_(&batch.value_prefixs_batch);
        self.target_policies.copy_(&batch.target_policies_batch);
        self.target_values.copy_(&batch.target_values_batch);
        self.model_values.copy_(&batch.model_values_batch);
        self.transition_boards.copy_(&batch.transition_boards_batch);
        self.transition_actions
            .copy_(&batch.transition_actions_batch);
        self.transition_metadata
            .copy_(&batch.transition_metadata_batch);
        self.loss_masks.copy_(&batch.loss_masks_batch);
        self.importance_weights
            .copy_(&batch.importance_weights_batch);
    }
}

pub struct GpuBatchTensors {
    pub state_features: Tensor,
    pub actions: Tensor,
    pub piece_identifiers: Tensor,
    pub value_prefixs: Tensor,
    pub target_policies: Tensor,
    pub target_values: Tensor,
    pub model_values: Tensor,
    pub transition_boards: Tensor,
    pub transition_actions: Tensor,
    pub transition_metadata: Tensor,
    pub loss_masks: Tensor,
    pub importance_weights: Tensor,
}

impl GpuBatchTensors {
    pub fn new(batch_size: usize, unroll: usize, device: Device) -> Self {
        Self {
            state_features: Tensor::zeros([batch_size as i64, 20, 8, 16], (Kind::Float, device)),
            actions: Tensor::zeros([batch_size as i64, unroll as i64], (Kind::Int64, device)),
            piece_identifiers: Tensor::zeros(
                [batch_size as i64, unroll as i64],
                (Kind::Int64, device),
            ),
            value_prefixs: Tensor::zeros([batch_size as i64, unroll as i64], (Kind::Float, device)),
            target_policies: Tensor::zeros(
                [batch_size as i64, (unroll + 1) as i64, 288],
                (Kind::Float, device),
            ),
            target_values: Tensor::zeros(
                [batch_size as i64, (unroll + 1) as i64],
                (Kind::Float, device),
            ),
            model_values: Tensor::zeros(
                [batch_size as i64, (unroll + 1) as i64],
                (Kind::Float, device),
            ),
            transition_boards: Tensor::zeros(
                [batch_size as i64, unroll as i64, 8, 2],
                (Kind::Int64, device),
            ),
            transition_actions: Tensor::zeros(
                [batch_size as i64, unroll as i64, 3],
                (Kind::Int, device),
            ),
            transition_metadata: Tensor::zeros(
                [batch_size as i64, unroll as i64, 4],
                (Kind::Int, device),
            ),
            loss_masks: Tensor::zeros(
                [batch_size as i64, (unroll + 1) as i64],
                (Kind::Float, device),
            ),
            importance_weights: Tensor::zeros([batch_size as i64], (Kind::Float, device)),
        }
    }

    pub fn copy_from_pinned(&mut self, pinned: &PinnedBatchTensors) {
        self.state_features.copy_(&pinned.state_features);
        self.actions.copy_(&pinned.actions);
        self.piece_identifiers.copy_(&pinned.piece_identifiers);
        self.value_prefixs.copy_(&pinned.value_prefixs);
        self.target_policies.copy_(&pinned.target_policies);
        self.target_values.copy_(&pinned.target_values);
        self.model_values.copy_(&pinned.model_values);
        self.transition_boards.copy_(&pinned.transition_boards);
        self.transition_actions.copy_(&pinned.transition_actions);
        self.transition_metadata.copy_(&pinned.transition_metadata);
        self.loss_masks.copy_(&pinned.loss_masks);
        self.importance_weights.copy_(&pinned.importance_weights);

        let _ = self.value_prefixs.f_nan_to_num_(0.0, Some(0.0), Some(0.0));
        let _ = self
            .target_policies
            .f_nan_to_num_(0.0, Some(0.0), Some(0.0));
        let _ = self.target_values.f_nan_to_num_(0.0, Some(0.0), Some(0.0));
        let _ = self.model_values.f_nan_to_num_(0.0, Some(0.0), Some(0.0));

        let flip_seed = rand::random::<u8>() % 4;
        match flip_seed {
            1 => {
                self.state_features = self.state_features.flip([2i64]);
            }
            2 => {
                self.state_features = self.state_features.flip([3i64]);
            }
            3 => {
                self.state_features = self.state_features.flip([2i64, 3i64]);
            }
            _ => {}
        }

        let rot_seed = rand::random::<u8>() % 2;
        if rot_seed == 1 {
            self.state_features = self.state_features.rot90(2, [2, 3]);
        }
    }
}
