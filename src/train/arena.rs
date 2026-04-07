use tch::{Device, Kind, Tensor};

pub struct PinnedBatchTensors {
    pub state_features: Tensor,
    pub actions: Tensor,
    pub piece_identifiers: Tensor,
    pub value_prefixs: Tensor,
    pub target_policies: Tensor,
    pub target_values: Tensor,
    pub model_values: Tensor,
    pub raw_unrolled_boards: Tensor,
    pub raw_unrolled_histories: Tensor,
    pub loss_masks: Tensor,
    pub importance_weights: Tensor,
}

impl PinnedBatchTensors {
    pub fn new(batch_size: usize, unroll: usize, _device: Device) -> Self {
        let pin = |size: &[i64], kind: Kind| Tensor::zeros(size, (kind, Device::Cpu));
        Self {
            state_features: pin(&[batch_size as i64, 20, 8, 16], Kind::Float),
            actions: pin(&[batch_size as i64, unroll as i64], Kind::Int64),
            piece_identifiers: pin(&[batch_size as i64, unroll as i64], Kind::Int64),
            value_prefixs: pin(&[batch_size as i64, unroll as i64], Kind::Float),
            target_policies: pin(&[batch_size as i64, (unroll + 1) as i64, 288], Kind::Float),
            target_values: pin(&[batch_size as i64, (unroll + 1) as i64], Kind::Float),
            model_values: pin(&[batch_size as i64, (unroll + 1) as i64], Kind::Float),
            raw_unrolled_boards: pin(&[batch_size as i64, unroll as i64, 2], Kind::Int64),
            raw_unrolled_histories: pin(&[batch_size as i64, unroll as i64, 14], Kind::Int64),
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
        self.raw_unrolled_boards
            .copy_(&batch.raw_unrolled_boards_batch);
        self.raw_unrolled_histories
            .copy_(&batch.raw_unrolled_histories_batch);
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
    pub raw_unrolled_boards: Tensor,
    pub raw_unrolled_histories: Tensor,
    pub loss_masks: Tensor,
    pub importance_weights: Tensor,
}

impl GpuBatchTensors {
    pub fn new(batch_size: usize, unroll: usize, device: Device) -> Self {
        let bf16_kind = if device.is_cuda() {
            Kind::BFloat16
        } else {
            Kind::Float
        };
        Self {
            state_features: Tensor::zeros([batch_size as i64, 20, 8, 16], (bf16_kind, device)),
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
            raw_unrolled_boards: Tensor::zeros(
                [batch_size as i64, unroll as i64, 2],
                (Kind::Int64, device),
            ),
            raw_unrolled_histories: Tensor::zeros(
                [batch_size as i64, unroll as i64, 14],
                (Kind::Int64, device),
            ),
            loss_masks: Tensor::zeros(
                [batch_size as i64, (unroll + 1) as i64],
                (Kind::Float, device),
            ),
            importance_weights: Tensor::zeros([batch_size as i64], (Kind::Float, device)),
        }
    }

    pub fn copy_from_pinned(&mut self, pinned: &PinnedBatchTensors) {
        let bf16_kind = if self.state_features.device().is_cuda() {
            Kind::BFloat16
        } else {
            Kind::Float
        };
        self.state_features =
            pinned
                .state_features
                .to_device_(self.state_features.device(), bf16_kind, false, false);
        self.actions = pinned
            .actions
            .to_device_(self.actions.device(), Kind::Int64, false, false);
        self.piece_identifiers = pinned.piece_identifiers.to_device_(
            self.piece_identifiers.device(),
            Kind::Int64,
            false,
            false,
        );
        self.value_prefixs =
            pinned
                .value_prefixs
                .to_device_(self.value_prefixs.device(), Kind::Float, false, false);
        self.target_policies = pinned.target_policies.to_device_(
            self.target_policies.device(),
            Kind::Float,
            false,
            false,
        );
        self.target_values =
            pinned
                .target_values
                .to_device_(self.target_values.device(), Kind::Float, false, false);
        self.model_values =
            pinned
                .model_values
                .to_device_(self.model_values.device(), Kind::Float, false, false);
        self.raw_unrolled_boards = pinned.raw_unrolled_boards.to_device_(
            self.raw_unrolled_boards.device(),
            Kind::Int64,
            false,
            false,
        );
        self.raw_unrolled_histories = pinned.raw_unrolled_histories.to_device_(
            self.raw_unrolled_histories.device(),
            Kind::Int64,
            false,
            false,
        );
        self.loss_masks =
            pinned
                .loss_masks
                .to_device_(self.loss_masks.device(), Kind::Float, false, false);
        self.importance_weights = pinned.importance_weights.to_device_(
            self.importance_weights.device(),
            Kind::Float,
            false,
            false,
        );

        let _ = self.value_prefixs.f_nan_to_num_(0.0, Some(0.0), Some(0.0));
        let _ = self
            .target_policies
            .f_nan_to_num_(0.0, Some(0.0), Some(0.0));
        let _ = self.target_values.f_nan_to_num_(0.0, Some(0.0), Some(0.0));
        let _ = self.model_values.f_nan_to_num_(0.0, Some(0.0), Some(0.0));
    }
}
