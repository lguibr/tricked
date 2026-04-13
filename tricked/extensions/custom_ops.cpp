#include <torch/extension.h>

extern "C" {
    void launch_extract_unrolled_features(const int64_t *boards, const int64_t *hist, float *out, int batch_size, int unroll_steps, int out_channels, int spatial_rows, int spatial_cols);
    void launch_extract_features(const int64_t *boards, const int32_t *avail,
                                 const int64_t *hist, const int32_t *acts,
                                 const int32_t *diff, float *out,
                                 const int32_t *canonical, const int64_t *compact,
                                 const int64_t *standard_pieces, int batch_size,
                                 int num_standard_pieces, int out_channels,
                                 int spatial_rows, int spatial_cols);
}

torch::Tensor extract_features(torch::Tensor boards, torch::Tensor avail, torch::Tensor hist, torch::Tensor acts, torch::Tensor diff, torch::Tensor canonical, torch::Tensor compact, torch::Tensor standard_pieces, int unroll_steps, int out_channels, int spatial_rows, int spatial_cols) {
    TORCH_CHECK(boards.is_contiguous(), "boards must be contiguous");
    TORCH_CHECK(hist.is_contiguous(), "hist must be contiguous");
    TORCH_CHECK(avail.is_contiguous(), "avail must be contiguous");
    TORCH_CHECK(acts.is_contiguous(), "acts must be contiguous");
    TORCH_CHECK(diff.is_contiguous(), "diff must be contiguous");
    
    int total_states = boards.numel() / 2;
    int batch_size = total_states / unroll_steps;

    int num_standard_pieces = standard_pieces.numel() / 2;
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(boards.device());
    torch::Tensor out_tensor;
    
    if (unroll_steps == 1) {
        out_tensor = torch::zeros({batch_size, out_channels, spatial_rows, spatial_cols}, options);
    } else {
        out_tensor = torch::zeros({batch_size, unroll_steps, out_channels, spatial_rows, spatial_cols}, options);
    }
    
    launch_extract_features(
        boards.data_ptr<int64_t>(),
        avail.data_ptr<int32_t>(),
        hist.data_ptr<int64_t>(),
        acts.data_ptr<int32_t>(),
        diff.data_ptr<int32_t>(),
        out_tensor.data_ptr<float>(),
        canonical.data_ptr<int32_t>(),
        compact.data_ptr<int64_t>(),
        standard_pieces.data_ptr<int64_t>(),
        total_states, num_standard_pieces, out_channels, spatial_rows, spatial_cols
    );
    
    return out_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("extract_features", &extract_features, "Extract Hexagonal Bitboard Features to 2D Spatial Tensors");
}
