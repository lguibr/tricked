#include <torch/script.h>

torch::Tensor extract_feature_cuda(
    torch::Tensor boards, torch::Tensor avail, torch::Tensor hist,
    torch::Tensor acts, torch::Tensor diff, torch::Tensor canonical,
    torch::Tensor compact, torch::Tensor standard);

TORCH_LIBRARY(tricked, m) {
    m.def("extract_feature", extract_feature_cuda);
}
