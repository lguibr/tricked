#include <c10/util/Exception.h>
#include <iostream>
#include <torch/script.h>

class TrickedWarningHandler : public c10::WarningHandler {
public:
  void process(const c10::Warning &warning) override {
    // Suppress ALL PyTorch warnings to keep the engine logs clean!
    return;
  }
};

static TrickedWarningHandler warning_handler;

struct InitWarningHandler {
  InitWarningHandler() {
    c10::WarningUtils::set_warning_handler(&warning_handler);
  }
};
static InitWarningHandler init_handler;

torch::Tensor extract_feature_cuda(torch::Tensor boards, torch::Tensor avail,
                                   torch::Tensor hist, torch::Tensor acts,
                                   torch::Tensor diff, torch::Tensor canonical,
                                   torch::Tensor compact,
                                   torch::Tensor standard);

torch::Tensor extract_unrolled_features_cuda(torch::Tensor boards, torch::Tensor hist);

TORCH_LIBRARY(tricked, m) { 
  m.def("extract_feature", extract_feature_cuda); 
  m.def("extract_unrolled_features", extract_unrolled_features_cuda); 
}
