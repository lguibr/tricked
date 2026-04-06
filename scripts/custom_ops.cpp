#include <torch/script.h>
#include <c10/util/Exception.h>
#include <iostream>

class TrickedWarningHandler : public c10::WarningHandler {
public:
    void process(const c10::Warning& warning) override {
        std::string msg = warning.msg();
        if (msg.find("is_pinned") != std::string::npos ||
            msg.find("pin_memory") != std::string::npos ||
            msg.find("autocast") != std::string::npos) {
            return; // Suppress explicitly
        }
        c10::WarningHandler::process(warning);
    }
};

static TrickedWarningHandler warning_handler;

struct InitWarningHandler {
    InitWarningHandler() {
        c10::WarningUtils::set_warning_handler(&warning_handler);
    }
};
static InitWarningHandler init_handler;

torch::Tensor extract_feature_cuda(
    torch::Tensor boards, torch::Tensor avail, torch::Tensor hist,
    torch::Tensor acts, torch::Tensor diff, torch::Tensor canonical,
    torch::Tensor compact, torch::Tensor standard);

TORCH_LIBRARY(tricked, m) {
    m.def("extract_feature", extract_feature_cuda);
}
