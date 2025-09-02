#include <torch/extension.h>

torch::Tensor f_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, _module) {
    _module.def("flash_forward", torch::wrap_pybind_function(f_forward));
}
