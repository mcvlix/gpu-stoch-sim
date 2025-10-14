#include <torch/extension.h>
torch::Tensor em_step_cuda(torch::Tensor y,
                           torch::Tensor ftk,
                           torch::Tensor gtk,
                           torch::Tensor dt,
                           torch::Tensor sqrt_dt,
                           torch::Tensor eps);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("em_step_cuda", &em_step_cuda, "Euler–Maruyama fused step (CUDA)");
}
