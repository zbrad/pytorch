// Test kernel for win_cpp_extension_cccl.py. It includes CCCL (cub), which on CUDA 13 needs /Zc:preprocessor under MSVC.
#include <torch/extension.h>
#include <cub/cub.cuh>

__global__ void add_one(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += 1.0f;
}

torch::Tensor add_one_cuda(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32);
  auto y = x.clone();
  int n = static_cast<int>(y.numel());
  add_one<<<(n + 255) / 256, 256>>>(y.data_ptr<float>(), n);
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("add_one", &add_one_cuda); }
