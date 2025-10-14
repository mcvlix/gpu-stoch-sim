#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void em_kernel(const T* __restrict__ ftk,
                          const T* __restrict__ gtk,
                          const T* __restrict__ sqrt_dt,
                          const T* __restrict__ eps,
                          const T   dt,
                          T* __restrict__ y,
                          const int64_t N)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // y[i] += ftk[i] * dt + gtk[i] * (sqrt_dt[i] * eps[i]);
        T inc = ftk[i] * dt + gtk[i] * (sqrt_dt[i] * eps[i]);
        y[i] += inc;
    }
}

torch::Tensor em_step_cuda(torch::Tensor y,
                           torch::Tensor ftk,
                           torch::Tensor gtk,
                           torch::Tensor dt,        // scalar or broadcasted
                           torch::Tensor sqrt_dt,   // broadcasted to y shape
                           torch::Tensor eps)
{
    TORCH_CHECK(y.is_cuda(), "y must be CUDA");
    auto N = y.numel();
    auto dtype = y.scalar_type();

    // ensure contiguous
    y = y.contiguous();
    ftk = ftk.contiguous();
    gtk = gtk.contiguous();
    sqrt_dt = sqrt_dt.contiguous();
    eps = eps.contiguous();

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    if (dtype == torch::kFloat32) {
        float dt_scalar = dt.item<float>();
        em_kernel<<<blocks, threads>>>(
            ftk.data_ptr<float>(),
            gtk.data_ptr<float>(),
            sqrt_dt.data_ptr<float>(),
            eps.data_ptr<float>(),
            dt_scalar,
            y.data_ptr<float>(),
            N
        );
    } else if (dtype == torch::kFloat64) {
        double dt_scalar = dt.item<double>();
        em_kernel<<<blocks, threads>>>(
            ftk.data_ptr<double>(),
            gtk.data_ptr<double>(),
            sqrt_dt.data_ptr<double>(),
            eps.data_ptr<double>(),
            dt_scalar,
            y.data_ptr<double>(),
            N
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype");
    }

    return y;
}
