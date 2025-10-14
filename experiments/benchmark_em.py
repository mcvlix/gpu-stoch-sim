import time, math, torch
from simkit.integrators import sdeint_euler
from simkit.models import gbm_f, gbm_g

def run(use_cuda_kernel):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, d, N = 131072, 1, 256  # 131k paths
    t = torch.linspace(0, 1, N, device=device)
    y0 = torch.ones(B, d, device=device, dtype=torch.float32)
    params = {"mu": torch.tensor(0.2, device=device), "sigma": torch.tensor(0.5, device=device)}
    gen = torch.Generator(device=device).manual_seed(1234)

    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()
    xT, _ = sdeint_euler(gbm_f, gbm_g, y0, t, params, generator=gen, use_cuda_kernel=use_cuda_kernel, return_path=False)
    torch.cuda.synchronize() if device == "cuda" else None
    dt = time.time() - t0
    return dt, float(xT.mean().item())

if __name__ == "__main__":
    for flag in [False, True]:
        dt, mean = run(flag)
        print(f"use_cuda_kernel={flag} | elapsed={dt:.3f}s | mean={mean:.4f}")
