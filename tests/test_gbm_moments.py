import math, torch
from sdegpu.integrators import sdeint_euler
from sdegpu.models import gbm_f, gbm_g

def test_gbm_moments_cpu():
    device = "cpu"
    B, d = 4096, 1
    t = torch.linspace(0, 1, 256, device=device)
    y0 = torch.ones(B, d, device=device)
    p = {"mu": torch.tensor(0.2, device=device), "sigma": torch.tensor(0.5, device=device)}
    gen = torch.Generator(device=device).manual_seed(42)
    xT,_ = sdeint_euler(gbm_f, gbm_g, y0, t, p, gen, return_path=False, use_cuda_kernel=False)
    mean = xT.mean().item()
    var  = xT.var(unbiased=False).item()
    mean_th = math.exp(0.2)
    var_th  = math.exp(0.4)*(math.exp(0.25)-1)
    assert abs(mean-mean_th)/mean_th < 0.03
    assert abs(var-var_th)/var_th   < 0.06
