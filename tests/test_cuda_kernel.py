import pytest, torch
from sdegpu.integrators import sdeint_euler
from sdegpu.models import gbm_f, gbm_g

cuda = torch.cuda.is_available()

@pytest.mark.skipif(not cuda, reason="CUDA not available")
def test_kernel_runs():
    device = "cuda"
    B, d = 8192, 1
    t = torch.linspace(0, 1, 128, device=device, dtype=torch.float32)
    y0 = torch.ones(B, d, device=device, dtype=torch.float32)
    p = {"mu": torch.tensor(0.1, device=device, dtype=torch.float32),
         "sigma": torch.tensor(0.3, device=device, dtype=torch.float32)}

    gen = torch.Generator(device=device).manual_seed(7)
    xT0,_ = sdeint_euler(gbm_f, gbm_g, y0, t, p, gen, return_path=False, use_cuda_kernel=False)
    gen.manual_seed(7)
    xT1,_ = sdeint_euler(gbm_f, gbm_g, y0, t, p, gen, return_path=False, use_cuda_kernel=True)

    assert xT0.dtype == xT1.dtype
    max_abs = (xT0 - xT1).abs().max().item()
    max_rel = ((xT0 - xT1).abs() / (xT0.abs().clamp_min(1e-12))).max().item()
    print("max_abs=", max_abs, " max_rel=", max_rel)

    assert torch.allclose(xT0, xT1, rtol=5e-5, atol=5e-6)
