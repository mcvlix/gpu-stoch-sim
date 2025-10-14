import torch
from typing import Callable, Optional, Tuple

try:
    from ._cuda import em_step_cuda
    _HAS_CUDA_EXT = em_step_cuda is not None
except Exception:
    em_step_cuda = None
    _HAS_CUDA_EXT = False

def sdeint_euler(
    f: Callable[[torch.Tensor, torch.Tensor, dict], torch.Tensor],
    g: Callable[[torch.Tensor, torch.Tensor, dict], torch.Tensor],
    y0: torch.Tensor,
    t: torch.Tensor,
    params: Optional[dict] = None,
    generator: Optional[torch.Generator] = None,
    use_cuda_kernel: bool = True,
    return_path: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

    params = params or {}
    y = y0.clone()
    T = t.shape[0]
    traj = y.new_empty((T,) + y.shape) if return_path else None
    if return_path: traj[0] = y

    for k in range(T - 1):
        dt = (t[k+1] - t[k]).to(y.dtype)
        sqrt_dt = torch.sqrt(dt)
        eps = torch.randn(y.shape, dtype=y.dtype, device=y.device, generator=generator)
        ftk = f(t[k], y, params)
        gtk = g(t[k], y, params)

        if use_cuda_kernel and _HAS_CUDA_EXT and y.is_cuda:
            # broadcast dt, sqrt_dt to element-wise tensors
            y = em_step_cuda( 
                y, ftk, gtk,
                dt, sqrt_dt.expand_as(y), eps
            )
        else:
            y = y + ftk * dt + gtk * (sqrt_dt * eps)

        if return_path: traj[k+1] = y
    return (y, traj) if return_path else (y, None)
