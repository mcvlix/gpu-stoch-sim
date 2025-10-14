import torch
import matplotlib.pyplot as plt
from simkit.integrators import sdeint_euler
from simkit.models import gbm_f, gbm_g

device = "cuda" if torch.cuda.is_available() else "cpu"
t = torch.linspace(0, 1, 400, device=device)
y0 = torch.ones(64, 1, device=device)
params = {"mu": torch.tensor(0.2, device=device), "sigma": torch.tensor(0.5, device=device)}
gen = torch.Generator(device=device).manual_seed(0)
_, traj = sdeint_euler(gbm_f, gbm_g, y0, t, params, generator=gen, return_path=True, use_cuda_kernel=False)
traj = traj[..., 0].to("cpu")  # [T, B]
plt.plot(traj[:, :20])
plt.title("GBM sample paths")
plt.show()
