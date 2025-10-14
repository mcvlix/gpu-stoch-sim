import torch

class Brownian:
    def __init__(self, shape, device=None, dtype=None, seed: int = 0):
        self.shape = shape
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.device = device
        self.dtype = dtype

    def eps(self):
        return torch.randn(self.shape, device=self.device, dtype=self.dtype, generator=self.gen)

    @staticmethod
    def dW_from_eps(eps, dt):
        return eps * torch.sqrt(torch.as_tensor(dt, device=eps.device, dtype=eps.dtype))
