import torch

def gbm_f(t, x, p): return p["mu"] * x
def gbm_g(t, x, p): return p["sigma"] * x

def ou_f(t, x, p):  return p["theta"] * (p["mu"] - x)
def ou_g(t, x, p):  return p["sigma"] * torch.ones_like(x)
