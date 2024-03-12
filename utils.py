import torch
import numpy as np

def to_numpy(var):
    if isinstance(var, torch.Tensor):
        # Ensure tensor is on CPU and convert to NumPy
        return var.cpu().detach().numpy()
    else:
        # Assume it's already a NumPy array or compatible type
        return var


def to_tensor(var, device):
    if isinstance(var, np.ndarray):
        # Convert NumPy array to PyTorch tensor
        return torch.tensor(var, device=device)
    else:
        # Assume it's already a PyTorch tensor or compatible type
        return var.to(device)
