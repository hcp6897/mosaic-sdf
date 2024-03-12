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
    if isinstance(var, np.ndarray) or isinstance(var, list):
        # Convert NumPy array to PyTorch tensor
        return torch.tensor(var, device=device)
    else:
        # Assume it's already a PyTorch tensor or compatible type
        return var.to(device)


def process_in_batches(tensor, func, batch_size):
    """
    Slices a tensor into batches, applies a function to each batch, and concatenates the results.

    Args:
    - tensor (Tensor): The input tensor to be processed in batches.
    - func (callable): The function to apply to each batch.
    - batch_size (int): The size of each batch.

    Returns:
    - Tensor: The concatenated result of applying the function to each batch of the input tensor.
    """
    # Move tensor to the specified device
    # tensor = tensor.to(device)
    # Initialize an empty list to hold processed batches
    results = []

    # Calculate the number of batches
    num_batches = len(tensor) // batch_size + (len(tensor) % batch_size > 0)

    for i in range(num_batches):
        # Determine batch indices
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(tensor))
        
        # Extract batch of grid points
        batch = tensor[start_idx:end_idx]
        
        # Apply the function to the batch
        batch_result = func(batch)
        
        # Store the result
        results.append(batch_result)

    # Concatenate all batch results into a single tensor
    return torch.cat(results, dim=0)


def min_l2_distance(tensor):
    diffs = tensor.unsqueeze(1) - tensor.unsqueeze(0)  # Pairwise subtraction
    dists = torch.sqrt((diffs ** 2).sum(-1))  # Squared L2 norm
    
    inf_diag = torch.diag(torch.full((tensor.size(0),), float('inf'))).to(tensor.device)
    dists += inf_diag

    min_dists, _ = dists.min(dim=1)
    
    return min_dists
