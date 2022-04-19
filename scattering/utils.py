"""
File containing basic functions
"""

import numpy as np
import torch


def to_numpy(data):
    """Converts a tensor/array/list to numpy array. Recurse over dictionaries and tuples. Values are left as-is."""
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, list):
        return np.array(data)
    elif isinstance(data, dict):
        return {k: to_numpy(v) for k, v in data.items()}
    elif isinstance(data, tuple):
        return tuple(to_numpy(v) for v in data)
    return data
