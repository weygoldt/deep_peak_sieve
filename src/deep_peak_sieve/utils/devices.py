"""
Check availability of CUDA-enabled GPU and return the correct device for PyTorch.
"""

import torch


def get_device() -> torch.device:
    """Check if a CUDA-enabled GPU is available, and return the correct device.

    Returns
    -------
    - `device`: `torch.device`
        The device to use for PyTorch computations. If a CUDA-enabled GPU is
        available, returns a device object
        representing that GPU. If an Apple M1 GPU is available, returns a
        device object representing that GPU.
        Otherwise, returns a device object representing the CPU.
    """
    if torch.cuda.is_available() is True:
        device = torch.device("cuda")  # nvidia / amd gpu
    elif torch.backends.mps.is_available() is True:
        device = torch.device("mps")  # apple m1 gpu
    else:
        device = torch.device("cpu")  # no gpu
    return device
