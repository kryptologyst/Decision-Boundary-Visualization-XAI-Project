"""Utility functions for device management and deterministic seeding."""

import os
import random
from typing import Optional, Union

import numpy as np
import torch


def set_deterministic_seed(seed: int = 42) -> None:
    """Set deterministic seeds for all random number generators.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional determinism
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        PyTorch device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_name() -> str:
    """Get human-readable device name.
    
    Returns:
        Device name string.
    """
    device = get_device()
    if device.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name()})"
    elif device.type == "mps":
        return "Apple Silicon (MPS)"
    else:
        return "CPU"
