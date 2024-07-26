import random
import numpy as np
import torch


def set_seeds(seed, cuda_deterministic=False):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.backends.mps.is_available():
            # No hay configuraciones específicas de MPS para determinismo similar a CUDA
            pass