import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    spectral_norm = np.linalg.norm(W_hh, ord=2)
    gradient_norms = []
    current_norm = 1.0
    for _ in range(T):
        gradient_norms.append(float(current_norm))
        current_norm *= spectral_norm
    return gradient_norms