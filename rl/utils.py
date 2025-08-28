
from __future__ import annotations
import os, time, json
from dataclasses import dataclass, asdict
import numpy as np

class Normalizer:
    def __init__(self, dim: int, eps: float = 1e-8):
        self.dim = dim
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.eps = eps
    def update_batch(self, x: np.ndarray):
        # x shape: (B, dim)
        if x.size == 0:
            return
        self.n += x.shape[0]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        # Running average (simple)
        alpha = x.shape[0] / max(self.n, 1)
        self.mean = (1 - alpha) * self.mean + alpha * batch_mean
        self.var = (1 - alpha) * self.var + alpha * batch_var
    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + self.eps)

@dataclass
class Checkpoint:
    iteration: int
    theta: np.ndarray
    config: dict

class Saver:
    def __init__(self, dirpath: str = 'rl/checkpoints'):
        self.dir = dirpath
        os.makedirs(self.dir, exist_ok=True)
    def save(self, iteration: int, theta: np.ndarray, config: dict, name: str = None) -> str:
        if name is None:
            name = f'iter_{iteration:06d}'
        path = os.path.join(self.dir, name + '.npz')
        np.savez(path, iteration=iteration, theta=theta)
        with open(path + '.json', 'w') as f:
            json.dump(config, f, indent=2)
        latest = os.path.join(self.dir, 'latest.npz')
        try:
            if os.path.islink(latest) or os.path.exists(latest):
                os.remove(latest)
        except Exception:
            pass
        try:
            os.symlink(os.path.basename(path), latest)
        except Exception:
            # on Windows or restricted env, skip symlink
            pass
        return path
