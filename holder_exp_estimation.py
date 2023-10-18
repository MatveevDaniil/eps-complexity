import numpy as np
from typing import Tuple
from scipy.interpolate import splrep, splev


def L1(x: np.array, y: np.array) -> np.array:
    return np.sum(abs(x - y))

def L2(x: np.array, y: np.array) -> np.array:
    return ((x - y) ** 2).sum() ** 0.5

def get_S_k(t: np.array, k: int) -> np.array:
    return t.reshape((-1, k)).T

def calc_eps_on_S_k(t: np.array, k: int, x: callable, norm: callable = L1) -> float:
    S_k = get_S_k(t, k)
    errors = []
    x_S_k = x(S_k)
    for i in range(k):
        error_i = float('inf')
        for deg in (1, 2, 3, 4):
            spl = splrep(S_k[i], x_S_k[i], k=deg)

            dropped_approx = splev(np.delete(S_k, i, axis=0), spl)
            dropped_value = np.delete(x_S_k, i, axis=0)
            error_i = min(error_i, norm(dropped_approx, dropped_value))

        errors.append(error_i)
    return np.mean(errors)

def calc_A_B(t: np.array, x: callable, norm: callable = L1) -> Tuple[float]:
    log_S, log_eps = [], []
    for k in [2, 3, 4, 5, 6, 10]:
        log_S.append(np.log(1 / k))
        log_eps.append(np.log(calc_eps_on_S_k(t, k, x, norm)))
    _log_S = np.vstack([log_S, np.ones_like(log_S)]).T
    (B, A), _, _, _ = np.linalg.lstsq(_log_S, log_eps, rcond=None)
    return (A, B)

def limit_A(x: callable, grid_steps: Tuple[float], norm: callable = L1):
    collect_A = []
    collect_B = []
    for h in grid_steps:
        t = np.arange(0, 120, step=h)
        n = len(t)
        A, B = calc_A_B(t, x, norm)
        collect_A.append(- A / np.log(n))
        collect_B.append(B)
    return collect_A, collect_B